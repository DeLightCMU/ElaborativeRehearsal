import os
import numpy as np
import itertools
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.trainer
import framework.dist_helper

import zeroshot.models.criterions
from zeroshot.models.metrics import eval_precisions


class DeviseModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    wcfg = self.config.bilinear
    self.bilinear = nn.Linear(wcfg.dim_vis_ft, wcfg.dim_txt_ft, bias=False)
    self.dropout = nn.Dropout(self.config.dropout)

  def forward(self, vis_fts, txt_fts):
    if self.config.txt_norm:
      txt_fts = F.normalize(txt_fts, p=2, dim=1)
    vis_fts = self.dropout(vis_fts)
    txt_fts = self.dropout(txt_fts)
    logits = torch.matmul(self.bilinear(vis_fts), txt_fts.t())
    return logits


class DeviseModelHelper(framework.trainer.TrnTstHelper):
  def build_model(self):
    return DeviseModel(self.config.model)

  def build_loss(self):
    criterion = zeroshot.models.criterions.ClsContrastiveLoss(
      **self.config.trainer.loss)
    return criterion

  def prepare_batch_inputs(self, batch_data):
    inputs = {}
    for key, value in batch_data.items():
      if isinstance(value, torch.Tensor):
        inputs[key] = value.to(self.device, non_blocking=True)
    return inputs

  def train_one_step(self, batch_data, step=None):
    inputs = self.prepare_batch_inputs(batch_data)

    logits = self.model(inputs['vis_fts'], inputs['txt_fts'])
    targets = inputs['targets']
    loss = self.criterion(logits, targets)

    if self.rank == 0 and step is not None and self.config.trainer.monitor_iter > 0 \
        and step % self.config.trainer.monitor_iter == 0:
      pos_scores = torch.gather(logits, 1, targets.unsqueeze(1))
      pos_masks = torch.zeros_like(logits).bool()
      pos_masks.scatter_(1, targets.unsqueeze(1), True)
      neg_scores = logits.masked_fill(pos_masks, -float('inf'))
      self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores %.2f'%(
        step, torch.mean(pos_scores), torch.mean(torch.max(neg_scores, 1)[0])))
    return loss

  def val_one_step(self, batch_data):
    inputs = self.prepare_batch_inputs(batch_data)
    logits = self.model(inputs['vis_fts'], inputs['txt_fts'])
    loss = self.criterion(logits, inputs['targets'])

    return {
      'loss': loss.data.item(),
      'logits': logits.data.cpu(),
      'targets': batch_data['targets']
    }

  def val_epoch_end(self, val_outs):
    loss = np.mean([x['loss'] for x in val_outs])
    logits = torch.cat([x['logits'] for x in val_outs], 0)
    targets = torch.cat([x['targets'] for x in val_outs], 0)

    if self.world_size > 1:
      loss = framework.dist_helper.all_gather(loss)
      logits = torch.cat(framework.dist_helper.all_gather(logits), 0)
      targets = torch.cat(framework.dist_helper.all_gather(targets), 0)

    if self.is_main_process:
      metrics = collections.OrderedDict()
      metrics['loss'] = np.mean(loss)
      acc = eval_precisions(logits, targets, topk=(1, 5))
      metrics.update({'prec@1': acc[0], 'prec@5': acc[1]})
      return metrics

  def test_one_step(self, batch_data):
    inputs = self.prepare_batch_inputs(batch_data)
    logits = self.model(inputs['vis_fts'], inputs['txt_fts'])

    outs = {
      'names': batch_data['names'],
      'logits': logits.data.cpu(),
    }
    if 'targets' in batch_data:
      outs['targets'] = batch_data['targets']
    return outs

  def test_epoch_end(self, tst_outs, tst_pred_file):
    names = list(itertools.chain(*[x['names'] for x in tst_outs]))
    logits = torch.cat([x['logits'] for x in tst_outs], 0)
    has_targets = 'targets' in tst_outs[0]
    if has_targets:
      targets = torch.cat([x['targets'] for x in tst_outs], 0)

    if self.world_size > 1:
      names = framework.dist_helper.all_gather(names)
      logits = torch.cat(framework.dist_helper.all_gather(logits), 0)
      if has_targets:
        targets = torch.cat(framework.dist_helper.all_gather(targets), 0)

    if self.is_main_process:
      if has_targets:
        acc = eval_precisions(logits, targets, topk=(1, 5))
        metrics = {'prec@1': acc[0], 'prec@5': acc[1]}
      else:
        metrics = None

      with open(tst_pred_file, 'wb') as outf:
        np.save(outf, {'names': names, 'logits': logits.numpy()})
      return metrics


class DEModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    wcfg = self.config.t2vfc
    self.t2vfc = nn.Sequential(
      nn.Linear(wcfg.dim_txt_ft, wcfg.hidden_size, bias=False),
      nn.ReLU(),
      nn.Linear(wcfg.hidden_size, wcfg.dim_vis_ft, bias=False),
      nn.ReLU()
    )
    self.dropout = nn.Dropout(self.config.dropout)

  def forward(self, txt_fts):
    txt_embeds = self.t2vfc(self.dropout(txt_fts))
    return txt_embeds

class DEModelHelper(DeviseModelHelper):
  def build_model(self):
    return DEModel(self.config.model)

  def build_loss(self):
    criterion = nn.MSELoss()
    return criterion

  def train_one_step(self, batch_data, step=None):
    inputs = self.prepare_batch_inputs(batch_data)
    txt_embeds = self.model(inputs['txt_fts'][inputs['targets']])
    loss = self.criterion(inputs['vis_fts'], txt_embeds)
    return loss

  def val_one_step(self, batch_data):
    inputs = self.prepare_batch_inputs(batch_data)
    txt_embeds = self.model(inputs['txt_fts'])
    loss = self.criterion(inputs['vis_fts'], txt_embeds[inputs['targets']])
    dists = torch.sum((inputs['vis_fts'].unsqueeze(1) - txt_embeds.unsqueeze(0))**2, 2)

    return {
      'loss': loss.data.item(),
      'logits': - dists.data.cpu(),
      'targets': batch_data['targets']
    }

  def test_one_step(self, batch_data):
    inputs = self.prepare_batch_inputs(batch_data)
    txt_embeds = self.model(inputs['txt_fts'])
    dists = torch.sum((inputs['vis_fts'].unsqueeze(1) - txt_embeds.unsqueeze(0))**2, 2)
    
    outs = {
      'names': batch_data['names'],
      'logits': - dists.data.cpu(),
    }
    if 'targets' in batch_data:
      outs['targets'] = batch_data['targets']
    return outs


class ESZSLModelHelper(object):
  '''
  Closed form solution without the common gradient descent optimization
  '''
  def __init__(self, config):
    self.config = config

  def closed_form_optimization(self, X, Y, S, alpha=3, gamma=0):
    '''
    Args:
      X: vis_fts, (dimX, ndata)
      Y: targets, (ndata, nclass) (-1 or 1)
      S: semantic embeds, (dimS, nclass)
    '''
    dimX = X.shape[0]
    dimS = S.shape[0]
    part1 = np.linalg.pinv(np.matmul(X, X.transpose()) + (10**alpha) * np.eye(dimX))
    part0 = np.matmul(np.matmul(X, Y), S.transpose())
    part2 = np.linalg.pinv(np.matmul(S, S.transpose()) + (10**gamma) * np.eye(dimS))
    W = np.matmul(np.matmul(part1, part0), part2)
    return W

  def train(self, trn_dataset, val_dataset, resume_file=None):
    import torch.utils.data

    trn_loader = torch.utils.data.DataLoader(dataset=trn_dataset, 
      batch_size=self.config.trainer.batch_size, 
      num_workers=self.config.dataset.num_workers,
      shuffle=True, collate_fn=trn_dataset.collate_fn, pin_memory=True)
    trn_fts, trn_targets = [], []
    for batch_data in trn_loader:
      trn_fts.append(batch_data['vis_fts'].numpy())
      trn_targets.append(batch_data['targets'].numpy())
    trn_vis_fts = np.concatenate(trn_fts, 0).transpose()
    trn_class_embeds = F.normalize(trn_dataset.label_embeds, p=2, dim=1).numpy().transpose()
    trn_targets = np.concatenate(trn_targets, 0)
    # 0 or -1
    trn_target_matrix = np.zeros((trn_vis_fts.shape[1], trn_class_embeds.shape[1])) 
    trn_target_matrix[np.arange(trn_vis_fts.shape[1]), trn_targets] = 1

    # (dimX, dimS)
    W = self.closed_form_optimization(trn_vis_fts, trn_target_matrix, trn_class_embeds)
    print(W.shape)
    with open(os.path.join(self.config.saver.model_dir, 'W.npy'), 'wb') as outf:
      np.save(outf, W)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
      batch_size=self.config.evaluate.batch_size,
      num_workers=self.config.dataset.num_workers,
      shuffle=False, collate_fn=val_dataset.collate_fn, pin_memory=True)
    val_fts, val_targets = [], []
    for batch_data in val_loader:
      val_fts.append(batch_data['vis_fts'].numpy())
      val_targets.append(batch_data['targets'].numpy())
    val_vis_fts = np.concatenate(val_fts, 0) # (ndata, dimX)
    val_class_embeds = F.normalize(val_dataset.label_embeds, p=2, dim=1).numpy().transpose() # (dimS, nclass)
    val_targets = np.concatenate(val_targets, 0)
    val_logits = np.matmul(np.matmul(val_vis_fts, W), val_class_embeds)

    acc = eval_precisions(torch.FloatTensor(val_logits), torch.LongTensor(val_targets), topk=(1,5))
    print('val: prec@1 %.2f prec@5 %.2f' % (acc[0], acc[1]))

  def test(self, tst_dataset, tst_pred_file, tst_model_file=None):
    W = np.load(os.path.join(self.config.saver.model_dir, 'W.npy'))
    
    tst_loader = torch.utils.data.DataLoader(dataset=tst_dataset,
      batch_size=self.config.evaluate.batch_size,
      num_workers=self.config.dataset.num_workers,
      shuffle=False, collate_fn=tst_dataset.collate_fn, pin_memory=True)
    tst_fts, tst_targets = [], []
    for batch_data in tst_loader:
      tst_fts.append(batch_data['vis_fts'].numpy())
      tst_targets.append(batch_data['targets'].numpy())
    tst_vis_fts = np.concatenate(tst_fts, 0) # (ndata, dimX)
    tst_class_embeds = F.normalize(tst_dataset.label_embeds, p=2, dim=1).numpy().transpose() # (dimS, nclass)
    tst_targets = np.concatenate(tst_targets, 0)
    tst_logits = np.matmul(np.matmul(tst_vis_fts, W), tst_class_embeds)

    acc = eval_precisions(torch.FloatTensor(tst_logits), torch.LongTensor(tst_targets), topk=(1,5))
    metrics = {'prec@1': acc[0], 'prec@5': acc[1]}
    return metrics
