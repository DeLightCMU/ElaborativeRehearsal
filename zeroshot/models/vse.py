import os
import itertools
import numpy as np
import collections
import json

import torch
import torch.nn as nn

import framework.trainer

import zeroshot.modules.video
import zeroshot.modules.text
import zeroshot.models.criterions
from zeroshot.models.metrics import eval_precisions


class PrecompVSEModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.vis_encoder = zeroshot.modules.video.GlobalFtEncoder(
      self.config.vis_encoder.dim_input, self.config.vis_encoder.dim_embed, 
      self.config.vis_encoder.dropout, self.config.vis_encoder.l2norm)

    self.txt_backbone = getattr(zeroshot.modules.text, self.config.txt_backbone.type)(
      **self.config.txt_backbone.kwargs)

    self.txt_encoder = zeroshot.modules.text.TextEncoder(
      self.config.txt_encoder.dim_input, self.config.txt_encoder.dim_embed, 
      self.config.txt_encoder.pooling_method,
      self.config.txt_encoder.dropout, self.config.txt_encoder.l2norm)

  def forward_video_embeds(self, vis_fts):
    vis_embeds = self.vis_encoder(vis_fts)
    return vis_embeds

  def forward_text_embeds(self, txt_ids, txt_masks):
    txt_hiddens = self.txt_backbone(txt_ids, txt_masks)
    txt_embeds = self.txt_encoder(txt_hiddens, txt_masks)
    return txt_embeds

  def forward(self, inputs):
    outs = {}
    outs['vis_embeds'] = self.forward_video_embeds(inputs['vis_fts'])
    outs['txt_embeds'] = self.forward_text_embeds(inputs['txt_ids'], inputs['txt_masks'])
    return outs


class PrecompVSEModelHelper(framework.trainer.TrnTstHelper):
  def build_model(self):
    return PrecompVSEModel(self.config.model)

  def build_loss(self):
    if self.config.trainer.loss.type == 'ce':
      criterion = nn.CrossEntropyLoss()
    elif self.config.trainer.loss.type == 'l2':
      criterion = nn.MSELoss()
    else:
      criterion = zeroshot.models.criterions.ClsContrastiveLoss(
        **self.config.trainer.loss.kwargs)
    return criterion

  def generate_scores(self, vis_embeds, class_embeds):
    # compute video-text similarity
    scores = torch.einsum('bd,cd->bc', vis_embeds, class_embeds)
    return scores

  def prepare_batch_inputs(self, batch_data):
    inputs = {}
    for key, value in batch_data.items():
      if isinstance(value, torch.Tensor):
        inputs[key] = value.to(self.device, non_blocking=True)
    return inputs

  def train_one_step(self, batch_data, step=None):
    inputs = self.prepare_batch_inputs(batch_data)
    outs = self.model(inputs)
    targets = inputs['targets']
    logits = self.generate_scores(outs['vis_embeds'], outs['txt_embeds'])

    if self.config.trainer.loss.type == 'l2':
      target_txt_embeds = outs['txt_embeds'][targets]
      loss = self.criterion(outs['vis_embeds'], target_txt_embeds)
    elif self.config.trainer.loss.type == 'ce':
      loss = self.criterion(logits / self.config.trainer.loss.kwargs.temporature, targets)
    else:
      loss = self.criterion(logits, targets)
    
    if self.is_main_process and step is not None and self.config.trainer.monitor_iter > 0 \
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
    outs = self.model(inputs)
    logits = self.generate_scores(outs['vis_embeds'], outs['txt_embeds'])

    return {
      'logits': logits.data.cpu(),
      'targets': batch_data['targets'],
    }

  def val_epoch_end(self, val_outs):
    logits = torch.cat([x['logits'] for x in val_outs], 0)
    targets = torch.cat([x['targets'] for x in val_outs], 0)

    if self.world_size > 1:
      logits = torch.cat(framework.dist_helper.all_gather(logits), 0)
      targets = torch.cat(framework.dist_helper.all_gather(targets), 0)

    if self.is_main_process:
      acc = eval_precisions(logits, targets, topk=(1, 5))
      metrics = {'prec@1': acc[0], 'prec@5': acc[1]}
      return metrics

  def test_one_step(self, batch_data):
    inputs = self.prepare_batch_inputs(batch_data)
    outs = self.model(inputs)
    logits = self.generate_scores(outs['vis_embeds'], outs['txt_embeds'])

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

