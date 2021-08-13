import os
import time
import datetime
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed

import framework.log_helper as log_helper
import framework.lr_helper as lr_helper
import framework.dist_helper as dist_helper


class TrnTstHelper(object):
  def __init__(self, config):
    self.config = config
    
    self.model = self.build_model()
    self.criterion = self.build_loss()
      
  def build_model(self):
    raise NotImplementedError('implement build_model function: return model')

  def build_loss(self):
    raise NotImplementedError('implement build_loss function: return criterion')

  def train_one_step(self, batch_data, step=None):
    raise NotImplementedError('implement train_one_step function: return loss')

  def val_one_step(self, batch_data):
    raise NotImplementedError('implement val_one_step function: return list')

  def val_epoch_end(self, val_outs):
    raise NotImplementedError('implement val_epoch function: return metrics(dict)')

  def test_one_step(self, batch_data):
    raise NotImplementedError('implement test_one_step function: return list')

  def test_epoch_end(self, tst_outs, tst_pred_file):
    raise NotImplementedError('implement test_epoch_end function: return metrics(dict)')
    
  
  ########################## boilerpipe functions ########################
  def _setup_env(self, is_train=False):
    # set random seed: keep the same initialization across processes
    np.random.seed(self.config.get('seed', 2020))
    torch.manual_seed(self.config.get('seed', 2020))

    self.rank = dist_helper.get_rank()
    self.world_size = dist_helper.get_world_size()
    self.is_main_process = dist_helper.is_main_process()
    self.device = torch.cuda.current_device()

    self.model = self.model.to(self.device)

    self.print_fn = print

    if self.is_main_process:
      if is_train:
        log_dir = self.config.saver.log_dir
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_file = os.path.join(log_dir, 'log-' + timestamp)
        _logger = log_helper.set_logger(self.config.saver.log_file, timestamp)
        self.print_fn = _logger.info
        self.tf_logger = log_helper.TensorboardLogger(log_dir)
    
      if self.config.verbose:
        self.print_model_weights()

  def print_model_weights(self):
    num_params, num_weights = 0, 0
    for varname, varvalue in self.model.state_dict().items():
      n_varvalue = np.prod(varvalue.size())
      self.print_fn('%s, shape=%s, num:%d' % (varname, str(varvalue.size()), n_varvalue))
      num_params += 1
      num_weights += n_varvalue
    self.print_fn('num params %d, num weights %d'%(num_params, num_weights))

  def build_optimizer(self):
    trn_params = []
    trn_param_ids = set()
    per_param_opts = []

    base_lr = self.config.trainer.optimizer.kwargs.lr

    for key, submod in self.model.named_children():
      if len(list(submod.parameters())) == 0:
        continue
      subcfg = self.config.model[key]
      sub_freeze = subcfg.get('freeze', False)
      if sub_freeze:
        for param in submod.parameters():
          param.requires_grad = False
      else:
        params = []
        for param in submod.parameters():
          # sometimes we share params in different submods
          if param.requires_grad and id(param) not in trn_param_ids:
            params.append(param)
            trn_param_ids.add(id(param))
        per_param_opts.append({
          'params': params, 
          'lr': base_lr * subcfg.get('lr_mult', 1),
          'weight_decay': subcfg.get('weight_decay', 0),
          })
        trn_params.extend(params)

    if len(trn_params) > 0:
      optimizer = getattr(torch.optim, self.config.trainer.optimizer.type)(
        per_param_opts, **self.config.trainer.optimizer.kwargs)
      lr_scheduler = lr_helper.build_scheduler(
        self.config.trainer.lr_scheduler, optimizer, base_lr, 
        self.config.trainer.batch_size * self.world_size)
    else:
      raise NotImplementedError('no parameters to train')

    return trn_params, optimizer, lr_scheduler

  def save_checkpoint(self, ckpt_file, last_save_all=False):
    state_dict = {}
    for key, value in self.model.state_dict().items():
      if self.world_size > 1:
        assert key.startswith('module.')
        key = key[7:]
      state_dict[key] = value.cpu()
    torch.save(state_dict, ckpt_file)

    if last_save_all:
      last_ckpt_file = os.path.join(os.path.dirname(ckpt_file), 'last.ckpt.pth.tar')
      torch.save({
        'optimizer': self.optimizer.state_dict(),
        'lr_scheduler': self.lr_scheduler.state_dict(),
        'resume_model': state_dict
        }, last_ckpt_file)

  def load_checkpoint(self, ckpt_file, resume_training=False):
    state_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)

    if 'resume_model' in state_dict:
      if resume_training:
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
      state_dict = state_dict['resume_model']

    own_state_dict = self.model.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
      if key in own_state_dict:
        new_state_dict[key] = value
    own_state_dict.update(new_state_dict)
    self.model.load_state_dict(own_state_dict, strict=True)
    return len(new_state_dict)
      
  def pretty_print_metrics(self, prefix, metrics):
    metric_str = []
    for measure, score in metrics.items():
      metric_str.append('%s %.6f' % (measure, score))
    metric_str = ', '.join(metric_str)
    if self.is_main_process:
      self.print_fn('%s: %s' % (prefix, metric_str))

  def get_current_base_lr(self):
    lrs = []
    for x in self.optimizer.param_groups:
      if x['lr'] not in lrs:
        lrs.append(x['lr'])
    return lrs

  def train(self, trn_dataset, val_dataset, resume_file=None):
    self._setup_env(is_train=True)

    self.params, self.optimizer, self.lr_scheduler = self.build_optimizer()
    if self.is_main_process:
      self.print_fn('trainable: num params %d, num weights %d'%(
        len(self.params), sum([np.prod(param.size()) for param in self.params])))

    if resume_file is not None:
      if isinstance(resume_file, list):
        num_resumed_params = 0
        for single_resume_file in resume_file:
          num_resumed_params += self.load_checkpoint(single_resume_file, 
            resume_training=single_resume_file.endswith('tar'))
      else:
        num_resumed_params = self.load_checkpoint(resume_file, resume_training=resume_file.endswith('tar'))
      if self.is_main_process:
        self.print_fn('number of resumed variables: %d' % num_resumed_params)

    if self.config.fp16_opt is not None:
      from apex import amp
      self.model, self.optimizer = amp.initialize(
        self.model, self.optimizer, opt_level='O%d'%self.config.fp16_opt) 

    if self.world_size > 1:
      if self.config.fp16_opt is None:
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])
      else:
        from apex.parallel import DistributedDataParallel as DDP
        self.model = DDP(self.model)
      trn_sampler = torch.utils.data.distributed.DistributedSampler(
        trn_dataset, num_replicas=self.world_size, rank=self.rank)
      val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=self.world_size, rank=self.rank)
    else:
      trn_sampler = val_sampler = None

    trn_loader = torch.utils.data.DataLoader(dataset=trn_dataset, 
      batch_size=self.config.trainer.batch_size, 
      num_workers=self.config.dataset.num_workers,
      sampler=trn_sampler, shuffle=(False if trn_sampler else True),
      pin_memory=True, collate_fn=trn_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
      batch_size=self.config.evaluate.batch_size,
      num_workers=self.config.dataset.num_workers,
      sampler=val_sampler, shuffle=False,
      pin_memory=True, collate_fn=val_dataset.collate_fn)

    if self.is_main_process:
      self.print_fn('# data: trn %d, val %d' % (len(trn_dataset), len(val_dataset)))
      self.print_fn('# batch_per_epoch: trn %d, val %d' % (len(trn_loader), len(val_loader)))

    # first validate
    metrics = self.validate(val_loader)
    if self.is_main_process:
      self.pretty_print_metrics('init val', metrics)
    
    # training
    step = 0
    for epoch in range(self.config.trainer.num_epochs):
      torch.set_grad_enabled(True)
      self.model.train()

      avg_loss, n_batches = 0, 0
      for batch_data in trn_loader:
        self.optimizer.zero_grad()

        loss = self.train_one_step(batch_data, step=step)

        if self.config.fp16_opt:
          with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        else:
          loss.backward()

        if self.config.trainer.clip_gradient > 0:
          if self.config.fp16_opt is None:
            nn.utils.clip_grad_norm_(self.params, self.config.trainer.clip_gradient)
          else:
            nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.config.trainer.clip_gradient)
        
        self.optimizer.step()
        self.lr_scheduler.step()

        loss_value = loss.data.item()
        if self.is_main_process and self.config.trainer.monitor_iter > 0 and step % self.config.trainer.monitor_iter == 0:
          self.print_fn('\ttrn step %d lr %s %s: %.6f' % (step, 
            ', '.join(['%.8f'%x for x in self.get_current_base_lr()]), 
            'loss', loss_value))
        if self.is_main_process and self.config.trainer.summary_iter > 0 and step % self.config.trainer.summary_iter == 0:
          self.tf_logger.scalar_summary('trn_loss', loss_value, step)
          self.tf_logger.scalar_summary('trn_lr', self.get_current_base_lr()[0], step)

        avg_loss += loss_value
        n_batches += 1
        step += 1

        if self.is_main_process and self.config.trainer.save_iter > 0 and step % self.config.trainer.save_iter == 0:
          self.save_checkpoint(os.path.join(self.config.saver.model_dir, 'epoch.%d.step.%d.th'%(epoch, step)), last_save_all=True)
        
        if (self.config.trainer.save_iter > 0 and step % self.config.trainer.save_iter == 0) \
            or (self.config.trainer.val_iter > 0 and step % self.config.trainer.val_iter == 0):
          self.validate(val_loader, epoch=epoch, step=step, log=True)
          torch.set_grad_enabled(True)
          self.model.train()

      if self.is_main_process:
        avg_loss /= n_batches
        self.pretty_print_metrics('epoch (%d/%d) trn'%(epoch, self.config.trainer.num_epochs), {'loss': avg_loss})
        if self.config.trainer.save_per_epoch:
          self.save_checkpoint(os.path.join(self.config.saver.model_dir, 'epoch.%d.step.%d.th'%(epoch, step)), last_save_all=True)
      
      if self.config.trainer.val_per_epoch:
        self.validate(val_loader, epoch=epoch, step=step, log=True)

  def validate(self, val_loader, epoch=None, step=None, log=False):
    torch.set_grad_enabled(False)
    self.model.eval()

    outs = []
    for batch_data in val_loader:
      out = self.val_one_step(batch_data)
      outs.append(out)

    metrics = self.val_epoch_end(outs)

    if log and self.is_main_process:
      with open(os.path.join(self.config.saver.log_dir, 'val.epoch.%d.step.%d.json' % (epoch, step)), 'w') as f:
        json.dump(metrics, f, indent=2)
      self.pretty_print_metrics('val step %d'%step, metrics)
      # Write validation result into summary
      for metric_name, metric_score in metrics.items():
        self.tf_logger.scalar_summary('val_%s'%metric_name, metric_score, step)

    return metrics

  def test(self, tst_dataset, tst_pred_file, tst_model_file=None):
    self._setup_env()

    if tst_model_file is not None:
      self.load_checkpoint(tst_model_file)

    torch.set_grad_enabled(False)
    self.model.eval()

    tst_loader = torch.utils.data.DataLoader(dataset=tst_dataset,
      batch_size=self.config.evaluate.batch_size,
      num_workers=self.config.dataset.num_workers,
      pin_memory=True, shuffle=False, collate_fn=tst_dataset.collate_fn)
    self.print_fn('# data: %d, # batch %d' % (len(tst_dataset), len(tst_loader)))

    outs = []
    for batch_data in tst_loader:
      out = self.test_one_step(batch_data)
      outs.append(out)

    metrics = self.test_epoch_end(outs, tst_pred_file)

    return metrics
    

