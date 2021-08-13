import torch
import torch.nn as nn
import torch.nn.functional as F

import zeroshot.models.vse


class PrecompERVSEModel(zeroshot.models.vse.PrecompVSEModel):
  def __init__(self, config):
    super().__init__(config)
    if self.config.fusion_type == 2:
      self.mm_fusion = nn.Sequential(
        nn.Linear(512 * 2, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512))
    elif self.config.fusion_type == 3:
      self.act_gate_fc = nn.Sequential(
        nn.Linear(512 * 2, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512), nn.Sigmoid())
      self.obj_gate_fc = nn.Sequential(
        nn.Linear(512 * 2, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512), nn.Sigmoid())

  def forward(self, inputs, ercpt_inputs=None):
    outs = super().forward(inputs)
    if self.config.comb_obj_stream:
      outs['vis_cpt_embeds'] = self.forward_text_embeds(inputs['vis_cpt_ids'], inputs['vis_cpt_masks'])
    if self.config.fusion_type == 2:
      outs['mm_embeds'] = F.normalize(self.mm_fusion(torch.cat([outs['vis_embeds'], outs['vis_cpt_embeds']], 1)), p=2, dim=1)
    if self.config.fusion_type == 3:
      act_gates = self.act_gate_fc(torch.cat([outs['vis_embeds'], outs['vis_cpt_embeds']], 1))
      obj_gates = self.obj_gate_fc(torch.cat([outs['vis_embeds'], outs['vis_cpt_embeds']], 1))
      outs['vis_embeds'] = F.normalize(act_gates * outs['vis_embeds'], dim=1, p=2)
      outs['vis_cpt_embeds'] = F.normalize(obj_gates * outs['vis_cpt_embeds'], dim=1, p=2)
    if ercpt_inputs:
      outs['er_cpt_embeds'] = self.forward_text_embeds(inputs['er_cpt_ids'], inputs['er_cpt_masks'])
    return outs

class PrecompERVSEModelHelper(zeroshot.models.vse.PrecompVSEModelHelper):
  def build_model(self):
    return PrecompERVSEModel(self.config.model)

  def train_one_step(self, batch_data, step=None):
    inputs = self.prepare_batch_inputs(batch_data)
    outs = self.model(inputs, ercpt_inputs=inputs if self.config.trainer.loss.er_loss_weight else None)
    vis_embeds = outs['vis_embeds'] + outs['vis_cpt_embeds']
    logits = self.generate_scores(vis_embeds, outs['txt_embeds'])

    targets = inputs['targets']
    if self.config.trainer.loss.type == 'ce':
      loss = self.criterion(logits / self.config.trainer.loss.kwargs.temporature, targets)
    else:
      loss = self.criterion(logits, targets)

    is_monitor_step = self.is_main_process and step is not None and \
      self.config.trainer.monitor_iter > 0 and step % self.config.trainer.monitor_iter == 0

    if self.config.trainer.loss.er_loss_weight > 0:
      er_cpt_embeds = outs['er_cpt_embeds'] # (ncpts, dim)
      er_cpt_ml_targets = inputs['er_cpt_ml_targets'] # (batch, ncpts)
      er_cpt_logits = self.generate_scores(vis_embeds, er_cpt_embeds)
      if self.config.trainer.loss.type == 'ce':
        er_cpt_logits = er_cpt_logits / self.config.trainer.loss.kwargs.temporature
      er_cpt_loss = er_cpt_logits - torch.logsumexp(er_cpt_logits, 1, keepdim=True)
      er_cpt_loss = - torch.mean(er_cpt_loss[er_cpt_ml_targets])
      if is_monitor_step:
        self.print_fn('\tstep %d: act_clf_loss %.6f (%d), er_cpt_loss %.6f (%d)' % (
          step, loss.data.item(), logits.size(1),
          er_cpt_loss.data.item(), er_cpt_ml_targets.size(1)))
      loss = loss + self.config.trainer.loss.er_loss_weight * er_cpt_loss

    if is_monitor_step:
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
    vis_embeds = outs['vis_embeds'] + outs['vis_cpt_embeds']
    logits = self.generate_scores(vis_embeds, outs['txt_embeds'])

    return {
      'logits': logits.data.cpu(),
      'targets': batch_data['targets'],
    }

  def test_one_step(self, batch_data):
    inputs = self.prepare_batch_inputs(batch_data)
    outs = self.model(inputs)
    vis_embeds = outs['vis_embeds'] + outs['vis_cpt_embeds']
    logits = self.generate_scores(vis_embeds, outs['txt_embeds'])

    outs = {
      'names': batch_data['names'],
      'logits': logits.data.cpu(),
    }
    if 'targets' in batch_data:
      outs['targets'] = batch_data['targets']
    return outs


class PrecompERVSEModelLFHelper(zeroshot.models.vse.PrecompVSEModelHelper):
  def build_model(self):
    return PrecompERVSEModel(self.config.model)

  def train_one_step(self, batch_data, step=None):
    inputs = self.prepare_batch_inputs(batch_data)
    outs = self.model(inputs, ercpt_inputs=inputs if self.config.trainer.loss.er_loss_weight else None)
    
    act_logits = self.generate_scores(outs['vis_embeds'], outs['txt_embeds'])
    obj_logits = self.generate_scores(outs['vis_cpt_embeds'], outs['txt_embeds'])
    if self.config.model.fusion_type == 2:
      logits = self.generate_scores(outs['mm_embeds'], outs['txt_embeds'])
    else:
      if self.config.model.clamp_obj_logits:
        logits = act_logits + obj_logits.clamp(min=0)
      else:
        logits = act_logits + obj_logits

    targets = inputs['targets']
    if self.config.trainer.loss.type == 'ce':
      act_loss = self.criterion(act_logits / self.config.trainer.loss.kwargs.temporature, targets)
      obj_loss = self.criterion(obj_logits / self.config.trainer.loss.kwargs.temporature, targets)
      loss = self.criterion(logits / self.config.trainer.loss.kwargs.temporature, targets)
      loss = loss + act_loss + obj_loss
    else:
      loss = self.criterion(logits, targets)

    is_monitor_step = self.is_main_process and step is not None and \
      self.config.trainer.monitor_iter > 0 and step % self.config.trainer.monitor_iter == 0

    if self.config.trainer.loss.er_loss_weight > 0:
      er_cpt_embeds = outs['er_cpt_embeds'] # (ncpts, dim)
      er_cpt_ml_targets = inputs['er_cpt_ml_targets'] # (batch, ncpts)
      
      er_cpt_act_logits = self.generate_scores(outs['vis_embeds'], er_cpt_embeds)
      er_cpt_obj_logits = self.generate_scores(outs['vis_cpt_embeds'], er_cpt_embeds)
      if self.config.model.fusion_type == 2:
        er_cpt_logits = self.generate_scores(outs['mm_embeds'], er_cpt_embeds)
      else:
        er_cpt_logits = er_cpt_act_logits + er_cpt_obj_logits

      if self.config.trainer.loss.type == 'ce':
        er_cpt_act_logits = er_cpt_act_logits / self.config.trainer.loss.kwargs.temporature
        er_cpt_obj_logits = er_cpt_obj_logits / self.config.trainer.loss.kwargs.temporature
        er_cpt_logits = er_cpt_logits / self.config.trainer.loss.kwargs.temporature

      er_cpt_act_loss = er_cpt_act_logits - torch.logsumexp(er_cpt_act_logits, 1, keepdim=True)
      er_cpt_act_loss = - torch.mean(er_cpt_act_loss[er_cpt_ml_targets])

      er_cpt_obj_loss = er_cpt_obj_logits - torch.logsumexp(er_cpt_obj_logits, 1, keepdim=True)
      er_cpt_obj_loss = - torch.mean(er_cpt_obj_loss[er_cpt_ml_targets])
     
      er_cpt_loss = er_cpt_logits - torch.logsumexp(er_cpt_logits, 1, keepdim=True)
      er_cpt_loss = - torch.mean(er_cpt_loss[er_cpt_ml_targets])

      er_cpt_loss = er_cpt_loss + er_cpt_act_loss + er_cpt_obj_loss

      if is_monitor_step:
        self.print_fn('\tstep %d: act_clf_loss %.6f (%d), er_cpt_loss %.6f (%d)' % (
          step, loss.data.item(), logits.size(1),
          er_cpt_loss.data.item(), er_cpt_ml_targets.size(1)))
      loss = loss + self.config.trainer.loss.er_loss_weight * er_cpt_loss

    if is_monitor_step:
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
    act_logits = self.generate_scores(outs['vis_embeds'], outs['txt_embeds'])
    obj_logits = self.generate_scores(outs['vis_cpt_embeds'], outs['txt_embeds'])
    if self.config.model.fusion_type == 2:
      logits = self.generate_scores(outs['mm_embeds'], outs['txt_embeds'])
    else:
      if self.config.model.clamp_obj_logits:
        logits = act_logits + obj_logits.clamp(min=0)
      else:
        logits = act_logits + obj_logits

    return {
      'logits': logits.data.cpu(),
      'targets': batch_data['targets'],
    }

  def test_one_step(self, batch_data):
    inputs = self.prepare_batch_inputs(batch_data)
    outs = self.model(inputs)
    act_logits = self.generate_scores(outs['vis_embeds'], outs['txt_embeds'])
    obj_logits = self.generate_scores(outs['vis_cpt_embeds'], outs['txt_embeds'])
    if self.config.model.fusion_type == 2:
      logits = self.generate_scores(outs['mm_embeds'], outs['txt_embeds'])
    else:
      if self.config.model.clamp_obj_logits:
        logits = act_logits + obj_logits.clamp(min=0)
      else:
        logits = act_logits + obj_logits

    outs = {
      'names': batch_data['names'],
      'logits': logits.data.cpu(),
    }
    if 'targets' in batch_data:
      outs['targets'] = batch_data['targets']
    return outs
