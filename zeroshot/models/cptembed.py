import torch
import torch.nn as nn
import torch.nn.functional as F

import zeroshot.modules.text
from zeroshot.models.ervse import PrecompERVSEModel, PrecompERVSEModelHelper

class ConceptEmbedModel(PrecompERVSEModel):
  def __init__(self, config):
    nn.Module.__init__(self)
    self.config = config
    self.txt_backbone = getattr(zeroshot.modules.text, self.config.txt_backbone.type)(
      **self.config.txt_backbone.kwargs)

    self.txt_encoder = zeroshot.modules.text.TextEncoder(
      self.config.txt_encoder.dim_input, self.config.txt_encoder.dim_embed,
      self.config.txt_encoder.pooling_method,
      self.config.txt_encoder.dropout, self.config.txt_encoder.l2norm)

  def forward(self, inputs, ercpt_inputs=None):
    outs = {}
    vis_cpt_ids, vis_cpt_masks = inputs['vis_cpt_ids'], inputs['vis_cpt_masks']
    if self.config.sep_in_cpts:
      batch_size = vis_cpt_ids.size(0)
      vis_cpt_ids = vis_cpt_ids.view(batch_size * self.config.topk_vis_cpts, -1)
      vis_cpt_masks = vis_cpt_masks.view(batch_size * self.config.topk_vis_cpts, -1)
    outs['vis_embeds'] = self.forward_text_embeds(vis_cpt_ids, vis_cpt_masks)
    if self.config.sep_in_cpts:
      outs['vis_embeds'] = outs['vis_embeds'].view(batch_size, self.config.topk_vis_cpts, -1)
    outs['txt_embeds'] = self.forward_text_embeds(inputs['txt_ids'], inputs['txt_masks'])
    if ercpt_inputs:
      outs['er_cpt_embeds'] = self.forward_text_embeds(inputs['er_cpt_ids'], inputs['er_cpt_masks'])
    return outs

class ConceptEmbedModelHelper(PrecompERVSEModelHelper):
  def build_model(self):
    return ConceptEmbedModel(self.config.model)

  def generate_scores(self, vis_embeds, class_embeds):
    # compute video-text similarity
    if len(vis_embeds.size()) == 2:
      scores = torch.einsum('bd,cd->bc', vis_embeds, class_embeds)
    else:
      scores = torch.einsum('bkd,cd->bkc', vis_embeds, class_embeds)
      cpt_scores = torch.einsum('bpd,bqd->bpq', vis_embeds, vis_embeds)
      cpt_scores = cpt_scores.clamp(min=0)
      cpt_scores = 1 / torch.sum(cpt_scores, dim=2)
      #print('scores', scores[0])
      # attns = torch.softmax(scores, dim=2) #scores / torch.sum(scores, 2, keepdim=True)
      #print('norm', attns[0])
      #attns = attns.masked_fill(attns == 0, -float('inf'))
      attns = torch.softmax(10 * cpt_scores, dim=1).unsqueeze(2)
      #attns = F.normalize(cpt_scores.unsqueeze(2), p=1, dim=1)
      #print('softmax', attns[0])
      scores = torch.sum(attns * scores, 1)
      # scores = torch.mean(scores, 1)
      # scores, _ = torch.max(scores, 1)
    return scores

  def train_one_step(self, batch_data, step=None):
    inputs = self.prepare_batch_inputs(batch_data)
    outs = self.model(inputs, ercpt_inputs=inputs if self.config.trainer.loss.er_loss_weight else None)
    vis_embeds = outs['vis_embeds']
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
    logits = self.generate_scores(outs['vis_embeds'], outs['txt_embeds'])

    return {
      'logits': logits.data.cpu(),
      'targets': batch_data['targets'],
    }

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


