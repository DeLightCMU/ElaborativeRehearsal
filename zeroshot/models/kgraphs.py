import os
import numpy as np
import itertools
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.trainer
import framework.dist_helper

from zeroshot.models.metrics import eval_precisions


class GCNLayer(nn.Module):
  def __init__(self, input_size, output_size, dropout, activation, bias=True):
    super().__init__()
    self.fc = nn.Linear(input_size, output_size, bias=bias)

    self.loop_weight = nn.Parameter(torch.Tensor(input_size, output_size))
    nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

    self.activation = activation
    self.dropout = nn.Dropout(dropout) 

  def forward(self, inputs, edges):
    '''Args:
      - inputs: (num_nodes, input_size)
      - edges: (num_nodes, num_nodes)
    '''
    inputs = self.dropout(inputs)
    outs = self.fc(inputs)
    outs = torch.matmul(edges, outs)
    loop_messages = torch.matmul(inputs, self.loop_weight)
    outs = outs + loop_messages
    if self.activation:
      outs = self.activation(outs)
    return outs


class KnowledgeGraphModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    gcn_cfg = self.config.gcn
    self.gcn = []
    input_size = gcn_cfg.input_size
    hidden_sizes = gcn_cfg.hidden_sizes + [gcn_cfg.output_size]
    for k, hidden_size in enumerate(hidden_sizes):
      if k == len(hidden_sizes) - 1:
        dropout = gcn_cfg.dropout
        activation = None
      else:
        dropout = 0
        activation = nn.LeakyReLU(0.2)
      self.gcn.append(GCNLayer(input_size, hidden_size, dropout, activation, bias=True))
      input_size = hidden_size
    self.gcn = nn.ModuleList(self.gcn)

  def forward(self, txt_fts, edges):
    if self.config.txt_norm:
      txt_fts = F.normalize(txt_fts, p=2, dim=1)
    embeds = txt_fts
    for layer in self.gcn:
      embeds = layer(embeds, edges)
    embeds = F.normalize(embeds, p=2, dim=1)
    return embeds


class KnowledgeGraphModelHelper(framework.trainer.TrnTstHelper):
  def build_model(self):
    return KnowledgeGraphModel(self.config.model)

  def build_loss(self):
    criterion = nn.MSELoss()
    return criterion

  def prepare_batch_inputs(self, batch_data):
    inputs = {}
    for key, value in batch_data.items():
      if isinstance(value, torch.Tensor):
        inputs[key] = value.to(self.device, non_blocking=True)
    return inputs

  def train_one_step(self, batch_data, step=None):
    inputs = self.prepare_batch_inputs(batch_data)

    txt_embeds = self.model(inputs['txt_fts'], inputs['graph_edges'])
    target_embeds = F.normalize(inputs['target_embeds'], p=2, dim=1)
    seen_class_idxs = inputs['seen_class_idxs']

    loss = self.criterion(txt_embeds[seen_class_idxs], target_embeds)
    return loss

  def val_one_step(self, batch_data):
    inputs = self.prepare_batch_inputs(batch_data)
    txt_embeds = self.model(inputs['txt_fts'], inputs['graph_edges'])[inputs['unseen_class_idxs']]
    logits = torch.matmul(inputs['vis_fts'], txt_embeds.t())

    return {
      'logits': logits.data.cpu(),
      'targets': batch_data['targets']
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
    txt_embeds = self.model(inputs['txt_fts'], inputs['graph_edges'])[inputs['unseen_class_idxs']]
    logits = torch.matmul(inputs['vis_fts'], txt_embeds.t())

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
