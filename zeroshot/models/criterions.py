import torch
import torch.nn as nn


class ClsContrastiveLoss(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0.2, max_violation=False, topk=1, reduction='sum'):

    super().__init__()
    self.margin = margin
    self.max_violation = max_violation
    self.topk = topk
    self.reduction = reduction
    if self.reduction == 'weighted':
      self.betas = torch.zeros(701)
      self.betas[1:] = torch.cumsum(1 / (torch.arange(700).float() + 1), 0)

  def forward(self, scores, int_labels):
    batch_size = scores.size(0)

    pos_scores = torch.gather(scores, 1, int_labels.unsqueeze(1))
    pos_masks = torch.zeros_like(scores).bool()
    pos_masks.scatter_(1, int_labels.unsqueeze(1), True)

    d1 = pos_scores.expand_as(scores)
    cost_s = (self.margin + scores - d1).clamp(min=0)
    cost_s = cost_s.masked_fill(pos_masks, 0)
    if self.max_violation:
      cost_s, _ = torch.topk(cost_s, self.topk, dim=1)
      if self.reduction == 'mean':
        cost_s = cost_s / self.topk
    else:
      if self.reduction == 'mean':
        cost_s = cost_s / (scores.size(1) - 1)
      elif self.reduction == 'weighted':
        gt_ranks = torch.sum(cost_s > 0, 1).unsqueeze(1)
        weights = self.betas.to(scores.device).unsqueeze(0).expand(batch_size, -1).gather(1, gt_ranks)
        weights = weights / (gt_ranks + 1e-8)
        cost_s = cost_s * weights
        
    cost_s = torch.sum(cost_s) / batch_size

    return cost_s

class RetContrastiveLoss(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0, max_violation=False, direction='bi', topk=1):
    '''Args:
      direction: i2t for negative sentence, t2i for negative image, bi for both
    '''
    super().__init__()
    self.margin = margin
    self.max_violation = max_violation
    self.direction = direction
    self.topk = topk

  def forward(self, scores, margin=None, average_batch=True):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
        the same row of im and s are positive pairs, different rows are negative pairs
    '''

    if margin is None:
      margin = self.margin

    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1) # positive pairs

    # mask to clear diagonals which are positive pairs
    pos_masks = torch.eye(batch_size).bool().to(scores.device)

    batch_topk = min(batch_size, self.topk)
    if self.direction == 'i2t' or self.direction == 'bi':
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (margin + scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill(pos_masks, 0)
      if self.max_violation:
        cost_s, _ = torch.topk(cost_s, batch_topk, dim=1)
        cost_s = cost_s / batch_topk
        if average_batch:
          cost_s = cost_s / batch_size
      else:
        if average_batch:
          cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = torch.sum(cost_s)

    if self.direction == 't2i' or self.direction == 'bi':
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (margin + scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)
      if self.max_violation:
        cost_im, _ = torch.topk(cost_im, batch_topk, dim=0)
        cost_im = cost_im / batch_topk
        if average_batch:
          cost_im = cost_im / batch_size
      else:
        if average_batch:
          cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = torch.sum(cost_im)

    if self.direction == 'i2t':
      return cost_s
    elif self.direction == 't2i':
      return cost_im
    else:
      return cost_s + cost_im

    
