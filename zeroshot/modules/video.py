import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalFtEncoder(nn.Module):
  def __init__(self, dim_ft, dim_embed, dropout, l2norm):
    super().__init__()
    self.ft_embed = nn.Linear(dim_ft, dim_embed)
    self.dropout = nn.Dropout(dropout)
    self.l2norm = l2norm

  def forward(self, inputs):
    embeds = self.ft_embed(self.dropout(inputs))
    if self.l2norm:
      embeds = F.normalize(embeds, p=2, dim=-1)
    return embeds
