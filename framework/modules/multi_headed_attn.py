""" 
Multi-Head Attention module 
modified from :
  https://opennmt.net/OpenNMT-py/_modules/onmt/modules/multi_headed_attn.html
"""

import math
import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
  """
  Multi-Head Attention module from
  "Attention is All You Need"
  
  Similar to standard `dot` attention but uses
  multiple attention distributions simulataneously
  to select relevant items.

  Args:
     num_heads (int): number of parallel heads
     dim_embed (int): the dimension of keys/values/queries,
       must be divisible by num_heads
     dropout (float): dropout parameter
     dim_key: in case the embed size of key is not the same as query
     dim_value: in case the embed size of value is not the same as query
  """

  def __init__(self, num_heads, dim_embed, dropout=0.1, dim_key=None, dim_value=None):
    super().__init__()

    assert dim_embed % num_heads == 0, 'dim_embed must be divisible by num_heads'
    self.dim_per_head = dim_embed // num_heads
    self.dim_embed = dim_embed
    self.num_heads = num_heads
    self.dim_key = dim_key if dim_key is not None else self.dim_embed
    self.dim_value = dim_value if dim_value is not None else self.dim_embed

    self.linear_keys = nn.Linear(self.dim_key, self.dim_embed)
    self.linear_values = nn.Linear(self.dim_value, self.dim_embed)
    self.linear_query = nn.Linear(self.dim_embed, self.dim_embed)
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(dropout)
    self.final_linear = nn.Linear(self.dim_embed, self.dim_embed)

  def forward(self, query, key, value, key_mask=None, layer_cache=None, attn_type=None):
    """
    Compute the context vector and the attention weight vector.

    Args:
       key (FloatTensor): (batch, key_len, dim_key)
       value (FloatTensor): (batch, key_len, dim_value)
       query (FloatTensor): (batch, query_len, dim_embed)
       key_mask: binary mask indicating which keys exist
        mask=0 denotes PADDING, mask=1 denotes real value
        shape: (batch, key_len) or
               (batch, query_len, key_len) or 
               (batch, num_heads, query_len, key_len)
       layer_cache: dict, the key/value differs across attn_type
          - self: {'self_keys': , 'self_values': }
          - context: {'memory_keys': , 'memory_values': }
          used in inference phase because we can only decode step by step
       attn_type: self (growing memories of each layer),
                  context (fixed memories of the last layer)

    Returns:
       output: context vector (batch, query_len, dim_embed)
       attn: attention score (batch, num_heads, query_len, key_len)
    """
    batch_size = key.size(0)
    num_heads = self.num_heads
    dim_per_head = self.dim_per_head

    def shape(x):
      """  projection """
      # (N, L, D) -> (N, H, L, D/H)
      return x.view(batch_size, -1, num_heads, dim_per_head) \
        .transpose(1, 2)

    def unshape(x):
      """  compute context """
      # (N, H, L, D/H) -> (N, L, D)
      return x.transpose(1, 2).contiguous() \
          .view(batch_size, -1, num_heads * dim_per_head)

    # 1) Project key, value, and query.
    if layer_cache is not None:
      if attn_type == "self":
        # e.g. in text generation, attention on self-generated words
        # the key/value continues to grow with the generated words
        # layer_cache contains the converted keys/values of previous timesteps
        # so linear transformation is only applied on new generated key/value
        key = self.linear_keys(key) # (batch, key_len=1, dim_embed)
        value = self.linear_values(value)
        key = shape(key) # (batch, num_heads, key_len=1, dim_per_head)
        value = shape(value)
        if layer_cache['self_keys'] is not None:
          key = torch.cat((layer_cache['self_keys'], key), dim=2)
        if layer_cache['self_values'] is not None:
          value = torch.cat((layer_cache['self_values'], value), dim=2)
        layer_cache['self_keys'] = key
        layer_cache['self_values'] = value

      elif attn_type == "context":
        # to attend on contextual values 
        # e.g. the decoder needs to attend on hidden states of the encoder
        if layer_cache["memory_keys"] is None:
          key = self.linear_keys(key)
          value = self.linear_values(value)
          key = shape(key)
          value = shape(value)
          layer_cache['memory_keys'] = key
          layer_cache['memory_values'] = value
        else:
          key = layer_cache['memory_keys']
          value = layer_cache['memory_values']

    else:
      key = self.linear_keys(key) # (batch, key_len, dim_embed)
      value = self.linear_values(value)
      key = shape(key) # (batch, num_heads, key_len, dim_per_head)
      value = shape(value)

    query = self.linear_query(query) # (batch, query_len, dim_embed)
    query = shape(query) # (batch, num_heads, query_len, dim_per_head)

    key_len = key.size(2)
    query_len = query.size(2)

    # 2) Calculate and scale scores.
    query = query / math.sqrt(dim_per_head)
    # scores.size = (batch, num_heads, query_len, key_len)
    scores = torch.matmul(query, key.transpose(2, 3))

    if key_mask is not None:
      key_padding_mask = (key_mask == 0)
      if len(key_padding_mask.size()) == 2:
        # (batch, 1, 1, key_len)
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
      elif len(key_padding_mask.size()) == 3:
        # (batch, 1, query_len, key_len)
        key_padding_mask = key_padding_mask.unsqueeze(1)
      # (batch, num_heads, query_len, key_len)
      scores = scores.masked_fill(key_padding_mask, -1e18)

    # 3) Apply attention dropout and compute context vectors.
    attn = self.softmax(scores) # (batch, num_heads, query_len, key_len)
    if key_mask is not None:
      # necessary when all key_len elements are masked
      attn = attn.masked_fill(key_padding_mask, 0)
    drop_attn = self.dropout(attn)

    # (batch, query_len, num_heads * dim_per_head)
    context = unshape(torch.matmul(drop_attn, value))

    output = self.final_linear(context)

    return output, attn

