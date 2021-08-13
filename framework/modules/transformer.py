"""
Transformer Model from "Attention is All You Need"
modified from:
  https://opennmt.net/OpenNMT-py/_modules/onmt/encoders/transformer.html
  https://opennmt.net/OpenNMT-py/_modules/onmt/decoders/transformer.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.modules.multi_headed_attn import MultiHeadedAttention
import framework.ops

class TransformerEncoderLayer(nn.Module):
  """A single layer of the transformer encoder.
  Args:
    d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
    heads (int): the number of head for MultiHeadedAttention.
    d_ff (int): the second-layer of the PositionwiseFeedForward.
    dropout (float): dropout probability(0-1.0).
  """
  def __init__(self, d_model, num_heads, d_ff, dropout, attention_dropout, 
    normalize_before=False):
    super().__init__()

    self.self_attn = MultiHeadedAttention(
      num_heads, d_model, dropout=attention_dropout)
    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)
    self.normalize_before = normalize_before

  def forward_post(self, src, src_mask):
    src2, _ = self.self_attn(src, src, src, key_mask=src_mask)
    src = src + self.dropout(src2)
    src = self.norm1(src)
    src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
    src = src + self.dropout(src2)
    src = self.norm2(src)
    return src

  def forward_pre(self, src, src_mask):
    """
    Args:
      src (`FloatTensor`): `[batch_size, src_len, model_dim]`
      src_mask (`LongTensor`): `[batch_size, src_len]`

    Returns:
      * outputs `[batch_size x src_len x model_dim]`
    """
    # Pre-LN (On Layer Normalization in the Transformer Architecture)
    src2 = self.norm1(src)
    src2, _ = self.self_attn(src2, src2, src2, key_mask=src_mask, attn_type='self')
    src = src + self.dropout(src2)
    src2 = self.norm2(src)
    src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
    src = src + self.dropout(src2)
    return src

  def forward(self, src, src_mask):
    if self.normalize_before:
      return self.forward_pre(src, src_mask)
    return self.forward_post(src, src_mask)


class TransformerEncoder(nn.Module):
  """
  The Transformer encoder from "Attention is All You Need".

  Args:
    num_layers (int): number of encoder layers
    d_model (int): size of the model
    heads (int): number of heads
    d_ff (int): size of the inner FF layer
    dropout (float): dropout parameters

  Returns:
    (`FloatTensor`, `FloatTensor`):
    * memory_bank `[src_len x batch_size x model_dim]`
  """

  def __init__(self, num_layers, d_model, num_heads, d_ff, 
    dropout, attention_dropout, normalize_before=False, norm=None):
    super().__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.layers = nn.ModuleList(
        [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, attention_dropout, 
          normalize_before=normalize_before) for _ in range(num_layers)])
    self.norm = norm

  def forward(self, src_embeds, src_lens):
    ''' Args:
      src_embeds: (batch, src_len, d_model)
      src_lens: (batch, )
    '''
    batch, src_max_len, d_model = src_embeds.size()

    # (batch, src_len)
    src_masks = framework.ops.sequence_mask(src_lens, max_len=src_max_len, inverse=False)

    # Run the forward pass of every layer of the tranformer.
    outs = src_embeds
    for i in range(self.num_layers):
      outs = self.layers[i](outs, src_masks)

    if self.norm is not None:
      outs = self.norm(outs)

    return outs


class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout, attention_dropout, 
    normalize_before=False):
    super().__init__()
    # attention on self generated text
    self.self_attn = MultiHeadedAttention(
      num_heads, d_model, dropout=attention_dropout)
    self.norm1 = nn.LayerNorm(d_model)

    # attention on encoder memory
    self.context_attn = MultiHeadedAttention(
      num_heads, d_model, dropout=attention_dropout)
    self.norm2 = nn.LayerNorm(d_model)

    # feedforward transform
    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.norm3 = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout)
    self.normalize_before = normalize_before

  def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None, layer_cache=None):
    tgt2, _ = self.self_attn(tgt, tgt, tgt, 
      key_mask=tgt_mask, layer_cache=layer_cache, attn_type='self')
    tgt = tgt + self.dropout(tgt2)
    tgt = self.norm1(tgt)
    
    tgt2, mem_attns = self.context_attn(tgt, memory, memory, 
      key_mask=memory_mask, layer_cache=layer_cache, attn_type='context')
    tgt = tgt + self.dropout(tgt2)
    tgt = self.norm2(tgt)
    # tmp = int(mem_attns.size(-1)**0.5)
    # print(torch.max(mem_attns[0], 0)[0][0].view(tmp, tmp))

    tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
    tgt = tgt + self.dropout(tgt2)
    tgt = self.norm3(tgt)
    return tgt, mem_attns

  def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None, layer_cache=None):
    ''' Pre-LN
    Args:
      tgt: (batch, query_len, dim_embed)
      memory: (batch, key_len, dim_key)
      tgt_mask: (batch, query_len, query_len)
      memory_mask: (batch, query_len, key_len)
    Returns:
      output: (batch, query_len, d_model)
      mem_attns: (batch, num_heads, query_len, key_len)
    '''
    tgt2 = self.norm1(tgt)
    tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, 
      key_mask=tgt_mask, layer_cache=layer_cache, attn_type='self')
    tgt = tgt + self.dropout(tgt2)
    
    tgt2 = self.norm2(tgt)
    tgt2, mem_attns = self.context_attn(tgt2, memory, memory, 
      key_mask=memory_mask, layer_cache=layer_cache, attn_type='context')
    tgt = tgt + self.dropout(tgt2)

    tgt2 = self.norm3(tgt)
    tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
    tgt = tgt + self.dropout(tgt2)
    return tgt, mem_attns

  def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, layer_cache=None):
    if self.normalize_before:
      return self.forward_pre(tgt, memory, tgt_mask, memory_mask, layer_cache)
    return self.forward_post(tgt, memory, tgt_mask, memory_mask, layer_cache)

class TransformerDecoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, 
    dropout, attention_dropout, normalize_before=False, norm=None):
    super().__init__()
    self.num_layers = num_layers
    self.d_model = d_model
    self.layers = nn.ModuleList(
      [TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, attention_dropout,
        normalize_before=normalize_before) for _ in range(num_layers)])
    self.norm = norm

    # decoder cache
    self.cache = None

  def _init_cache(self):
    self.cache = {}
    for i, layer in enumerate(self.layers):
      self.cache['layer_%d'%i] = {
        'memory_keys': None, 
        'memory_values': None,
        'self_keys': None,
        'self_values': None,
      }

  def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, step=None):
    '''Args:
      tgt: (batch, query_len, dim_embed)
           (query_len = 1 in inference decoding)
      memory: (batch, key_len, dim_embed)
      tgt_mask: (batch, query_len, prev_query_len)
      memory_mask: (batch, key_len)
    '''
    if step == 0:
      self._init_cache()

    output = tgt
    for i, layer in enumerate(self.layers):
      layer_cache = self.cache['layer_%d'%i] if step is not None else None
      output, _ = layer(output, memory, tgt_mask=tgt_mask,
        memory_mask=memory_mask, layer_cache=layer_cache)

    if self.norm is not None:
      output = self.norm(output)

    return output
