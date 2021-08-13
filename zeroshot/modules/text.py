import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.ops

import transformers

##################### Text Backbones #####################
class TextEmbedding(nn.Module):
  def __init__(self, num_words, dim_word, fix_word_embed):
    super().__init__()
    self.embedding = nn.Embedding(num_words, dim_word)
    if fix_word_embed:
      self.embedding.weight.requires_grad = False

  def forward(self, inputs, masks):
    return self.embedding(inputs)
    

class TextRNN(nn.Module):
  def __init__(self, num_words, dim_word, fix_word_embed, 
    rnn_type, rnn_bidirectional, dim_embed, dropout):
    super().__init__()
    self.rnn_bidirectional = rnn_bidirectional
    self.dim_embed = dim_embed
    
    self.embedding = nn.Embedding(num_words, dim_word)
    if fix_word_embed:
      self.embedding.weight.requires_grad = False

    self.rnn = getattr(nn, rnn_type.upper())(input_size=dim_word, 
      hidden_size=dim_embed, num_layers=1, dropout=0,
      bidirectional=rnn_bidirectional, bias=True, batch_first=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs, masks):
    word_embeds = self.dropout(self.embedding(inputs))
    
    seq_lens = torch.sum(masks, 1)
    # hiddens.size = (batch, len, num_directions * hidden_size)
    hiddens, states = framework.ops.calc_rnn_outs_with_sort(
      self.rnn, word_embeds, seq_lens)

    if self.rnn_bidirectional:
      splited_hiddens = torch.split(hiddens, self.dim_embed, dim=2) 
      hiddens = (splited_hiddens[0] + splited_hiddens[1]) / 2

    return hiddens


class TextBert(nn.Module):
  def __init__(self, fix_bert_before, data_parallel=False):
    super().__init__()

    self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
    self.fix_bert_before = fix_bert_before

    self.freeze_all = True
    if self.fix_bert_before:
      for (name, param) in self.bert.named_parameters():
        if name.startswith(self.fix_bert_before):
          self.freeze_all = False
          break
        param.requires_grad = False
    if self.freeze_all:
      self.caches = {'inputs': None}

    if data_parallel:
      self.bert = nn.DataParallel(self.bert)

  def forward(self, inputs, masks):
    '''
    Args:
      inputs: LongTensor, (batch, len)
      masks: BoolTensor, (batch, len)
    '''
    if self.freeze_all:
      if self.caches['inputs'] is None or inputs.size() != self.caches['inputs'].size() or torch.sum(inputs != self.caches['inputs']) > 0:
        outputs = self.bert(inputs, attention_mask=masks, return_dict=True)
        self.caches['inputs'] = inputs
        self.caches['last_hidden_state'] = outputs['last_hidden_state'].detach()
      outputs = self.caches
    else:
      outputs = self.bert(inputs, attention_mask=masks, return_dict=True)

    return outputs['last_hidden_state']


##################### Text Pooling and Embedding #####################
class TextEncoder(nn.Module):
  def __init__(self, dim_input, dim_embed, pooling_method, dropout, l2norm):
    # first, last, avg, max, attn
    super().__init__()
        
    if dim_input == dim_embed:
      self.ft_embed = None
    else:
      self.ft_embed = nn.Linear(dim_input, dim_embed)
    self.dropout = nn.Dropout(dropout)
    self.pooling_method = pooling_method
    self.l2norm = l2norm

    if self.pooling_method == 'attn':
      self.ft_attn = nn.Sequential(
        nn.Linear(dim_input, dim_input // 2),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(dim_input // 2, 1))

  def forward(self, inputs, masks):
    '''
    Args:
      inputs: FloatTensor, (batch, len, dim_input)
      masks: BoolTensor, (batch, len)
    '''
    inputs = self.dropout(inputs)

    if self.pooling_method == 'first':
      embeds = inputs[:, 0]

    elif self.pooling_method == 'avg':
      embeds = torch.sum(inputs.masked_fill(~masks.unsqueeze(2), 0), dim=1)
      embeds = embeds / torch.sum(masks, 1, keepdim=True)

    elif self.pooling_method == 'max':
      embeds, _ = torch.max(inputs.masked_fill(~masks.unsqueeze(2), -float('inf')), dim=1)

    elif self.pooling_method == 'last':
      input_size = inputs.size(-1)
      seq_len = torch.sum(masks, 1)
      idxs = (seq_len - 1).view(-1, 1, 1).repeat(1, 1, input_size)
      embeds = torch.gather(inputs, 1, idxs).squeeze(1)

    elif self.pooling_method == 'attn':
      weights = self.ft_attn(inputs).squeeze(2).masked_fill(~masks, -float('inf'))
      weights = torch.softmax(weights, dim=1)
      embeds = torch.sum(inputs * weights.unsqueeze(2), 1)

    if self.ft_embed is not None:
      embeds = self.ft_embed(embeds)
    if self.l2norm:
      embeds = F.normalize(embeds, p=2, dim=-1)

    return embeds




