import os
import json
import jsonlines
import numpy as np
import logging

from nltk.corpus import wordnet as wn
import transformers

import torch.utils.data
import torch.nn.functional as F

class ZSLPrecompVideoLabelEmbedDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, ft_dir, label_embed_file, meta_file,
    class_idxs_file, k_class_split):
    # ZSL only utilize a subset of all classes
    self.class_idxs = json.load(open(class_idxs_file))[k_class_split]
    self.ft_dir = ft_dir
    self.label_embeds = torch.FloatTensor(np.load(label_embed_file)[self.class_idxs])

    self.names, self.name2targets = [], {}
    names_set = set([os.path.splitext(x)[0] for x in json.load(open(name_file))])
    class_map = {ori_idx: new_idx for new_idx, ori_idx in enumerate(self.class_idxs)}
    with jsonlines.open(meta_file, 'r') as f:
      for item in f:
        name = os.path.splitext(item['videoname'])[0]
        target = item['label']
        if name in names_set and target in class_map:
          self.names.append(name)
          self.name2targets[name] = class_map[target]

  def __len__(self):
    return len(self.names)

  def __getitem__(self, idx):
    name = self.names[idx]
    vis_ft = np.load(os.path.join(self.ft_dir, name+'.npy'))[0]
    target = self.name2targets[name]
    return {
      'names': name, 
      'vis_fts': vis_ft, 
      'txt_fts': self.label_embeds, 
      'targets': target
    }

  @staticmethod
  def collate_fn(data):
    outs = {
      'names': [x['names'] for x in data],
      'vis_fts': torch.FloatTensor(np.stack([x['vis_fts'] for x in data], 0)),
      'txt_fts': data[0]['txt_fts'],
      'targets': torch.LongTensor(np.stack([x['targets'] for x in data]))
    }
    return outs

class ZSLPrecompVideoLabelGraphDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, label_embed_file, meta_file,
    class_idxs_file, k_class_split, target_embed_file=None, ft_dir=None,
    graph_topk_neighbors=5, is_train=False):

    self.class_idxs = torch.LongTensor(json.load(open(class_idxs_file))[k_class_split])
    # GCN-KG need all class labels during GCN encoding: (all_classes, dim)
    self.label_embeds = torch.FloatTensor(np.load(label_embed_file))
    self.label_graph = self.build_graph(self.label_embeds, topk=graph_topk_neighbors)

    self.ft_dir = ft_dir
    self.is_train = is_train

    if self.is_train:
      # ZSL requires only target embeds for seen classes: (seen_classes, dim)
      self.target_embeds = torch.FloatTensor(np.load(target_embed_file))[self.class_idxs]
    else:
      self.names, self.name2targets = [], {}
      names_set = set([os.path.splitext(x)[0] for x in json.load(open(name_file))])
      class_map = {ori_idx: new_idx for new_idx, ori_idx in enumerate(self.class_idxs.numpy())}
      with jsonlines.open(meta_file, 'r') as f:
        for item in f:
          name = os.path.splitext(item['videoname'])[0]
          target = item['label']
          if name in names_set and target in class_map:
            self.names.append(name)
            self.name2targets[name] = class_map[target]
 
  def build_graph(self, embeds, topk=5):
    num_nodes = embeds.size(0)
    normed_embeds = F.normalize(embeds, p=2, dim=1)
    cosine_sims = torch.matmul(normed_embeds, normed_embeds.t())
    _, topk_nodes = torch.topk(cosine_sims, topk, 1)

    A = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
      for j in topk_nodes[i]:
        A[i, j] = 1
        A[j, i] = 1
    A = A / torch.sum(A, 1, keepdim=True)
    return A

  def __len__(self):
    if self.is_train:
      return 1
    else:
      return len(self.names)

  def __getitem__(self, idx):
    if self.is_train:
      return {
        'txt_fts': self.label_embeds,
        'graph_edges': self.label_graph,
        'seen_class_idxs': self.class_idxs,
        'target_embeds': self.target_embeds
      }
    else:
      name = self.names[idx]
      vis_ft = np.load(os.path.join(self.ft_dir, name+'.npy'))[0]
      target = self.name2targets[name]
      return {
        'names': name, 
        'vis_fts': vis_ft, 
        'txt_fts': self.label_embeds, 
        'graph_edges': self.label_graph,
        'unseen_class_idxs': self.class_idxs,
        'targets': target
      }

  @staticmethod
  def collate_fn(data):
    if 'target_embeds' in data[0]: # is_train:
      return data[0]
    else:
      return {
        'names': [x['names'] for x in data],
        'vis_fts': torch.FloatTensor(np.stack([x['vis_fts'] for x in data], 0)),
        'txt_fts': data[0]['txt_fts'],
        'graph_edges': data[0]['graph_edges'],
        'unseen_class_idxs': data[0]['unseen_class_idxs'],
        'targets': torch.LongTensor(np.stack([x['targets'] for x in data]))
      }


class ZSLPrecompVideoLabelWordDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, ft_dir, label_text_file, meta_file,
    class_idxs_file, k_class_split, int2word_file=None, max_words_in_text=100, 
    _tokenizer=None, cat_word_with_defn=0, is_train=False):

    self.ft_dir = ft_dir
    self.max_words_in_text = max_words_in_text

    self.class_idxs = json.load(open(class_idxs_file))[k_class_split]
    self.label_texts = json.load(open(label_text_file))
    self.label_texts = [self.label_texts[idx] for idx in self.class_idxs]
    for i, x in enumerate(self.label_texts):
      if 'cleaned_word' not in x:
        self.label_texts[i]['cleaned_word'] = x['word']

    self.names, self.name2targets, self.name2nframes = [], {}, {}
    names_set = set([os.path.splitext(x)[0] for x in json.load(open(name_file))])
    class_map = {ori_idx: new_idx for new_idx, ori_idx in enumerate(self.class_idxs)}
    with jsonlines.open(meta_file, 'r') as f:
      for item in f:
        name = os.path.splitext(item['videoname'])[0]
        target = item['label']
        if name in names_set and target in class_map:
          self.names.append(name)
          self.name2nframes[name] = item['nframes']
          self.name2targets[name] = class_map[target]

    if _tokenizer is None:
      int2word = json.load(open(int2word_file))
      word2int = {w: i for i, w in enumerate(int2word)}
      self.label_text_ids, self.label_text_masks = [], []
      for x in self.label_texts:
        if cat_word_with_defn == 0:
          text = x['cleaned_word']
        elif cat_word_with_defn == 1:
          text = x['cleaned_defn']
        else:
          text = x['cleaned_word'] + ' ' + x['cleaned_defn']
        text_ids = [word2int[w] for w in text.split() if w in word2int][:self.max_words_in_text]
        text_masks = [1] * len(text_ids)
        if len(text_ids) < self.max_words_in_text:
          pad_length = self.max_words_in_text - len(text_ids)
          text_ids = text_ids + [0] * pad_length
          text_masks = text_masks + [0] * pad_length
        self.label_text_ids.append(text_ids)
        self.label_text_masks.append(text_masks)
      max_length = np.max(np.sum(self.label_text_masks, 1))
      self.label_text_ids = torch.LongTensor(self.label_text_ids)[:, :max_length]
      self.label_text_masks = torch.BoolTensor(self.label_text_masks)[:, :max_length]
    else:
      self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
      if cat_word_with_defn == 0:
        label_texts = [x['cleaned_word'] for x in self.label_texts]
        label_text_pairs = None
      elif cat_word_with_defn == 1:
        label_texts = [x['cleaned_defn'] for x in self.label_texts]
        label_text_pairs = None
      elif cat_word_with_defn == 2:
        label_texts = [x['cleaned_word'] + ': ' + x['cleaned_defn'] for x in self.label_texts]
        label_text_pairs = None
      elif cat_word_with_defn == 3:
        label_texts = [x['cleaned_word'] for x in self.label_texts]
        label_text_pairs = [x['cleaned_defn'] for x in self.label_texts]
      tmp = self.tokenizer(label_texts, text_pair=label_text_pairs, 
        add_special_tokens=True, padding='longest', 
        truncation=True, max_length=max_words_in_text)
      self.label_text_ids = torch.LongTensor(tmp['input_ids'])
      self.label_text_masks = torch.BoolTensor(tmp['attention_mask'])
    print('label %s' % str(self.label_text_ids.size()))

  def __len__(self):
    return len(self.names)

  def __getitem__(self, idx):
    name = self.names[idx]
    ft = np.load(os.path.join(self.ft_dir, name+'.npy'))[0]
    target = self.name2targets[name]
    return {
      'names': name, 'vis_fts': ft, 
      'txt_ids': self.label_text_ids, 'txt_masks': self.label_text_masks, 
      'targets': target}

  @staticmethod
  def collate_fn(data):
    return {
      'names': [x['names'] for x in data],
      'vis_fts': torch.FloatTensor(np.stack([x['vis_fts'] for x in data], 0)),
      'txt_ids': data[0]['txt_ids'],
      'txt_masks': data[0]['txt_masks'],
      'targets': torch.LongTensor([x['targets'] for x in data])
    }


class ZSLPrecompERVideoLabelWordDataset(ZSLPrecompVideoLabelWordDataset):
  def __init__(self, name_file, ft_dir, label_text_file, meta_file,
    class_idxs_file, k_class_split, video_concept_file=None,
    bit_cpt_lemma_file=None, bit_cpt_wnid_file=None, cat_cpt_defn={'in': False, 'out': False},
    max_words_in_text=10, cat_word_with_defn=0,
    topk_in_cpts=5, topk_out_cpts=5, num_neg_cpts=0, is_train=False):

    super().__init__(name_file, ft_dir, label_text_file, meta_file,
      class_idxs_file, k_class_split, max_words_in_text=max_words_in_text, 
      cat_word_with_defn=cat_word_with_defn,
      int2word_file=None, _tokenizer='bert')

    if video_concept_file:
      self.vid_concepts = json.load(open(video_concept_file))
      self.bit_cpt_names = [x.strip().lower().replace('_', ' ') for x in open(bit_cpt_lemma_file).readlines()]
      self.bit_cpt_defns = [wn.synset_from_pos_and_offset('n', int(x.strip()[1:])).definition().lower() for x in open(bit_cpt_wnid_file).readlines()]
      self.cat_cpt_defn = cat_cpt_defn
    self.topk_in_cpts = topk_in_cpts
    self.topk_out_cpts = topk_out_cpts
    self.num_neg_cpts = num_neg_cpts
    self.is_train = is_train

  def __getitem__(self, idx):
    outs = super().__getitem__(idx)
    name = self.names[idx]

    if self.is_train and self.topk_out_cpts:
      out_cpt_labels = []
      neg_cpts = [idx for idx in np.random.randint(len(self.bit_cpt_names), size=(self.num_neg_cpts,)) if idx not in self.vid_concepts[name][0]] 
      for idx in self.vid_concepts[name][0][:self.topk_out_cpts] + neg_cpts:
        cpt_name = self.bit_cpt_names[idx]
        if self.cat_cpt_defn['out']:
          cpt_name = cpt_name + ': ' + self.bit_cpt_defns[idx]
        out_cpt_labels.append(cpt_name)
      out_cpts = self.tokenizer(out_cpt_labels, add_special_tokens=True, 
        padding='max_length', truncation=True, max_length=self.max_words_in_text)
      outs['er_cpt_labels'] = out_cpt_labels[:self.topk_out_cpts]
      outs['er_cpt_ids'] = out_cpts['input_ids'][:self.topk_out_cpts]
      outs['er_cpt_masks'] = out_cpts['attention_mask'][:self.topk_out_cpts]
      outs['er_neg_cpt_labels'] = out_cpt_labels[self.topk_out_cpts:]
      outs['er_neg_cpt_ids'] = out_cpts['input_ids'][self.topk_out_cpts:]
      outs['er_neg_cpt_masks'] = out_cpts['attention_mask'][self.topk_out_cpts:]

    if self.topk_in_cpts:
      in_cpt_label = []
      for idx in self.vid_concepts[name][0][:self.topk_in_cpts]:
        cpt_name = self.bit_cpt_names[idx]
        if self.cat_cpt_defn['in']:
          cpt_name = cpt_name + ': ' + self.bit_cpt_defns[idx]
        in_cpt_label.append(cpt_name)
      in_cpt_label = ' '.join(in_cpt_label)
      in_cpts = self.tokenizer(in_cpt_label, add_special_tokens=True,
        padding='max_length', truncation=True, max_length=self.max_words_in_text)
      outs['vis_cpt_ids'] = in_cpts['input_ids']
      outs['vis_cpt_masks'] = in_cpts['attention_mask']

    return outs

  @staticmethod
  def collate_fn(data):
    outs = {
      'names': [x['names'] for x in data],
      'vis_fts': torch.FloatTensor(np.stack([x['vis_fts'] for x in data], 0)),
      'txt_ids': data[0]['txt_ids'],
      'txt_masks': data[0]['txt_masks'],
      'targets': torch.LongTensor([x['targets'] for x in data]),
      
    }
    if 'vis_cpt_ids' in data[0]:
      outs['vis_cpt_ids'] = torch.LongTensor(np.array([x['vis_cpt_ids'] for x in data]))
      outs['vis_cpt_masks'] = torch.BoolTensor(np.array([x['vis_cpt_masks'] for x in data]))
      max_len = torch.max(torch.sum(outs['vis_cpt_masks'], 1))
      outs['vis_cpt_ids'] = outs['vis_cpt_ids'][:, :max_len]
      outs['vis_cpt_masks'] = outs['vis_cpt_masks'][:, :max_len]
    
    if 'er_cpt_ids' in data[0]:
      er_cpt_ids, er_cpt_masks = [], []
      er_cpt_targets = []
      cpt2idx = {}
      for x in data:
        er_cpt_targets.append([])
        for k, cpt in enumerate(x['er_cpt_labels']):
          if cpt not in cpt2idx:
            cpt2idx[cpt] = len(cpt2idx) 
            er_cpt_ids.append(x['er_cpt_ids'][k])
            er_cpt_masks.append(x['er_cpt_masks'][k])
          er_cpt_targets[-1].append(cpt2idx[cpt])
        for k, cpt in enumerate(x['er_neg_cpt_labels']):
          if cpt not in cpt2idx:
            cpt2idx[cpt] = len(cpt2idx)
            er_cpt_ids.append(x['er_neg_cpt_ids'][k])
            er_cpt_masks.append(x['er_neg_cpt_masks'][k])
      er_cpt_ml_targets = torch.zeros(len(data), len(cpt2idx)).bool()
      for k, er_cpt_target in enumerate(er_cpt_targets):
        for idx in er_cpt_target:
          er_cpt_ml_targets[k, idx] = True
      
      outs['er_cpt_ml_targets'] = er_cpt_ml_targets
      outs['er_cpt_ids'] = torch.LongTensor(np.stack(er_cpt_ids, 0))
      outs['er_cpt_masks'] = torch.BoolTensor(np.stack(er_cpt_masks, 0))
      max_len = torch.max(torch.sum(outs['er_cpt_masks'], 1))
      for key in ['er_cpt_ids', 'er_cpt_masks']:
        outs[key] = outs[key][:, :max_len]
    return outs

