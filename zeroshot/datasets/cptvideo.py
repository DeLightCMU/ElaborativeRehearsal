import os
import numpy as np
import torch

from zeroshot.datasets.prevideo import ZSLPrecompERVideoLabelWordDataset


class ZSLConceptVideoLabelWordDataset(ZSLPrecompERVideoLabelWordDataset):
  def __init__(self, name_file, video_concept_file, label_text_file, meta_file,
    class_idxs_file, k_class_split, bit_cpt_lemma_file=None, bit_cpt_wnid_file=None, 
    sep_in_cpts=False, cat_cpt_defn={'in': False, 'out': False},
    max_words_in_text=10, cat_word_with_defn=0,
    topk_in_cpts=5, topk_out_cpts=5, num_neg_cpts=0, is_train=False):

    super().__init__(name_file, None, label_text_file, meta_file,
      class_idxs_file, k_class_split, video_concept_file=video_concept_file,
      bit_cpt_lemma_file=bit_cpt_lemma_file, bit_cpt_wnid_file=bit_cpt_wnid_file, 
      cat_cpt_defn=cat_cpt_defn, max_words_in_text=max_words_in_text, 
      cat_word_with_defn=cat_word_with_defn,
      topk_in_cpts=topk_in_cpts, topk_out_cpts=topk_out_cpts, 
      num_neg_cpts=num_neg_cpts, is_train=is_train)
    self.sep_in_cpts = sep_in_cpts

  def __getitem__(self, idx):
    name = self.names[idx]
    
    outs = {'names': name, 'targets': self.name2targets[name],
            'txt_ids': self.label_text_ids, 'txt_masks': self.label_text_masks}
    in_cpt_labels = []
    for idx in self.vid_concepts[name][0][:self.topk_in_cpts]:
      cpt_name = self.bit_cpt_names[idx]
      if self.cat_cpt_defn['in']:
        cpt_name = cpt_name + ': ' + self.bit_cpt_defns[idx]
      in_cpt_labels.append(cpt_name)
    if not self.sep_in_cpts:
      in_cpt_labels = ' '.join(in_cpt_labels)
    in_cpts = self.tokenizer(in_cpt_labels, add_special_tokens=True,
      padding='max_length', truncation=True, max_length=self.max_words_in_text)
    outs['vis_cpt_ids'] = in_cpts['input_ids']
    outs['vis_cpt_masks'] = in_cpts['attention_mask']

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

    return outs

  @staticmethod
  def collate_fn(data):
    outs = {
      'names': [x['names'] for x in data],
      'txt_ids': data[0]['txt_ids'],
      'txt_masks': data[0]['txt_masks'],
      'targets': torch.LongTensor([x['targets'] for x in data]),
    }
    outs['vis_cpt_ids'] = torch.LongTensor(np.array([x['vis_cpt_ids'] for x in data]))
    outs['vis_cpt_masks'] = torch.BoolTensor(np.array([x['vis_cpt_masks'] for x in data]))
    max_len = torch.max(torch.sum(outs['vis_cpt_masks'], -1))
    if len(outs['vis_cpt_ids'].size()) == 2:
      outs['vis_cpt_ids'] = outs['vis_cpt_ids'][:, :max_len]
      outs['vis_cpt_masks'] = outs['vis_cpt_masks'][:, :max_len]
    else:
      outs['vis_cpt_ids'] = outs['vis_cpt_ids'][:, :, :max_len]
      outs['vis_cpt_masks'] = outs['vis_cpt_masks'][:, :, :max_len]
    
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
