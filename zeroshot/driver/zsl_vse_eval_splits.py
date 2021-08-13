import os
import json
import yaml
from easydict import EasyDict
import time
import numpy as np
import argparse
import glob
import torch.utils.data

import framework.run_utils

import zeroshot.models.vse
import zeroshot.datasets.prevideo

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('cfg_file')
  parser.add_argument('k_class_split', type=int)
  parser.add_argument('--shard_id', type=int, default=0)
  parser.add_argument('--num_shards', type=int, default=1)
  parser.add_argument('--init_method', default='tcp://localhost:23456')
  parser.add_argument('--dist_backend', default='nccl', type=str)
  opts = parser.parse_args()

  with open(opts.cfg_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  config = EasyDict(config['config'])

  saver_cfg = framework.run_utils.gen_common_pathcfg(config.saver, is_train=False)
  data_cfg = config.dataset
  model_cfg = config.model

  _model = zeroshot.models.vse.PrecompVSEModelHelper(config)
  _dataset_fn = zeroshot.datasets.prevideo.ZSLPrecompVideoLabelWordDataset

  _model._setup_env()
  
  val_dataset = _dataset_fn(data_cfg.name_files['val'], data_cfg.ft_dir,
    data_cfg.label_text_file, data_cfg.meta_files['val'], 
    data_cfg.class_idxs_files['val'], opts.k_class_split,
    int2word_file=data_cfg.int2word_file, max_words_in_text=data_cfg.max_words_in_text, 
    _tokenizer=data_cfg.tokenizer, cat_word_with_defn=data_cfg.cat_word_with_defn)
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    batch_size=config.evaluate.batch_size, num_workers=data_cfg.num_workers,
    pin_memory=True, shuffle=False, collate_fn=val_dataset.collate_fn)
  _model.print_fn('# data: %d, # batch %d' % (len(val_dataset), len(val_loader)))

  model_files = glob.glob(os.path.join(saver_cfg.model_dir, 'epoch.*.step.*.th'))
  model_files.sort(key=lambda x: int(os.path.basename(x).split('.')[1]))
  pred_dir = os.path.join(saver_cfg.pred_dir, 'split%02d'%opts.k_class_split)
  val_pred_dir = os.path.join(pred_dir, 'val')
  tst_pred_dir = os.path.join(pred_dir, 'tst')
  os.makedirs(val_pred_dir, exist_ok=True)
  os.makedirs(tst_pred_dir, exist_ok=True)

  for model_file in model_files:
    model_name = os.path.basename(model_file)
    outfile = os.path.join(val_pred_dir, 'val.'+model_name[:-2]+'json')
    if os.path.exists(outfile):
      continue
    _model.load_checkpoint(model_file)
    model_metrics = _model.validate(val_loader, log=False)
    with open(outfile, 'w') as outf:
      json.dump(model_metrics, outf, indent=2)
    _model.pretty_print_metrics(model_name, model_metrics)

  model_str_scores = []
  model_files = framework.run_utils.find_best_val_models(val_pred_dir, saver_cfg.model_dir)
  _model.config.verbose = False

  tst_dataset = _dataset_fn(data_cfg.name_files['tst'], data_cfg.ft_dir,
    data_cfg.label_text_file, data_cfg.meta_files['tst'],
    data_cfg.class_idxs_files['tst'], opts.k_class_split,
    int2word_file=data_cfg.int2word_file, max_words_in_text=data_cfg.max_words_in_text,
    _tokenizer=data_cfg.tokenizer, cat_word_with_defn=data_cfg.cat_word_with_defn)

  for measure_name, model_file in model_files.items():
    tst_pred_file = os.path.join(tst_pred_dir,
      os.path.splitext(os.path.basename(model_file))[0]+'.npy')

    scores = _model.test(tst_dataset, tst_pred_file, tst_model_file=model_file)
    if len(model_str_scores) == 0:
      model_str_scores.append(','.join(list(scores.keys())))
      print(model_str_scores[0])
    str_scores = [measure_name, os.path.basename(model_file)]
    for score_name in scores.keys():
      str_scores.append('%.2f'%(scores[score_name]))
    str_scores = ','.join(str_scores)
    print(str_scores)
    model_str_scores.append(str_scores)

  if len(model_str_scores) > 0:
    score_log_file = os.path.join(tst_pred_dir, 'scores.csv')
    with open(score_log_file, 'a') as f:
      for str_scores in model_str_scores:
        print(str_scores, file=f)


if __name__ == '__main__':
  main()
