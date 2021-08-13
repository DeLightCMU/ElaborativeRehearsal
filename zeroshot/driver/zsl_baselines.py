import os
import json
import yaml
from easydict import EasyDict
import time
import numpy as np
import argparse

import framework.run_utils
from framework.multiprocessing import mrun

import torch
import torch.multiprocessing as mp

import zeroshot.models.baselines
import zeroshot.datasets.prevideo

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('cfg_file')
  parser.add_argument('mtype', choices=['devise', 'sje', 'ale', 'dem', 'eszsl'])
  parser.add_argument('--is_train', action='store_true', default=False)
  parser.add_argument('--resume_file', default=None)
  parser.add_argument('--eval_set')
  parser.add_argument('--shard_id', type=int, default=0)
  parser.add_argument('--num_shards', type=int, default=1)
  parser.add_argument('--init_method', default='tcp://localhost:23456')
  parser.add_argument('--dist_backend', default='nccl', type=str)
  opts = parser.parse_args()

  with open(opts.cfg_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  config = EasyDict(config['config'])
  config.gpus = min(torch.cuda.device_count(), config.gpus)

  saver_cfg = framework.run_utils.gen_common_pathcfg(config.saver, is_train=opts.is_train)
  data_cfg = config.dataset
  model_cfg = config.model

  if opts.mtype in ['devise', 'sje', 'ale']:
    _model = zeroshot.models.baselines.DeviseModelHelper(config)
  elif opts.mtype == 'dem':
    _model = zeroshot.models.baselines.DEModelHelper(config)
  elif opts.mtype == 'eszsl':
    _model = zeroshot.models.baselines.ESZSLModelHelper(config)
    
  _dataset_fn = zeroshot.datasets.prevideo.ZSLPrecompVideoLabelEmbedDataset
  
  if opts.is_train:
    yaml.dump({'config': framework.run_utils.edict2dict(config)},
      open(os.path.join(saver_cfg.log_dir, 'config.yml'), 'w'))
    json.dump(vars(opts), open(os.path.join(saver_cfg.log_dir, 'opts.json'), 'w'), indent=2)

    trn_dataset = _dataset_fn(data_cfg.name_files['trn'], data_cfg.ft_dir,
      data_cfg.label_embed_file, data_cfg.meta_files['trn'], 
      data_cfg.class_idxs_files['trn'], data_cfg.k_class_split)
    val_dataset = _dataset_fn(data_cfg.name_files['val'], data_cfg.ft_dir,
      data_cfg.label_embed_file, data_cfg.meta_files['val'], 
      data_cfg.class_idxs_files['val'], data_cfg.k_class_split)

    if config.gpus > 1:
      func = _model.train
      func_args = {'trn_dataset': trn_dataset, 'val_dataset': val_dataset,
        'resume_file': opts.resume_file}
      mp.spawn(mrun, nprocs=config.gpus,
        args=(config.gpus, opts.init_method, opts.shard_id, opts.num_shards,
          opts.dist_backend, func, func_args), daemon=False)
    else:
      _model.train(trn_dataset, val_dataset, resume_file=opts.resume_file)

  else:
    tst_dataset = _dataset_fn(data_cfg.name_files[opts.eval_set], data_cfg.ft_dir,
      data_cfg.label_embed_file, data_cfg.meta_files[opts.eval_set], 
      data_cfg.class_idxs_files[opts.eval_set], data_cfg.k_class_split)

    if opts.mtype == 'eszsl':
      tst_pred_file = os.path.join(saver_cfg.pred_dir, opts.eval_set+'.npy')
      print(_model.test(tst_dataset, tst_pred_file))
      exit(0)

    model_str_scores = []
    if opts.resume_file is None:
      model_files = framework.run_utils.find_best_val_models(saver_cfg.log_dir, saver_cfg.model_dir)
    else:
      model_files = {'predefined': opts.resume_file}

    for measure_name, model_file in model_files.items():
      set_pred_dir = os.path.join(saver_cfg.pred_dir, opts.eval_set)
      if not os.path.exists(set_pred_dir):
        os.makedirs(set_pred_dir)
      tst_pred_file = os.path.join(set_pred_dir,
        os.path.splitext(os.path.basename(model_file))[0]+'.npy')

      scores = _model.test(tst_dataset, tst_pred_file, tst_model_file=model_file)
      if scores:
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
      score_log_file = os.path.join(saver_cfg.pred_dir, opts.eval_set, 'scores.csv')
      with open(score_log_file, 'a') as f:
        for str_scores in model_str_scores:
          print(str_scores, file=f)


if __name__ == '__main__':
  main()
