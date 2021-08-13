import os
import json
import datetime
import numpy as np
import glob
from easydict import EasyDict

def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, EasyDict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj 

def gen_common_pathcfg(saver_cfg, is_train=False):
  output_dir = saver_cfg.output_dir

  saver_cfg.log_dir = os.path.join(output_dir, 'log')
  saver_cfg.model_dir = os.path.join(output_dir, 'model')
  saver_cfg.pred_dir = os.path.join(output_dir, 'pred')
  if not os.path.exists(saver_cfg.log_dir):
    os.makedirs(saver_cfg.log_dir)
  if not os.path.exists(saver_cfg.model_dir):
    os.makedirs(saver_cfg.model_dir)
  if not os.path.exists(saver_cfg.pred_dir):
    os.makedirs(saver_cfg.pred_dir)

  if is_train:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    saver_cfg.log_file = os.path.join(saver_cfg.log_dir, 'log-' + timestamp)
  else:
    saver_cfg.log_file = None

  return saver_cfg


def find_best_val_models(log_dir, model_dir):
  model_jsons = glob.glob(os.path.join(log_dir, 'val.epoch.*.step.*.json'))

  val_names, val_scores = [], []
  for i, json_file in enumerate(model_jsons):
    json_name = os.path.basename(json_file)
    scores = json.load(open(json_file))
    val_names.append(json_name)
    val_scores.append(scores)
    
  measure_names = list(val_scores[0].keys())
  model_files = {}
  for measure_name in measure_names:
    # for metrics: the lower the better
    if 'loss' in measure_name or 'medr' in measure_name or 'meanr' in measure_name:
      idx = np.argmin([scores[measure_name] for scores in val_scores])
    # for metrics: the higher the better
    else:
      idx = np.argmax([scores[measure_name] for scores in val_scores])
    json_name = val_names[idx]
    model_file = os.path.join(model_dir,  json_name[4:-5] + '.th')
    model_files.setdefault(model_file, [])
    model_files[model_file].append(measure_name)

  name2file = {'-'.join(measure_name): model_file for model_file, measure_name in model_files.items()}

  return name2file
