import os
import numpy as np
import argparse
import yaml
from easydict import EasyDict
import copy
import json
import time

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from x_temporal.interface.temporal_helper import TemporalHelper
from x_temporal.utils.multiprocessing import mrun

from x_temporal.utils.dist_helper import get_rank, get_world_size, all_gather, all_reduce
from x_temporal.utils.utils import format_cfg, accuracy, AverageMeter, load_checkpoint
from x_temporal.utils.dataset_helper import get_val_crop_transform, get_dataset, shuffle_dataset
from x_temporal.core.models_entry import get_model, get_augmentation
from x_temporal.core.transforms import *

class TemporalFeatureExtractor(TemporalHelper):
  def __init__(self, config, ckpt_dict=None, inference_only=True):
    """
    Args:
      config: configuration for training and testing, sometimes
      ckpt_dict: {model: , epoch:, optimizer:, lr_scheduler:}
      inference_only:
    """
    self.inference_only = inference_only
    self.config = copy.deepcopy(config)

    self._setup_env()
    self.model = self.build_model()
    self._resume(ckpt_dict)
    self._ready()
    self._last_time = time.time()
    self.logger.info('Running with config:\n{}'.format(format_cfg(self.config)))

  def _build_dataloader(self, data_type):
    dargs = self.config.dataset
    if dargs.modality == 'RGB':
      data_length = 1
    elif dargs.modality in ['Flow', 'RGBDiff']:
      data_length = 5

    if dargs.modality != 'RGBDiff':
      normalize = GroupNormalize(dargs.input_mean, dargs.input_std)
    else:
      normalize = IdentityTransform()

    if self.inference_only:
      spatial_crops = self.config.get('evaluate', {}).get('spatial_crops', 1)
      temporal_samples = self.config.get('evaluate', {}).get('temporal_samples', 1)
    else:
      spatial_crops = 1
      temporal_samples = 1

    crop_aug = get_val_crop_transform(self.config.dataset, spatial_crops)
    transform = torchvision.transforms.Compose([
      GroupScale(int(dargs.scale_size)),
      crop_aug,
      Stack(roll=False),
      ToTorchFormatTensor(div=True),
      normalize,
      ConvertDataFormat(self.config.net.model_type),
    ])

    dataset = get_dataset(dargs, data_type, True, transform, data_length, temporal_samples)
    sampler = DistributedSampler(dataset) if self.config.gpus > 1 else None
    val_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=dargs.batch_size, shuffle=(False if sampler else False), 
      drop_last=False, num_workers=dargs.workers, 
      pin_memory=True, sampler=sampler)
    return val_loader

  def _ready(self):
    self.model = self.model.cuda()
  
  def build_model(self):
    model = get_model(self.config).cuda()
    return model

  @torch.no_grad()
  def extract_features(self, output_dir, setname='test'):
    os.makedirs(output_dir, exist_ok=True)

    batch_time = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)

    if self.inference_only:
      spatial_crops = self.config.get('evaluate', {}).get('spatial_crops', 1)
      temporal_samples = self.config.get('evaluate', {}).get('temporal_samples', 1)
    else:
      spatial_crops = 1
      temporal_samples = 1
    dup_samples = spatial_crops * temporal_samples

    self.model.cuda().eval()
    if self.config.gpus > 1:
      _net = self.model.module
    else:
      _net = self.model

    test_loader = self._build_dataloader(setname)
    test_len = len(test_loader)

    end = time.time()
    iter_idx = 0
    for batch in test_loader:
      inputs = [batch[1].cuda(non_blocking=True), batch[2].cuda(non_blocking=True)]
      isizes = inputs[0].shape

      if self.config.net.model_type == '2D':
        input_video = inputs[0].view(
          isizes[0] * dup_samples, -1, isizes[2], isizes[3])
        sample_len = (3 if _net.modality == "RGB" else 2) * _net.new_length
        if _net.modality == 'RGBDiff':
          sample_len = 3 * _net.new_length
          input_video = _net._get_diff(input_video)
        base_out = _net.base_model(input_video.view(
          (-1, sample_len) + input_video.size()[-2:])) # (n * dup * seg, d)

        if _net.is_shift and _net.temporal_pool:
          base_out = base_out.view(
            (-1, _net.num_segments // 2) + base_out.size()[1:])
        else:
          base_out = base_out.view(
            (-1, _net.num_segments) + base_out.size()[1:])

        fts = _net.consensus(base_out).squeeze(1) # (n * dup, d)
        logits = _net.new_fc(fts)

      else:
        inputs[0] = inputs[0].view(
          isizes[0], isizes[1], dup_samples, -1, isizes[3], isizes[4]
            )
        inputs[0] = inputs[0].permute(0, 2, 1, 3, 4, 5).contiguous()
        inputs[0] = inputs[0].view(isizes[0] * dup_samples, isizes[1], -1, isizes[3], isizes[4])

      fts = fts.view((fts.size(0) // dup_samples, -1, fts.size(1))) # (n, dup, d)
      logits = logits.view((logits.size(0), dup_samples, logits.size(1))) 
      logits = torch.mean(logits, 1) # (n, c)

      num = inputs[0].size(0)
      prec1, prec5 = accuracy(logits, inputs[1], topk=(1, 5))
      top1.update(prec1.item(), num)
      top5.update(prec5.item(), num)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      fts = fts.data.cpu().numpy()
      for videoname, ft in zip(batch[0], fts):
        with open(os.path.join(output_dir, '%s.npy'%videoname), 'wb') as outf:
          np.save(outf, ft)

      if iter_idx % self.config.trainer.print_freq == 0:
        self.logger.info('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
             iter_idx, test_len, batch_time=batch_time))
      iter_idx += 1

    total_num = torch.Tensor([top1.count]).cuda()
    top1_sum = torch.Tensor([top1.avg*top1.count]).cuda()
    top5_sum = torch.Tensor([top5.avg*top5.count]).cuda()
    if self.config.gpus > 1:
      all_reduce(total_num, False)
      all_reduce(top1_sum, False)
      all_reduce(top5_sum, False)
    final_top1 = top1_sum.item()/total_num.item()
    final_top5 = top5_sum.item()/total_num.item()
    self.logger.info(' * Prec@1 {:.3f}\tPrec@5 {:.3f}\ttotal_num={}'.format(final_top1,
      final_top5, total_num.item()))



def mrun(
  local_rank, num_proc, init_method, shard_id, num_shards, backend, config, setnames, output_dir
):  
  # Initialize the process group.
  world_size = num_proc * num_shards
  rank = shard_id * num_proc + local_rank

  try:
    torch.distributed.init_process_group(
      backend=backend,
      init_method=init_method,
      world_size=world_size,
      rank=rank,
    )
  except Exception as e:
    raise e

  torch.cuda.set_device(local_rank)
  temporal_helper = TemporalFeatureExtractor(config, inference_only=True)
  for setname in setnames:
    temporal_helper.extract_features(output_dir, setname=setname)


def main():
  parser = argparse.ArgumentParser(description='X-Temporal')
  parser.add_argument('--config', type=str, help='the path of config file')
  parser.add_argument("--shard_id", help="The shard id of current node, Starts from 0 to num_shards - 1",
      default=0, type=int)
  parser.add_argument("--num_shards", help="Number of shards using by the job",
      default=1, type=int)
  parser.add_argument("--init_method", help="Initialization method, includes TCP or shared file-system",
      default="tcp://localhost:9999", type=str)
  parser.add_argument('--dist_backend', default='nccl', type=str)
  parser.add_argument('--output_dir', required=True)
  parser.add_argument('--setnames', required=True, nargs='+')
  args = parser.parse_args()

  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  num_gpus = torch.cuda.device_count()
  config = EasyDict(config['config'])
  config.gpus = min(num_gpus, config.gpus)
  
  if config.gpus > 1:
    torch.multiprocessing.spawn(
        mrun,
        nprocs=config.gpus,
        args=(config.gpus,
          args.init_method,
          args.shard_id,
          args.num_shards,
          args.dist_backend,
          config,
          args.setnames,
          args.output_dir
          ),
        daemon=False)
  else:
    temporal_helper = TemporalFeatureExtractor(config, inference_only=True)
    for setname in args.setnames:
      temporal_helper.extract_features(args.output_dir, setname=setname)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method("forkserver")
  main()
