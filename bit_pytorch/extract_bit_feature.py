import os
import time
import argparse
import jsonlines
import json
import shutil

import cv2
from PIL import Image

from progressbar import ProgressBar

import numpy as np
import decord

import torch
import torchvision as tv
import torch.multiprocessing as mp

import bit_pytorch.models as models


class VideoDataset(torch.utils.data.Dataset):
  def __init__(self, video_dir, video_list, num_frames_per_video):
    super().__init__()
    self.video_dir = video_dir
    self.video_list = video_list
    self.num_frames_per_video = num_frames_per_video

    self.transform = tv.transforms.Compose([
      tv.transforms.Resize(256),
      tv.transforms.CenterCrop(224),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

  def __len__(self):
    return len(self.video_list)

  def __getitem__(self, idx):
    videoname = self.video_list[idx]['videoname']
    nframes = self.video_list[idx]['nframes']

    vid = decord.VideoReader(os.path.join(self.video_dir, videoname), ctx=decord.cpu(0))
    if nframes <= self.num_frames_per_video:
      idxs = np.arange(0, nframes).astype(np.int32)
    else:
      idxs = np.linspace(0, nframes-1, self.num_frames_per_video)
      idxs = np.round(idxs).astype(np.int32)

    images = []
    for k in idxs:
      frame = Image.fromarray(vid[k].asnumpy())
      frame = self.transform(frame)
      images.append(frame)
    images = torch.stack(images, 0)

    name = os.path.splitext(videoname)[0]

    return name, images


def extract_image_features(proc_id, log_queue, video_list, device, args):
  print(f"Process{proc_id} starts to extract features using GPU{device}")

  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True
  device = torch.device("cuda:%d" % device)
  torch.set_grad_enabled(False)

  print(f"\tProcess{proc_id}: loading model from {args.model}.npz")
  model = models.KNOWN_MODELS[args.model]()
  model.load_from(np.load(
    os.path.join(args.model_dir, f"{args.model}.npz")))
  model = model.to(device)
  model.eval()

  video_dataset = VideoDataset(
    args.video_dir, video_list, args.num_frames_per_video)
  video_loader = torch.utils.data.DataLoader(
    video_dataset, batch_size=1, shuffle=False, 
    drop_last=False, num_workers=args.num_data_workers, pin_memory=True)
  
  output_dir = args.output_dir
  os.makedirs(output_dir, exist_ok=True)
  i = 0
  for name, images in video_loader:
    name = name[0]
    images = images[0]

    if os.path.exists(os.path.join(output_dir, name+'.npy')):
      log_queue.put(name)
      continue
    
    images = images.to(device)
    x = model.body(model.root(images))
    x = model.head.gn(x)
    x = model.head.relu(x)
    fts = model.head.avg(x)
    logits = model.head.conv(fts)[...,0,0]
    
    logits = torch.mean(logits, 0, keepdim=True).data.cpu().numpy()
    
    with open(os.path.join(output_dir, name+'.npy'), 'wb') as outf:
      np.save(outf, logits)

    log_queue.put(name)

  log_queue.put(None)

  
def gather_data(args, video_list):
  labels = open(os.path.join(args.model_dir, 'imagenet21k_wordnet_lemmas.txt')).readlines()
  labels = [x.strip() for x in labels]

  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  outs = {}
  for item in video_list:
    name = os.path.splitext(item['videoname'])[0]
    if not os.path.exists(os.path.join(args.output_dir, name+'.npy')):
      print(name)
      continue
    logits = np.load(os.path.join(args.output_dir, name+'.npy'))
    assert logits.shape[0] == 1
    logits = logits[0]
    idxs = np.argsort(-logits)[:20]
    outs[name] = [idxs.tolist(), sigmoid(logits[idxs]).tolist()]
    # outs[name] = []
    # for idx in idxs[:20]:
    #   outs[name].append((sigmoid(logits[idx]), labels[idx]))

  print(f'Gathered %d data' % (len(outs)))

  json.dump(outs, open(os.path.join(args.output_dir, '..', 
    '%s_top20_preds.json'%(args.model.lower())), 'w'))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True)
  parser.add_argument('--model_dir')
  parser.add_argument('--video_dir', required=True)
  parser.add_argument('--video_meta_file', required=True)
  parser.add_argument('--num_frames_per_video', type=int, default=8)
  parser.add_argument('--output_dir', required=True)
  parser.add_argument('--num_workers', type=int, default=8)
  parser.add_argument('--num_data_workers', type=int, default=2)
  args = parser.parse_args()

  video_list, todo_video_list = [], []
  for setname in ['trn', 'val', 'tst']:
    with jsonlines.open(args.video_meta_file.format(setname), 'r') as f:
      for item in f:
        video_list.append(item)
        if not os.path.exists(os.path.join(args.output_dir, os.path.splitext(item['videoname'])[0]+'.npy')):
          todo_video_list.append(item)
  print('total videos: %d, todo videos: %d' % (len(video_list), len(todo_video_list)))

  if len(todo_video_list) > 0:
    mp.set_start_method('spawn')
    log_queue = mp.Queue()

    num_workers = min(len(todo_video_list), args.num_workers)
    avg_videos_per_worker = len(todo_video_list) // num_workers
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, 'No GPU available'

    processes = []
    for i in range(num_workers):
      sidx = avg_videos_per_worker * i
      eidx = None if i == num_workers - 1 else sidx + avg_videos_per_worker
      device = i % num_gpus

      process = mp.Process(
        target=extract_image_features, args=(i, log_queue, todo_video_list[sidx: eidx], device, args)
      )
      process.start()
      processes.append(process)

    progress_bar = ProgressBar(max_value=len(todo_video_list))
    progress_bar.start()

    num_finished_workers, num_finished_files = 0, 0
    while num_finished_workers < num_workers:
      res = log_queue.get()
      if res is None:
        num_finished_workers += 1
      else:
        num_finished_files += 1
        progress_bar.update(num_finished_files)

    progress_bar.finish()

    for i in range(num_workers):
      processes[i].join()

  # Gather data
  gather_data(args, video_list)
  print('Removing worker directories')
  if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)

if __name__ == "__main__":
  main()
