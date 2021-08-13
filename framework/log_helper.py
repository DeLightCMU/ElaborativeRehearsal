import os
import logging

import numpy as np
try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x

import tensorboardX as tb

def set_logger(log_path, log_name='training'):
  if log_path is None:
    print('log_path is empty')
    return None
    
  if os.path.exists(log_path):
    print('%s already exists'%log_path)
    return None

  logger = logging.getLogger(log_name)
  logger.setLevel(logging.DEBUG)

  logfile = logging.FileHandler(log_path)
  console = logging.StreamHandler()
  logfile.setLevel(logging.INFO)
  logfile.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  console.setLevel(logging.DEBUG)
  console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  logger.addHandler(logfile)
  logger.addHandler(console)
  
  logger.propagate = False
  return logger


class TensorboardLogger(object):
  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    self.writer = tb.writer.SummaryWriter(log_dir=log_dir)

  def scalar_summary(self, key, value, step):
    """Log a scalar variable."""
    self.writer.add_scalar(key, value, global_step=step)
    self.writer.flush()

  def image_summary(self, key, images, step, dataformats='NCHW'):
    """Log images."""
    self.writer.add_images(key, images, global_step=step, dataformats=dataformats)
    self.writer.flush()
        
  def histogram_summary(self, key, values, step, bins=1000):
    """Log a histogram of the tensor of values."""
    self.writer.add_summary(key, values, global_step=step)
    self.writer.flush()
