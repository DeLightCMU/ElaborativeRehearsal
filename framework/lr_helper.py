import torch

class ExponentialWarmUpLR(object):
  def __init__(self, warmup_iter, init_lr, target_lr):
    self.lr_scale = target_lr / init_lr
    self.gamma = self.lr_scale**(1.0 / max(1, warmup_iter))
    self.warmup_iter = warmup_iter

  def get_lr(self, last_iter, base_lrs):
    return [base_lr * self.gamma**last_iter / self.lr_scale for base_lr in base_lrs]

class LinearWarmUpLR(object):
  def __init__(self, warmup_iter, init_lr, target_lr):
    self.lr_gap = target_lr - init_lr
    self.gamma = self.lr_gap / max(1, warmup_iter)
    self.warmup_iter = warmup_iter

  def get_lr(self, last_iter, base_lrs):
    return [base_lr + self.gamma * last_iter - self.lr_gap for base_lr in base_lrs]

_warmup_lr = {
  'linear': LinearWarmUpLR,
  'exp': ExponentialWarmUpLR,
}

class ConstantLRScheduler(torch.optim.lr_scheduler._LRScheduler):
  def get_lr(self):
    return [group['lr'] for group in self.optimizer.param_groups]

def build_scheduler(cfg_scheduler, optimizer, base_lr, batch_size):
  target_lr = base_lr
  warmup_scheduler = _warmup_lr[cfg_scheduler['warmup_type']](
    cfg_scheduler['warmup_iter'], 
    target_lr / batch_size, target_lr)

  if cfg_scheduler['type'] is None:
    standard_scheduler_class = ConstantLRScheduler
  else:
    standard_scheduler_class = getattr(torch.optim.lr_scheduler, cfg_scheduler['type'])

  class ChainIterLR(standard_scheduler_class):
    def __init__(self, *args, **kwargs):
      super(ChainIterLR, self).__init__(*args, **kwargs)

    def get_lr(self):
      if warmup_scheduler.warmup_iter > 0 and self.last_iter <= warmup_scheduler.warmup_iter:
        return warmup_scheduler.get_lr(self.last_iter, self.base_lrs)
      else:
        return super().get_lr()

    @property
    def last_iter(self):
      return self.last_epoch

  return ChainIterLR(optimizer, **cfg_scheduler['kwargs'])




