import torch
import torch.distributed as dist

def mrun(local_rank, num_proc, init_method, shard_id, num_shards, backend, 
  func, func_args):
  """
  Runs a function from a child process.
  Args:
    local_rank (int): rank of the current process on the current machine.
    num_proc (int): number of processes per machine.
    init_method (string): method to initialize the distributed training.
      TCP initialization: equiring a network address reachable from all
      processes followed by the port.
      Shared file-system initialization: makes use of a file system that
      is shared and visible from all machines. The URL should start with
      file:// and contain a path to a non-existent file on a shared file
      system.
    shard_id (int): the rank of the current machine.
    num_shards (int): number of overall machines for the distributed
      training job.
    backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
      supports, each with different capabilities. Details can be found
      here: https://pytorch.org/docs/stable/distributed.html
    func: a function
    func_args: dict-based arguments, e.g. trn_dataset=xxx, val_dataset=xxx
  """
  world_size = num_proc * num_shards
  rank = shard_id * num_proc + local_rank

  try:
    dist.init_process_group(
      backend=backend,
      init_method=init_method,
      world_size=world_size,
      rank=rank
    )
  except Exception as e:
    raise e

  torch.cuda.set_device(local_rank)
  func(**func_args)

