# pytorch_multi_gpu_training

```
>>> import torch
# next line in blocking until all processes join
>>> torch.distributed.init_process_group("nccl", init_method='file:///scratch/network/jdh4/sharedfile', rank=0, world_size=1)
>>> torch.distributed.is_initialized()
True
>>> torch.distributed.is_initialized()
True
```
