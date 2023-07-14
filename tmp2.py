import torch
import torch.distributed as dist
import os
import argparse

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
# os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()
# parser.add_argument("--node_rank", type=int)
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()


# 初始化进程组
# , init_method='file:///raid/ckh/VLN-HAMT/pretrain_src/datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks/.torch_distributed_sync' 
print("start to init")
dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=2, init_method="file:///raid/ckh/VLN-HAMT/tmp_folder/.sdf")
print("init finish")
world_size = dist.get_world_size()
rank = dist.get_rank()
print("rank: ", rank)
print("args.local_rank: ", args.local_rank)
print("world_size: ", world_size)

# 打印其他进程的信息
for i in range(world_size):
    if i != rank:
        print(f"Rank {rank}: Other process in the group: Rank {i}")

# 释放进程组资源
dist.destroy_process_group()