import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    device = torch.device("cpu")
    torch.random.manual_seed(42 + rank)
    torch.manual_seed(42 + rank)

    # all reduce
    # All rank must have the same shape tensor
    data_reduce = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data_reduce}")
    dist.all_reduce(data_reduce, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data_reduce}")
    dist.barrier()

    # all gather
    # All rank must have a gather list which has the same length as world_size
    data_gather = torch.randint(0, 10, (3,))
    gather_list = [torch.zeros_like(data_gather) for _ in range(world_size)]
    print(f"rank {rank} data (before all-gather): {data_gather}")
    dist.all_gather(gather_list, data_gather, async_op=False)
    print(f"rank {rank} data (after all-gather): {data_gather}")
    dist.barrier()

    # scatter
    # Only rank 0 needs to provide the scatter list with length equal to world_size
    if rank == 0:
        scatter_list = [torch.tensor([i, i + 1]) for i in range(world_size)]
    else:
        scatter_list = None
    recv_tensor = torch.zeros(2, dtype=torch.int64)
    dist.scatter(recv_tensor, scatter_list, src=0)
    print(f"rank {rank} after scatter: {recv_tensor}")
    dist.barrier()

    # all to all
    input_tensor = torch.tensor([i for i in range(world_size)])
    output_tensor = torch.zeros(world_size, dtype=torch.int64)
    dist.all_to_all_single(output_tensor, input_tensor)
    print(f"rank {rank} after all_to_all: {output_tensor}")
    dist.barrier()

    # broadcast
    # All rank must have the same shape tensor
    if rank == 0:
        data_broadcast = torch.randint(0, 10, (3,))
    else:
        data_broadcast = torch.zeros(3, dtype=torch.int64)
    print(f"rank {rank} data (before broadcast): {data_broadcast}")
    dist.broadcast(data_broadcast, src=0, async_op=False)
    print(f"rank {rank} data (after broadcast): {data_broadcast}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(
        fn=distributed_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )