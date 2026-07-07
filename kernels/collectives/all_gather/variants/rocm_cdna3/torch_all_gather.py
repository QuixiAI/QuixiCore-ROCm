import os
import sys
import torch
import torch.distributed as dist


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
    x = torch.arange(n, device=dev, dtype=torch.float32) + rank * 1000.0
    out = torch.empty(world * n, device=dev, dtype=torch.float32)

    dist.all_gather_into_tensor(out, x)
    ok = True
    for src, chunk in enumerate(out.chunk(world)):
        ok = ok and torch.equal(chunk, torch.arange(n, device=dev, dtype=torch.float32) + src * 1000.0)
    flags = torch.tensor([ok], device=dev, dtype=torch.int32)
    dist.all_reduce(flags)
    t = bench(lambda: dist.all_gather_into_tensor(out, x))
    bytes_moved = world * n * x.element_size()
    if rank == 0:
        passed = int(flags.item()) == world
        print(f"== all_gather RCCL ({world} GPUs, n={n})")
        print(f"correctness: {'PASS' if passed else 'FAIL'}")
        print(f"time: {t:.3f} ms  payload/rank: {bytes_moved / 1e6:.1f} MB")
        print("ALL PASS" if passed else "FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
