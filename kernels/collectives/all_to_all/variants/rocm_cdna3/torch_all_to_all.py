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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 18
    chunks = [torch.full((n,), rank * 1000.0 + dst, device=dev) for dst in range(world)]
    inp = torch.cat(chunks, 0).contiguous()
    out = torch.empty_like(inp)

    dist.all_to_all_single(out, inp)
    ok = True
    for src, chunk in enumerate(out.chunk(world)):
        ok = ok and torch.equal(chunk, torch.full((n,), src * 1000.0 + rank, device=dev))
    flags = torch.tensor([ok], device=dev, dtype=torch.int32)
    dist.all_reduce(flags)
    t = bench(lambda: dist.all_to_all_single(out, inp))
    if rank == 0:
        passed = int(flags.item()) == world
        print(f"== all_to_all RCCL ({world} GPUs, chunk={n})")
        print(f"correctness: {'PASS' if passed else 'FAIL'}")
        print(f"time: {t:.3f} ms  payload/rank: {inp.numel() * inp.element_size() / 1e6:.1f} MB")
        print("ALL PASS" if passed else "FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
