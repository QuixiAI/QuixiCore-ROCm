# Step 8 feasibility: Iris device-initiated XGMI all_reduce (Triton, one-sided over
# the symmetric heap) vs RCCL all_reduce. Motivated by the Step 5 finding that RCCL
# collectives run on the CUs and contend with the GEMM. torchrun one-process-per-GPU.
#   torchrun --nproc_per_node=<P> iris_allreduce.py [M N]
import os, sys, torch, torch.distributed as dist, iris, iris.ccl

def bench(fn, it=30, w=10):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it

def main():
    dist.init_process_group("nccl")                # Iris requires torch.distributed already up
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    ctx = iris.iris(1 << 32)                        # symmetric heap over the existing process group
    r, W = ctx.get_rank(), ctx.get_num_ranks()
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    x = ctx.full((M, N), float(r + 1))             # rank-distinct, on the symmetric heap
    out = ctx.zeros((M, N))
    ctx.ccl.all_reduce(out, x, op=iris.ccl.ReduceOp.SUM)
    ctx.barrier()
    expect = W * (W + 1) / 2.0                      # sum_{r=1..W} r
    ok = torch.allclose(out, torch.full_like(out, expect))

    # A/B vs RCCL all_reduce (same buffer, torch.distributed already up via iris)
    ti = bench(lambda: ctx.ccl.all_reduce(out, x, op=iris.ccl.ReduceOp.SUM))
    xr = x.clone().contiguous()
    tr = bench(lambda: dist.all_reduce(xr.clone()))
    bytes_moved = 2.0 * (W - 1) / W * M * N * 4     # ring all_reduce traffic model
    if r == 0:
        print(f"== Iris XGMI all_reduce vs RCCL ({W} GPUs, M={M} N={N} f32)")
        print(f"correctness: {'PASS' if ok else 'FAIL'}  (expect {expect}, got {out[0,0].item()})")
        print(f"iris : {ti:.3f} ms  {bytes_moved/(ti*1e-3)/1e9:.0f} GB/s")
        print(f"rccl : {tr:.3f} ms  {bytes_moved/(tr*1e-3)/1e9:.0f} GB/s")
        print(f"iris/rccl speedup: {tr/ti:.2f}x")
    ctx.barrier()

main()
