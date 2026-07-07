# FP8 ag_gemm and gemm_rs coverage via torchrun + RCCL.
# FP8 payloads move as uint8 views; compute dequantizes locally to fp16 and
# accumulates/collects in fp32 for correctness-first parity.
import os
import sys
import torch
import torch.distributed as dist


def bench(fn, iters=20, warmup=5):
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


def f8_dtype():
    return torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    assert M % world == 0 and K % world == 0
    Ms, Ks = M // world, K // world
    f8 = f8_dtype()

    torch.manual_seed(123)
    A8 = torch.randn(M, K, device=dev, dtype=torch.float16).mul_(0.25).to(f8)
    B8 = torch.randn(N, K, device=dev, dtype=torch.float16).mul_(0.25).to(f8)
    ref = A8.to(torch.float16) @ B8.to(torch.float16).t()
    tol = dict(atol=2e-2, rtol=2e-2)

    def ag_gemm():
        shard = A8[rank * Ms:(rank + 1) * Ms, :].contiguous()
        recv_u8 = torch.empty(M * K, device=dev, dtype=torch.uint8)
        dist.all_gather_into_tensor(recv_u8, shard.view(torch.uint8))
        gathered = recv_u8.view(f8).reshape(M, K)
        return gathered.to(torch.float16) @ B8.to(torch.float16).t()

    def gemm_rs():
        Ash = A8[:, rank * Ks:(rank + 1) * Ks].contiguous()
        Bsh = B8[:, rank * Ks:(rank + 1) * Ks].contiguous()
        partial = (Ash.to(torch.float16) @ Bsh.to(torch.float16).t()).float().contiguous()
        out = torch.empty(Ms, N, device=dev, dtype=torch.float32)
        dist.reduce_scatter(out, list(partial.chunk(world, 0)), op=dist.ReduceOp.SUM)
        return out

    Yag = ag_gemm()
    Yrs = gemm_rs()
    ag_ok = torch.allclose(Yag, ref, **tol)
    rs_ok = torch.allclose(Yrs, ref[rank * Ms:(rank + 1) * Ms, :].float(), **tol)
    flags = torch.tensor([ag_ok, rs_ok], device=dev, dtype=torch.int32)
    dist.all_reduce(flags)

    tag = bench(ag_gemm)
    trs = bench(gemm_rs)
    flop = 2.0 * M * N * K
    if rank == 0:
        ok = [int(v) == world for v in flags.tolist()]
        print(f"== FP8 GEMM collectives ({world} GPUs, M={M} N={N} K={K}, dtype={f8})")
        print(f"ag_gemm_fp8 {'PASS' if ok[0] else 'FAIL'}  {tag:.3f} ms  {flop/(tag*1e-3)/1e12:.2f} TFLOP/s")
        print(f"gemm_rs_fp8 {'PASS' if ok[1] else 'FAIL'}  {trs:.3f} ms  {flop/(trs*1e-3)/1e12:.2f} TFLOP/s")
        print("ALL PASS" if all(ok) else "FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
