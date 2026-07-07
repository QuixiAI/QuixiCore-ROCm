# CDNA3 fused collective+GEMM via one-process-per-GPU (torchrun + torch.distributed
# NCCL/RCCL backend) - the production tensor/sequence-parallel model. Validates
# gemm_ar / ag_gemm / gemm_rs vs a single-GPU reference.
#   torchrun --nproc_per_node=<P> torch_gemm_collectives.py
import torch, torch.distributed as dist
def main():
    dist.init_process_group("nccl")
    r, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r); dev = torch.device(f"cuda:{r}")
    M, N, K = 256, 256, 512; Ks, Ms = K//W, M//W
    torch.manual_seed(0)                       # identical A,B on every rank
    A = torch.randn(M, K, device=dev); B = torch.randn(K, N, device=dev)
    ref = A @ B
    tol = dict(atol=1e-2, rtol=1e-2)
    # gemm_ar: K-parallel local GEMM + all_reduce(sum)
    Yar = (A[:, r*Ks:(r+1)*Ks] @ B[r*Ks:(r+1)*Ks, :]).contiguous()
    dist.all_reduce(Yar)
    ar = torch.allclose(Yar, ref, **tol)
    # ag_gemm: all_gather A row-shards -> full A, then GEMM
    As = A[r*Ms:(r+1)*Ms, :].contiguous()
    gl = [torch.empty_like(As) for _ in range(W)]; dist.all_gather(gl, As)
    Yag = torch.cat(gl, 0) @ B
    ag = torch.allclose(Yag, ref, **tol)
    # gemm_rs: K-parallel partial GEMM + reduce_scatter over M rows
    part = (A[:, r*Ks:(r+1)*Ks] @ B[r*Ks:(r+1)*Ks, :]).contiguous()
    out = torch.empty(Ms, N, device=dev)
    dist.reduce_scatter(out, list(part.chunk(W, 0)))
    rs = torch.allclose(out, ref[r*Ms:(r+1)*Ms, :], **tol)
    flags = torch.tensor([ar, ag, rs], device=dev, dtype=torch.int)
    dist.all_reduce(flags)                      # AND across ranks (min == W means all true)
    if r == 0:
        ok = [int(x)==W for x in flags.tolist()]
        print(f"== fused collective+GEMM (torchrun, {W} GPUs, M={M} N={N} K={K})")
        print(f"gemm_ar {'PASS' if ok[0] else 'FAIL'} | ag_gemm {'PASS' if ok[1] else 'FAIL'} | gemm_rs {'PASS' if ok[2] else 'FAIL'}")
        print("ALL PASS" if all(ok) else "FAILED")
    dist.destroy_process_group()
main()
