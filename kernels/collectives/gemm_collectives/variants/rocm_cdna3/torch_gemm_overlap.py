# CDNA3 compute/comm OVERLAP for the fused collective+GEMM path (torchrun +
# torch.distributed RCCL). Overlaps the collective with the GEMM by chunking
# along N and issuing per-chunk async collectives: chunk c's collective runs on
# RCCL's stream while chunk c+1's GEMM runs on the compute stream. Validates
# overlapped == single-GPU ref, then A/Bs overlapped vs non-overlapped.
#   torchrun --nproc_per_node=<P> torch_gemm_overlap.py [M N K CHUNKS ITERS]
import os, sys, torch, torch.distributed as dist

def bench(fn, iters, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms

def main():
    dist.init_process_group("nccl")
    r, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r); dev = torch.device(f"cuda:{r}")
    a = sys.argv
    M = int(a[1]) if len(a) > 1 else 4096
    N = int(a[2]) if len(a) > 2 else 4096
    K = int(a[3]) if len(a) > 3 else 4096
    C = int(a[4]) if len(a) > 4 else 4          # N-chunks
    IT = int(a[5]) if len(a) > 5 else 30
    assert N % C == 0 and K % W == 0 and M % W == 0
    Ks, Ms, cw = K // W, M // W, N // C
    torch.manual_seed(0)                         # identical A,B on every rank
    A = torch.randn(M, K, device=dev); B = torch.randn(K, N, device=dev)
    ref = A @ B
    Ak = A[:, r*Ks:(r+1)*Ks].contiguous()        # K-shard of A (this rank)
    Bk = B[r*Ks:(r+1)*Ks, :].contiguous()        # K-shard of B (this rank)
    tol = dict(atol=2e-2, rtol=2e-2)

    # ---- gemm_ar: K-parallel local GEMM + all_reduce(sum) ----
    def ar_base():
        Y = Ak @ Bk
        dist.all_reduce(Y)
        return Y
    def ar_overlap():
        outs, works = [], []
        for c in range(C):
            Yc = (Ak @ Bk[:, c*cw:(c+1)*cw]).contiguous()   # partial for this N-chunk
            w = dist.all_reduce(Yc, async_op=True)           # overlaps next chunk's GEMM
            outs.append(Yc); works.append(w)
        for w in works: w.wait()
        return torch.cat(outs, 1)

    # ---- gemm_rs: K-parallel partial GEMM + reduce_scatter over M ----
    def rs_base():
        part = (Ak @ Bk).contiguous()
        out = torch.empty(Ms, N, device=dev)
        dist.reduce_scatter(out, list(part.chunk(W, 0)))
        return out
    def rs_overlap():
        outs, works = [], []
        for c in range(C):
            part = (Ak @ Bk[:, c*cw:(c+1)*cw]).contiguous()
            oc = torch.empty(Ms, cw, device=dev)
            w = dist.reduce_scatter(oc, list(part.chunk(W, 0)), async_op=True)
            outs.append(oc); works.append(w)
        for w in works: w.wait()
        return torch.cat(outs, 1)

    ar_ok = torch.allclose(ar_overlap(), ref, **tol)
    rs_ok = torch.allclose(rs_overlap(), ref[r*Ms:(r+1)*Ms, :], **tol)

    tb_ar, to_ar = bench(ar_base, IT), bench(ar_overlap, IT)
    tb_rs, to_rs = bench(rs_base, IT), bench(rs_overlap, IT)
    flags = torch.tensor([ar_ok, rs_ok], device=dev, dtype=torch.int)
    dist.all_reduce(flags)
    flop = 2.0 * M * N * K
    if r == 0:
        ok = [int(x) == W for x in flags.tolist()]
        print(f"== collective+GEMM overlap ({W} GPUs, M={M} N={N} K={K}, C={C} chunks)")
        print(f"gemm_ar: {'PASS' if ok[0] else 'FAIL'}  base {tb_ar:.3f} ms ({flop/(tb_ar*1e-3)/1e12:.1f} TF) "
              f"-> overlap {to_ar:.3f} ms ({flop/(to_ar*1e-3)/1e12:.1f} TF)  {tb_ar/to_ar:.2f}x")
        print(f"gemm_rs: {'PASS' if ok[1] else 'FAIL'}  base {tb_rs:.3f} ms ({flop/(tb_rs*1e-3)/1e12:.1f} TF) "
              f"-> overlap {to_rs:.3f} ms ({flop/(to_rs*1e-3)/1e12:.1f} TF)  {tb_rs/to_rs:.2f}x")
        print("ALL PASS" if all(ok) else "FAILED")
    dist.destroy_process_group()

main()
