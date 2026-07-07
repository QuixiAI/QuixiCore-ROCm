# CDNA3 expert-parallel MoE dispatch GEMM via torchrun + torch.distributed RCCL,
# one process per GPU. Experts are sharded across ranks (E/W experts each). Each
# rank holds T local tokens with per-token expert ids; tokens are all_to_all_v
# dispatched to the rank owning their expert, run through that expert's GEMM
# (grouped by expert), then all_to_all_v returned and unsorted to original order.
# Variable per-expert/per-rank counts are exchanged first, then used as
# all_to_all_single split sizes. Each rank validates its returned tokens vs a local
# reference (X @ W[expert]).
#   torchrun --nproc_per_node=<P> moe_dispatch_gemm.py [E T D]   (requires E % P == 0)
import sys, torch, torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    r, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r); dev = torch.device(f"cuda:{r}")
    a = sys.argv
    E = int(a[1]) if len(a) > 1 else 8
    T = int(a[2]) if len(a) > 2 else 512
    D = int(a[3]) if len(a) > 3 else 256
    assert E % W == 0, "experts must be divisible by world size"
    El = E // W                                    # experts per rank
    # full expert weight table identical on every rank (seed 0); local slice = experts [r*El, (r+1)*El)
    g0 = torch.Generator(device=dev).manual_seed(0)
    Wt = torch.randn(E, D, D, generator=g0, device=dev) * (D ** -0.5)
    # per-rank distinct tokens + expert assignment
    gr = torch.Generator(device=dev).manual_seed(100 + r)
    X = torch.randn(T, D, generator=gr, device=dev)
    eid = torch.randint(0, E, (T,), generator=gr, device=dev)      # global expert id per token
    ref = torch.empty(T, D, device=dev)                            # local reference
    for t_e in range(E):
        msk = eid == t_e
        if msk.any(): ref[msk] = X[msk] @ Wt[t_e]

    # group local tokens by destination rank (= expert // El), stable so we can unsort
    dst = (eid // El).to(torch.int64)
    order = torch.argsort(dst, stable=True)
    Xs, eids, dsts = X[order], eid[order], dst[order]
    send_cnt = torch.bincount(dsts, minlength=W).to(torch.int64)   # tokens to each rank
    recv_cnt = torch.empty(W, dtype=torch.int64, device=dev)
    dist.all_to_all_single(recv_cnt, send_cnt)                     # learn incoming counts
    sc, rc = send_cnt.tolist(), recv_cnt.tolist()
    Rtot = int(recv_cnt.sum())

    # dispatch tokens + their expert ids to the owning ranks
    Xr = torch.empty(Rtot, D, device=dev)
    dist.all_to_all_single(Xr, Xs.contiguous(), rc, sc)
    er = torch.empty(Rtot, dtype=torch.int64, device=dev)
    dist.all_to_all_single(er, eids.contiguous(), rc, sc)

    # local grouped GEMM: each received token through its (local) expert
    Yr = torch.empty(Rtot, D, device=dev)
    for le in range(El):
        ge = r * El + le
        msk = er == ge
        if msk.any(): Yr[msk] = Xr[msk] @ Wt[ge]

    # return results to source ranks (counts swapped), then unsort
    Ys = torch.empty(T, D, device=dev)
    dist.all_to_all_single(Ys, Yr.contiguous(), sc, rc)
    Y = torch.empty_like(Ys); Y[order] = Ys

    ok = torch.allclose(Y, ref, atol=1e-2, rtol=1e-2)
    flags = torch.tensor([ok], device=dev, dtype=torch.int); dist.all_reduce(flags)
    if r == 0:
        passed = int(flags.item()) == W
        print(f"== MoE dispatch GEMM (torchrun, {W} GPUs, E={E} experts, {El}/rank, T={T} tok/rank, D={D})")
        print(f"moe_dispatch_gemm: {'PASS' if passed else 'FAIL'}  (rank0 max abs err {(Y-ref).abs().max().item():.3e})")
        print("ALL PASS" if passed else "FAILED")
    dist.destroy_process_group()

main()
