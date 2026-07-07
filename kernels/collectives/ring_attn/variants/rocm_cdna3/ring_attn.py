# CDNA3 ring attention (sequence-parallel) via torchrun + torch.distributed RCCL,
# one process per GPU. The sequence N is sharded across W ranks; each rank holds
# its Q/K/V shard, ring-rotates the KV shard around all W ranks (deadlock-free
# batch_isend_irecv), and merges each block's partial attention with online
# softmax. After W steps each rank has the full-context attention for its Q rows.
# Validated allclose vs a single-GPU full-attention reference.
#   torchrun --nproc_per_node=<P> ring_attn.py [N H D]
import sys, torch, torch.distributed as dist

def ring_exchange(t, r, W):                       # send->(r+1), recv<-(r-1)
    recv = torch.empty_like(t)
    ops = [dist.P2POp(dist.isend, t.contiguous(), (r + 1) % W),
           dist.P2POp(dist.irecv, recv, (r - 1) % W)]
    for w in dist.batch_isend_irecv(ops): w.wait()
    return recv

def full_attn(Q, K, V, scale):                    # [N,H,D] -> [N,H,D]
    q, k, v = (X.permute(1, 0, 2) for X in (Q, K, V))   # [H,N,D]
    a = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
    return (a @ v).permute(1, 0, 2)

def main():
    dist.init_process_group("nccl")
    r, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r); dev = torch.device(f"cuda:{r}")
    a = sys.argv
    N = int(a[1]) if len(a) > 1 else 2048
    H = int(a[2]) if len(a) > 2 else 8
    D = int(a[3]) if len(a) > 3 else 128
    assert N % W == 0; Ms = N // W; scale = 1.0 / D ** 0.5
    torch.manual_seed(0)                           # identical full tensors on every rank
    Q = torch.randn(N, H, D, device=dev); K = torch.randn(N, H, D, device=dev); V = torch.randn(N, H, D, device=dev)
    ref = full_attn(Q, K, V, scale)[r * Ms:(r + 1) * Ms]      # [Ms,H,D] this rank's rows

    sl = slice(r * Ms, (r + 1) * Ms)
    q = Q[sl].permute(1, 0, 2).contiguous()        # [H,Ms,D]
    kv = torch.stack([K[sl].permute(1, 0, 2), V[sl].permute(1, 0, 2)]).contiguous()  # [2,H,Ms,D]

    o = torch.zeros(H, Ms, D, device=dev)
    m = torch.full((H, Ms), -3e38, device=dev)
    l = torch.zeros(H, Ms, device=dev)
    for step in range(W):
        k, v = kv[0], kv[1]                         # [H,Ms,D]
        s = (q @ k.transpose(-1, -2)) * scale       # [H,Ms,Ms] (query i, key j)
        m_new = torch.maximum(m, s.max(-1).values)  # [H,Ms]
        p = torch.exp(s - m_new.unsqueeze(-1))       # [H,Ms,Ms]
        corr = torch.exp(m - m_new)                  # [H,Ms]
        l = l * corr + p.sum(-1)
        o = o * corr.unsqueeze(-1) + p @ v           # [H,Ms,D]
        m = m_new
        if step < W - 1: kv = ring_exchange(kv, r, W)
    o = (o / l.unsqueeze(-1)).permute(1, 0, 2)       # [Ms,H,D]

    ok = torch.allclose(o, ref, atol=1e-2, rtol=1e-2)
    flags = torch.tensor([ok], device=dev, dtype=torch.int); dist.all_reduce(flags)
    if r == 0:
        passed = int(flags.item()) == W
        maxerr = (o - ref).abs().max().item()
        print(f"== ring attention (torchrun, {W} GPUs, N={N} H={H} D={D}, shard {Ms}/rank)")
        print(f"ring_attn: {'PASS' if passed else 'FAIL'}  (rank0 max abs err {maxerr:.3e} vs single-GPU full attn)")
        print("ALL PASS" if passed else "FAILED")
    dist.destroy_process_group()

main()
