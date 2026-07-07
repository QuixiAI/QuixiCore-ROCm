# CDNA3 Ulysses attention (DeepSpeed-Ulysses sequence parallelism) via torchrun +
# torch.distributed RCCL, one process per GPU. Input is sequence-sharded ([Ms,H,D]
# per rank); an all_to_all reshards it to head-parallel ([N,Hs,D], all sequence /
# a head subset), local full self-attention runs on that head subset, then a second
# all_to_all reshards the output back to sequence-sharded. Validated allclose vs a
# single-GPU full-attention reference.
#   torchrun --nproc_per_node=<P> ulysses_attn.py [N H D]   (requires H % P == 0)
import sys, torch, torch.distributed as dist

def full_attn(Q, K, V, scale):                    # [N,H,D] -> [N,H,D]
    q, k, v = (X.permute(1, 0, 2) for X in (Q, K, V))
    a = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
    return (a @ v).permute(1, 0, 2)

def a2a(chunks):                                  # all_to_all a list of W equal tensors
    recv = [torch.empty_like(c) for c in chunks]
    dist.all_to_all(recv, chunks)
    return recv

def main():
    dist.init_process_group("nccl")
    r, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r); dev = torch.device(f"cuda:{r}")
    a = sys.argv
    N = int(a[1]) if len(a) > 1 else 2048
    H = int(a[2]) if len(a) > 2 else 8
    D = int(a[3]) if len(a) > 3 else 128
    assert N % W == 0 and H % W == 0, "N and H must be divisible by world size"
    Ms, Hs = N // W, H // W; scale = 1.0 / D ** 0.5
    torch.manual_seed(0)
    Q = torch.randn(N, H, D, device=dev); K = torch.randn(N, H, D, device=dev); V = torch.randn(N, H, D, device=dev)
    ref = full_attn(Q, K, V, scale)[r * Ms:(r + 1) * Ms]     # [Ms,H,D]

    sl = slice(r * Ms, (r + 1) * Ms)
    Qr, Kr, Vr = Q[sl].contiguous(), K[sl].contiguous(), V[sl].contiguous()  # [Ms,H,D]
    # all_to_all: split local seq-shard by head-group g -> rank g; gather sources -> full seq
    def to_head_parallel(X):                        # [Ms,H,D] -> [N,Hs,D]
        recv = a2a([X[:, g*Hs:(g+1)*Hs, :].contiguous() for g in range(W)])  # W x [Ms,Hs,D]
        return torch.cat(recv, dim=0)               # [N,Hs,D]
    Qh, Kh, Vh = to_head_parallel(Qr), to_head_parallel(Kr), to_head_parallel(Vr)
    Oh = full_attn(Qh, Kh, Vh, scale)               # [N,Hs,D] local attention over full seq, head subset
    # all_to_all back: split by source-seq-shard s -> rank s; gather head-groups -> full heads
    recv = a2a([Oh[s*Ms:(s+1)*Ms, :, :].contiguous() for s in range(W)])     # W x [Ms,Hs,D]
    Or = torch.cat(recv, dim=1)                     # [Ms,H,D]

    ok = torch.allclose(Or, ref, atol=1e-2, rtol=1e-2)
    flags = torch.tensor([ok], device=dev, dtype=torch.int); dist.all_reduce(flags)
    if r == 0:
        passed = int(flags.item()) == W
        print(f"== Ulysses attention (torchrun, {W} GPUs, N={N} H={H} D={D}: seq {Ms}/rank, heads {Hs}/rank)")
        print(f"ulysses_attn: {'PASS' if passed else 'FAIL'}  (rank0 max abs err {(Or-ref).abs().max().item():.3e})")
        print("ALL PASS" if passed else "FAILED")
    dist.destroy_process_group()

main()
