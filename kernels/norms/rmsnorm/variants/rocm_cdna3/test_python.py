import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import rms_norm_kernel

B = 16
N = 4096
D = 128
EPSILON = 1e-6

torch.random.manual_seed(42)
x = torch.randn((1, B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
gamma_1d = torch.ones(D, dtype=torch.bfloat16, device='cuda')
gamma = gamma_1d.view(1, 1, 1, D).expand(1, B, N, D).contiguous()


def flops(batch, seqlen, hidden_dim):
    """Calculate FLOPs for RMSNorm operation."""
    B, N, D = batch, seqlen, hidden_dim
    # Square: B * N * D
    # Mean: B * N * D (sum + divide)
    # Rsqrt: B * N (one per sequence position)
    # Multiply by rsqrt: B * N * D
    # Multiply by gamma: B * N * D
    square_flops = B * N * D
    mean_flops = B * N * D  # sum operations
    rsqrt_flops = B * N
    norm_flops = B * N * D  # multiply by rsqrt
    scale_flops = B * N * D  # multiply by gamma
    total_flops = square_flops + mean_flops + rsqrt_flops + norm_flops + scale_flops
    return total_flops


def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time


def pytorch_rmsnorm(x, gamma, eps=EPSILON):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * gamma


start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, D)
num_warmup = 50
num_iters = 50


# Benchmark and test correctness
# PyTorch
timings = []
print("\nPyTorch:")
for _ in range(num_warmup):
    o_ref = pytorch_rmsnorm(x, gamma_1d.view(1, 1, 1, D), EPSILON)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    o_ref = pytorch_rmsnorm(x, gamma_1d.view(1, 1, 1, D), EPSILON)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
avg_time_ref = sum(timings) / len(timings)
eff = efficiency(flops_ref, avg_time_ref)
print(f"PyTorch average execution time: {avg_time_ref:.4f} ms")
print(f"PyTorch performance: {eff:.2f} TFLOPS for {B=} {N=} {D=}.")


# PyTorch (Compiled)
compiled_pytorch_rmsnorm = torch.compile(pytorch_rmsnorm)
print("\nPyTorch (Compiled):")
timings_compiled = []
for _ in range(num_warmup):
    o_compiled = compiled_pytorch_rmsnorm(x, gamma_1d.view(1, 1, 1, D), EPSILON)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    o_compiled = compiled_pytorch_rmsnorm(x, gamma_1d.view(1, 1, 1, D), EPSILON)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_compiled.append(elapsed_time)
avg_time_compiled = sum(timings_compiled) / len(timings_compiled)
eff_compiled = efficiency(flops_ref, avg_time_compiled)
print(f"PyTorch compiled average execution time: {avg_time_compiled:.4f} ms")
print(f"PyTorch compiled performance: {eff_compiled:.2f} TFLOPS for {B=} {N=} {D=}.")
speedup = avg_time_ref / avg_time_compiled
print(f"Speedup from torch.compile: {speedup:.2f}x")


# TK
print("\nTK (PyTorch):")
o_tk = torch.zeros_like(o_ref).bfloat16()
gamma_tk = gamma.detach().clone().to(dtype=torch.bfloat16, device='cuda')
timings = []
for _ in range(num_warmup):
    rms_norm_kernel.dispatch_rmsnorm(x, o_tk, gamma_tk, EPSILON)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    rms_norm_kernel.dispatch_rmsnorm(x, o_tk, gamma_tk, EPSILON)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)

avg_time = sum(timings) / len(timings)
eff = efficiency(flops_ref, avg_time)
print(f"TK average execution time: {avg_time:.4f} ms")
print(f"TK performance: {eff:.2f} TFLOPS for {B=} {N=} {D=}.")
speedup = avg_time_ref / avg_time
print(f"Speedup from TK: {speedup:.2f}x")

# Correctness
print("\nCorrectness:")
o_diff = o_ref - o_tk
max_diff = o_diff.abs().max()
mean_diff = o_diff.abs().mean()
print(f"max_diff: {max_diff}")
print(f"mean_diff: {mean_diff}")
if max_diff > 0.1:
    print(f"o: ", o_ref[0, 0, 0, :8])
    print(f"o_tk: ", o_tk[0, 0, 0, :8])
