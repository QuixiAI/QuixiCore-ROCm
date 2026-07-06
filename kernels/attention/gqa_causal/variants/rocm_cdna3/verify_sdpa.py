import torch, tk_kernel
from torch.nn.functional import scaled_dot_product_attention as sdpa
B,N,H,HKV,D = 16,2048,64,8,128   # must match compiled ATTN_* macros
dtype=torch.bfloat16
torch.manual_seed(0)
q=torch.randn(B,N,H,D,dtype=dtype,device='cuda')
k=torch.randn(B,N,HKV,D,dtype=dtype,device='cuda')
v=torch.randn(B,N,HKV,D,dtype=dtype,device='cuda')
out=torch.zeros(B,N,H,D,dtype=dtype,device='cuda')
lse=torch.zeros(B,H,1,N,dtype=torch.float32,device='cuda')
tk_kernel.dispatch_micro(q,k,v,out,lse)
torch.cuda.synchronize()
qs=q.transpose(1,2); ks=k.transpose(1,2).repeat_interleave(H//HKV,1); vs=v.transpose(1,2).repeat_interleave(H//HKV,1)
ref=sdpa(qs,ks,vs,is_causal=True).transpose(1,2)
d=(out.float()-ref.float()).abs()
rel=(d.sum()/ref.float().abs().sum()).item(); mx=d.max().item()
print(f"gqa fwd vs SDPA: mean rel {100*rel:.4f}%  max abs {mx:.4g}  ({'PASS' if rel<0.02 else 'FAIL'})")
