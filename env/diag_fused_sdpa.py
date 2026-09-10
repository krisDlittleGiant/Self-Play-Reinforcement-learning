"""Isolate the FusedSDPA NaN. Seconds, one card, no Ray/verl/sglang.

Reproduces the exact shape and mask verl feeds the actor: Qwen3-0.6B geometry
(16 heads x 128 dim), bf16, causal + LEFT-padded prompt, then checks the BACKWARD
pass -- which is where the NaN actually lands (actor/grad_norm), not the forward.
"""
import os, torch
import habana_frameworks.torch as htorch  # noqa: F401
from habana_frameworks.torch.hpex.kernels import FusedSDPA

B, H, L, D = 2, 16, 2560, 128
PROMPT_PAD = 434          # 512 - ~78 mean prompt: verl left-pads, torch_functional.py:512
dev, dt = "hpu", torch.bfloat16
neg = torch.finfo(dt).min
torch.manual_seed(0)


def make_mask(repair_empty_rows):
    m = torch.full((L, L), neg, dtype=dt).triu(1)          # causal
    m[:, :PROMPT_PAD] = neg                                # left-pad keys masked
    m = m[None, None].expand(B, 1, L, L).contiguous()
    if repair_empty_rows:                                   # the F39 fix
        keep = m > (neg / 2)
        empty = ~keep.any(dim=-1)
        m[..., 0] = torch.where(empty, torch.zeros_like(m[..., 0]), m[..., 0])
    return m.to(dev)


def trial(name, repair, **kw):
    q, k, v = (torch.randn(B, H, L, D, dtype=dt, device=dev, requires_grad=True) for _ in range(3))
    m = make_mask(repair)
    try:
        out = FusedSDPA.apply(q, k, v, m, 0.0, False, D ** -0.5, *(
            kw.get("softmax_mode", "None"), kw.get("recompute_mode", None)))
        f_nan = bool(torch.isnan(out).any().item())
        out.sum().backward()
        g_nan = any(bool(torch.isnan(t.grad).any().item()) for t in (q, k, v))
        gn = torch.cat([t.grad.flatten() for t in (q, k, v)]).float().norm().item()
        print(f"{name:<46} fwd_nan={f_nan!s:<5} bwd_nan={g_nan!s:<5} grad_norm={gn:.4f}")
    except Exception as e:
        print(f"{name:<46} RAISED {type(e).__name__}: {str(e)[:70]}")


print(f"shape B={B} H={H} L={L} D={D} dtype={dt} left_pad={PROMPT_PAD} "
      f"LAZY_MODE={os.environ.get('PT_HPU_LAZY_MODE')}\n")
for repair in (False, True):
    tag = "repair=ON " if repair else "repair=OFF"
    trial(f"{tag} softmax=None  recompute=None ", repair)
    trial(f"{tag} softmax=fp32  recompute=None ", repair, softmax_mode="fp32")
    trial(f"{tag} softmax=None  recompute=False", repair, recompute_mode=False)
    trial(f"{tag} softmax=fp32  recompute=False", repair, softmax_mode="fp32", recompute_mode=False)
    print()

# eager reference: the configuration that currently trains correctly
q, k, v = (torch.randn(B, H, L, D, dtype=dt, device=dev, requires_grad=True) for _ in range(3))
o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=make_mask(False))
o.sum().backward()
print(f"{'EAGER reference':<46} bwd_nan="
      f"{any(bool(torch.isnan(t.grad).any().item()) for t in (q,k,v))!s:<5} "
      f"grad_norm={torch.cat([t.grad.flatten() for t in (q,k,v)]).float().norm().item():.4f}")
