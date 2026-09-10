"""Compare the installed Habana kernel on bool and additive masks, including backward."""
import argparse

import torch
import habana_frameworks.torch  # noqa: F401
from habana_frameworks.torch.hpex.kernels import FusedSDPA

p = argparse.ArgumentParser()
p.add_argument('--length', type=int, default=32)
p.add_argument('--amplitude', type=float, default=1.0)
args = p.parse_args()
torch.manual_seed(42)
B, H, L, D = 2, 16, args.length, 128
pad = min(434, L // 4)
allowed = torch.ones(L, L, dtype=torch.bool).tril()
allowed[:, :pad] = False
allowed = allowed[None, None].expand(B, 1, L, L).clone()
repaired = allowed.clone()
repaired[..., 0] |= ~repaired.any(dim=-1)
inputs = [torch.randn(B, H, L, D, dtype=torch.bfloat16) for _ in range(3)]
inputs[0] *= args.amplitude
inputs[1] *= args.amplitude
upstream = torch.randn_like(inputs[0]) / 16
upstream[:, :, :pad] = 0
print(f'shape={inputs[0].shape} padding={pad}', flush=True)
reference_inputs = [x.to('hpu').requires_grad_() for x in inputs]
q, k, v = reference_inputs
scores = torch.matmul(q, k.transpose(-1, -2)) * D**-0.5
scores = scores.masked_fill(~repaired.to('hpu'), -float('inf'))
reference = torch.matmul(torch.softmax(scores.float(), -1).to(q.dtype), v)
reference.backward(upstream.to('hpu'))
reference_out = reference.detach().cpu()
reference_grads = [x.grad.cpu() for x in reference_inputs]
del reference, scores, reference_inputs, q, k, v
for name, mask in [('bool_original', allowed), ('bool_repaired', repaired),
                   ('additive_original', torch.zeros_like(allowed, dtype=torch.bfloat16).masked_fill(~allowed, -float('inf'))),
                   ('additive_repaired', torch.zeros_like(repaired, dtype=torch.bfloat16).masked_fill(~repaired, -float('inf')))]:
    for recompute in (False, True):
        q, k, v = [x.to('hpu').requires_grad_() for x in inputs]
        try:
            o = FusedSDPA.apply(q, k, v, mask.to('hpu'), 0.0, False, D**-0.5, 'None', recompute)
            o.backward(upstream.to('hpu'))
            print(name, 'recompute=', recompute, 'finite=',
                  [bool(torch.isfinite(x).all().item()) for x in (o, q.grad, k.grad, v.grad)],
                  'norm=', q.grad.float().norm().item(),
                  'out_maxerr=', (o.detach().cpu()[:, :, pad:].float() - reference_out[:, :, pad:].float()).abs().max().item(),
                  'grad_maxerr=', [float((x.grad.cpu().float() - r.float()).abs().max()) for x, r in zip((q,k,v), reference_grads)],
                  flush=True)
        except Exception as exc:
            print(name, 'recompute=', recompute, type(exc).__name__, str(exc), flush=True)
