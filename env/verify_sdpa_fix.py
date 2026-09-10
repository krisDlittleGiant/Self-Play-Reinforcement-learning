"""Reproduce padded native-SDPA NaNs and assert the fixed HPU adapter's semantics."""
import sys
import types

import torch
import habana_frameworks.torch  # noqa: F401
import verl
from torch.utils.checkpoint import checkpoint
from habana_frameworks.torch.hpex.kernels import FusedSDPA

torch.manual_seed(42)
device = 'hpu'
mask = torch.tensor([[False, False, False, False],
                     [False, False, False, False],
                     [False, False, True, False],
                     [False, False, True, True]], device=device)[None, None]

def inputs():
    q = torch.zeros(1, 1, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    v = torch.tensor([10., 20., 30., 40.], dtype=torch.bfloat16, device=device)
    v = v[None, None, :, None].expand_as(q).clone().requires_grad_()
    return q, k, v

q, k, v = inputs()
old = verl._verl_orig_sdpa(q, k, v, attn_mask=mask)
old[:, :, 2:].sum().backward()
print('NATIVE: outputs=', old.detach().cpu()[0, 0, :, 0].tolist(),
      'finite_gradients=', all(bool(torch.isfinite(x.grad).all()) for x in (q,k,v)), flush=True)
q, k, v = inputs()
raw_mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, -float('inf'))
raw = FusedSDPA.apply(q, k, v, raw_mask, 0.0, False, None, 'None', True)
raw[:, :, 2:].sum().backward()
print('UNREPAIRED_ADDITIVE: outputs=', raw.detach().cpu()[0,0,:,0].tolist(),
      'finite_gradients=', all(bool(torch.isfinite(x.grad).all()) for x in (q,k,v)), flush=True)

# Regression: importing a scheduler module must not switch the training backend.
module_name = 'sglang.srt.managers.scheduler'
previous_module = sys.modules.get(module_name)
sys.modules[module_name] = types.ModuleType(module_name)
original_native = verl._verl_orig_sdpa
def forbidden_native(*args, **kwargs):
    raise AssertionError('Training incorrectly bypassed FusedSDPA because scheduler was imported')
verl._verl_orig_sdpa = forbidden_native
try:
    for mask_kind in ('bool', 'negative_infinity', 'dtype_min'):
        test_mask = mask.clone()
        if mask_kind != 'bool':
            blocked = -float('inf') if mask_kind == 'negative_infinity' else torch.finfo(torch.bfloat16).min
            test_mask = torch.zeros_like(mask, dtype=torch.bfloat16).masked_fill(~mask, blocked)
        before = test_mask.clone()
        q, k, v = inputs()
        def fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=test_mask)
        fixed = checkpoint(fn, q, k, v, use_reentrant=False)
        fixed[:, :, 2:].sum().backward()
        torch.testing.assert_close(fixed[0,0,:,0], torch.tensor([0., 0., 30., 35.], device=device, dtype=q.dtype))
        torch.testing.assert_close(v.grad[0,0,:,0], torch.tensor([0., 0., 1.5, .5], device=device, dtype=q.dtype))
        torch.testing.assert_close(test_mask, before)
        assert all(bool(torch.isfinite(x.grad).all()) for x in (q,k,v))
        print('FIXED:', mask_kind, 'outputs=', fixed.detach().cpu()[0,0,:,0].tolist(),
              'dV=', v.grad.cpu()[0,0,:,0].tolist(), 'finite_gradients=True checkpoint=True mask_unchanged=True', flush=True)
finally:
    verl._verl_orig_sdpa = original_native
    if previous_module is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous_module
print('PASS: padded attention example, gradients, checkpointing and scheduler-import regression', flush=True)
