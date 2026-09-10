"""Exercise real Qwen weights with padded inputs and an unmasked, nonzero training loss."""
import argparse
import os
import json
import time

import torch
import habana_frameworks.torch  # noqa: F401
import verl  # noqa: F401 -- install the same SDPA adapter as the actor
from transformers import Qwen3ForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
p.add_argument('--length', type=int, default=256)
p.add_argument('--model', default=os.environ.get('VERIFY_SDPA_MODEL', 'Qwen/Qwen3-0.6B'))
p.add_argument('--legacy', action='store_true', help='Reproduce the previous adapter for a before/after comparison')
p.add_argument('--native', action='store_true', help='Test the native SDPA path selected by the old scheduler-import gate')
args = p.parse_args()
started = time.monotonic()
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.legacy:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    def legacy_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        try:
            if attn_mask is not None:
                if 0 in attn_mask.stride():
                    attn_mask = attn_mask.contiguous()
                if attn_mask.dtype == torch.bool:
                    attn_mask[..., 0] |= ~attn_mask.any(dim=-1)
                else:
                    empty = ~(attn_mask > torch.finfo(attn_mask.dtype).min / 2).any(dim=-1)
                    attn_mask[..., 0] = torch.where(empty, 0.0, attn_mask[..., 0])
            return FusedSDPA.apply(q, k, v, attn_mask, dropout_p, is_causal, scale)
        except Exception:
            return verl._verl_orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                        is_causal=is_causal, scale=scale, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = legacy_sdpa
if args.native:
    torch.nn.functional.scaled_dot_product_attention = verl._verl_orig_sdpa
model = Qwen3ForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attention,
).to('hpu').train()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
text = 'Question: There are 3 apples and I buy 2 more. How many apples are there? Answer: 3 + 2 = 5. #### 5\n'
tokens = tokenizer.encode(text, add_special_tokens=False)
ids = torch.tensor((tokens * (args.length // len(tokens) + 1))[:args.length])[None].repeat(2, 1)
mask = torch.ones_like(ids)
pad = min(434, args.length // 4)
mask[:, :pad] = 0
mask[1, -16:] = 0
ids[mask == 0] = tokenizer.pad_token_id
labels = ids.clone()
labels[mask == 0] = -100
labels[:, pad] = -100  # Do not train a padding query to predict the first real token.
positions = (mask.cumsum(-1) - 1).clamp_min(0)
seen = set()
@torch.no_grad()
def check_attention(module, inputs, output):
    if module.layer_idx not in seen:
        seen.add(module.layer_idx)
        y = output[0]
        print('layer', module.layer_idx, 'finite', bool(torch.isfinite(y).all()),
              'maxabs', float(y.float().abs().max()), flush=True)
for layer in model.model.layers:
    layer.self_attn.register_forward_hook(check_attention)
with torch.autocast('hpu', dtype=torch.bfloat16):
    result = model(input_ids=ids.to('hpu'), attention_mask=mask.to('hpu'),
                   position_ids=positions.to('hpu'), labels=labels.to('hpu'), use_cache=False)
loss = result.loss
print('loss=', float(loss), flush=True)
loss.backward()
bad = [name for name, x in model.named_parameters() if x.grad is not None and not bool(torch.isfinite(x.grad).all())]
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(json.dumps({'attention': args.attention, 'length': args.length, 'loss': float(loss),
                  'grad_norm': float(norm), 'nonfinite_grad_parameters': bad,
                  'elapsed_s': time.monotonic() - started}), flush=True)
assert torch.isfinite(loss) and torch.isfinite(norm) and norm > 0 and not bad
