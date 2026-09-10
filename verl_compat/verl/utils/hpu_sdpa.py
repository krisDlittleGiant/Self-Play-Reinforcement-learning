# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Habana SDPA adapter for padded training and non-reentrant checkpointing."""

import torch


def prepare_mask(attn_mask, query):
    """Return an additive mask and empty-row indicator without mutating the input.

    An empty query gets a temporary key so softmax has a finite denominator. Its
    output is subsequently zeroed. Real queries retain exactly their allowed keys.
    HF uses both Boolean (True = allowed) and additive masks; dtype-min represents
    a blocked position in its eager masks, while other callers use negative infinity.
    """
    if attn_mask is None:
        return None, None
    if attn_mask.dtype == torch.bool:
        allowed = attn_mask
        bias = torch.zeros_like(attn_mask, dtype=query.dtype)
    elif attn_mask.is_floating_point():
        allowed = ~torch.isneginf(attn_mask) & (attn_mask != torch.finfo(attn_mask.dtype).min)
        bias = attn_mask.to(query.dtype)
    else:
        raise TypeError(f"SDPA mask must be Boolean or floating point, got {attn_mask.dtype}")
    empty = ~allowed.any(dim=-1, keepdim=True)
    first_key = torch.arange(attn_mask.shape[-1], device=attn_mask.device) == 0
    bias = bias.masked_fill(~allowed, -float("inf"))
    bias = torch.where(empty & first_key, torch.zeros_like(bias), bias)
    return bias, empty


def fused_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    # Use one explicit mask when both causal and padding constraints are present.
    # The Habana training API does not support every mask + is_causal combination.
    if attn_mask is not None and is_causal:
        causal = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device).tril()
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask & causal
        else:
            attn_mask = attn_mask.masked_fill(~causal, -float("inf"))
        is_causal = False
    mask, empty = prepare_mask(attn_mask, query)
    # Do not catch exceptions here. Non-reentrant checkpointing uses an internal
    # exception to stop recomputation after the saved tensors have been recovered.
    # Swallowing it and calling another backend changes the recomputed graph.
    output = FusedSDPA.apply(
        query.contiguous(), key.contiguous(), value.contiguous(), mask,
        dropout_p, is_causal, scale, "None", True,
    )
    if empty is not None:
        output = output.masked_fill(empty, 0.0)
    return output
