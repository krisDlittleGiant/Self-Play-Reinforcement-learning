#!/usr/bin/env python3
"""Regression check for Gaudi-safe FSDP2 gradient packing on one HPU."""

import os
import importlib.util
from pathlib import Path

os.environ.setdefault("VERL_PLATFORM", "hpu")
os.environ.setdefault("VERL_HPU_WEIGHT_SYNC_DEBUG", "1")

import habana_frameworks.torch as htorch
import torch

utility_path = Path(__file__).parents[1] / "verl_compat/verl/utils/hpu_fsdp2.py"
spec = importlib.util.spec_from_file_location("verl_hpu_fsdp2_test", utility_path)
assert spec is not None and spec.loader is not None
utility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utility)
patch_hpu_fsdp2_reduce_scatter_copy_in = utility.patch_hpu_fsdp2_reduce_scatter_copy_in


def main() -> None:
    patch_hpu_fsdp2_reduce_scatter_copy_in()
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    # Transpose creates last-dimension stride 6. The fallback must pack it into
    # FSDP's rank-major reduce-scatter layout without invoking aten::_chunk_cat.
    grad = torch.arange(24, dtype=torch.float32, device="hpu").reshape(4, 6).transpose(0, 1)
    assert not grad.is_contiguous()
    assert grad.stride(-1) != 1

    world_size = 2
    copy_in = torch.empty(grad.numel(), dtype=grad.dtype, device="hpu")
    grads = [grad]
    _fsdp_collectives.foreach_reduce_scatter_copy_in(grads, copy_in, world_size)
    htorch.core.mark_step()
    torch.hpu.synchronize()

    expected = torch.stack([chunk.reshape(-1) for chunk in torch.chunk(grad, world_size, dim=0)])
    torch.testing.assert_close(copy_in.view(world_size, -1).cpu(), expected.cpu())
    print("PASS HPU FSDP2 gradient copy fallback matches the expected rank-major layout", flush=True)


if __name__ == "__main__":
    main()
