"""Narrow runtime compatibility fixes for PyTorch FSDP2 on Gaudi."""

import math
import os

import torch


def patch_hpu_fsdp2_reduce_scatter_copy_in() -> None:
    """Pack FSDP2 gradients without Gaudi's unsupported chunk-cat kernel.

    FSDP2 packs unsharded autograd gradients with ``torch.ops.fsdp.chunk_cat``
    before reduce-scatter. On Gaudi that operator may compile to an FCD-strided
    tensor even when every Python input reports contiguous, and Synapse rejects
    the graph with ``stride on fcd isn't supported``. Use simple contiguous
    copies into the same rank-major padded buffer. This preserves FSDP2's buffer
    layout and collective ordering while avoiding only the unsupported fused op.
    """
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    original = _fsdp_collectives.foreach_reduce_scatter_copy_in
    if getattr(original, "_verl_hpu_contiguous_grads", False):
        return

    def hpu_foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size):
        output = reduce_scatter_input.view(world_size, -1)
        output.zero_()
        column_offset = 0
        repaired = 0
        for grad in unsharded_grads:
            if not grad.is_contiguous():
                repaired += 1
                grad = grad.contiguous()
            rows_per_rank = math.ceil(grad.shape[0] / world_size)
            row_numel = grad.numel() // grad.shape[0]
            padded_chunk_numel = rows_per_rank * row_numel
            for rank in range(world_size):
                row_start = rank * rows_per_rank
                rows = min(rows_per_rank, grad.shape[0] - row_start)
                if rows <= 0:
                    continue
                source = grad.narrow(0, row_start, rows).reshape(-1)
                output[rank, column_offset : column_offset + source.numel()].copy_(source)
            column_offset += padded_chunk_numel

        if column_offset != output.shape[1]:
            raise RuntimeError(
                "HPU FSDP2 gradient pack size mismatch: "
                f"packed={column_offset}, expected={output.shape[1]}"
            )
        if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(
                "VERL HPU FSDP2 GRAD PACK "
                f"rank={rank} fallback=copy tensors={len(unsharded_grads)} "
                f"materialized={repaired} output_numel={reduce_scatter_input.numel()}",
                flush=True,
            )

    hpu_foreach_reduce_scatter_copy_in._verl_hpu_contiguous_grads = True
    _fsdp_collectives.foreach_reduce_scatter_copy_in = hpu_foreach_reduce_scatter_copy_in
