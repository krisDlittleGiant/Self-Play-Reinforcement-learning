"""Fail-closed single-node HPU distributed runtime contracts.

This module deliberately avoids importing Torch or Habana at import time so
that environment discovery remains covered by accelerator-free unit tests.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Mapping


TORCHRUN_VARIABLES = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")


def synthetic_gradient_values(rank: int, parameter_count: int) -> tuple[float, ...]:
    """Return a deterministic, rank-distinct synthetic adapter gradient."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    multiplier = float(rank + 1)
    return tuple(
        multiplier * float(index + 1) * (1.0 if index % 2 == 0 else -1.0)
        for index in range(parameter_count)
    )


def expected_synthetic_average_gradient(
    world_size: int,
    parameter_count: int,
) -> tuple[float, ...]:
    """Return the exact mean of ``synthetic_gradient_values`` over all ranks."""

    if world_size < 2:
        raise ValueError("world_size must be at least 2")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    mean_multiplier = float(world_size + 1) / 2.0
    return tuple(
        mean_multiplier * float(index + 1) * (1.0 if index % 2 == 0 else -1.0)
        for index in range(parameter_count)
    )


@dataclass(frozen=True)
class DistributedRuntime:
    """The single-node rank assignment supplied by ``torchrun``."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    backend: str = "hccl"

    @property
    def device(self) -> str:
        # The pinned Habana eager bridge rejects CUDA-style indexed strings
        # such as ``hpu:1``. initialize_distributed_hpu(local_rank=...) binds
        # the process; tensors must then target the unindexed ``hpu`` device.
        return "hpu"

    @property
    def logical_device(self) -> str:
        return f"hpu:{self.local_rank}"

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self) | {
            "device": self.device,
            "logical_device": self.logical_device,
        }


def discover_torchrun_runtime(
    environ: Mapping[str, str],
    *,
    expected_world_size: int,
) -> DistributedRuntime:
    """Parse and validate the single-node ``torchrun`` environment.

    Distributed validation must never silently fall back to rank zero or a
    one-process world.  All rank variables are therefore mandatory.
    """

    if expected_world_size < 2:
        raise ValueError("expected_world_size must be at least 2")
    missing = [name for name in TORCHRUN_VARIABLES if name not in environ]
    if missing:
        raise RuntimeError(
            "Distributed validation must be launched by torchrun; missing "
            + ", ".join(missing)
        )
    values: dict[str, int] = {}
    for name in TORCHRUN_VARIABLES:
        raw = environ[name]
        try:
            values[name] = int(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc

    rank = values["RANK"]
    local_rank = values["LOCAL_RANK"]
    world_size = values["WORLD_SIZE"]
    local_world_size = values["LOCAL_WORLD_SIZE"]
    if world_size != expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} differs from expected {expected_world_size}"
        )
    if not 0 <= rank < world_size:
        raise RuntimeError(f"RANK={rank} is outside [0, {world_size})")
    # The first implementation intentionally supports one node only.  Requiring
    # the local ranks to span the whole world catches accidental multi-node or
    # duplicate-device launches before model loading.
    if not 0 <= local_rank < world_size:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside [0, {world_size}) for a "
            "single-node launch"
        )
    if local_world_size != world_size:
        raise RuntimeError(
            f"LOCAL_WORLD_SIZE={local_world_size} differs from WORLD_SIZE={world_size}; "
            "only one-node launches are supported"
        )
    return DistributedRuntime(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
    )


def initialize_hccl_process_group(
    runtime: DistributedRuntime,
    *,
    timeout_seconds: int = 120,
) -> tuple[Any, Any]:
    """Initialize one HCCL process group and select this rank's HPU."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    try:
        import torch
        import torch.distributed as dist
        import habana_frameworks.torch.distributed.hccl as hccl
    except ImportError as exc:
        raise RuntimeError(
            "The Gaudi Torch and Habana distributed runtime are required"
        ) from exc

    if dist.is_initialized():
        raise RuntimeError("A Torch process group is already initialized")
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        raise RuntimeError("No HPU is available to this rank")

    initialized_world_size, initialized_rank, initialized_local_rank = (
        hccl.initialize_distributed_hpu(
            world_size=runtime.world_size,
            rank=runtime.rank,
            local_rank=runtime.local_rank,
        )
    )
    if (
        initialized_world_size,
        initialized_rank,
        initialized_local_rank,
    ) != (runtime.world_size, runtime.rank, runtime.local_rank):
        raise RuntimeError("Habana distributed initialization changed rank metadata")
    visible_modules = os.environ.get("HABANA_VISIBLE_MODULES")
    expected_module_id = (
        visible_modules.split(",")[runtime.local_rank]
        if visible_modules is not None
        else str(runtime.local_rank)
    )
    if os.environ.get("HLS_MODULE_ID") != expected_module_id:
        raise RuntimeError(
            "Habana module binding differs from LOCAL_RANK: "
            f"HLS_MODULE_ID={os.environ.get('HLS_MODULE_ID')!r}, "
            f"expected={expected_module_id!r}"
        )
    dist.init_process_group(
        backend=runtime.backend,
        rank=runtime.rank,
        world_size=runtime.world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return torch, dist


def validate_initialized_hccl(
    runtime: DistributedRuntime,
    torch: Any,
    dist: Any,
) -> dict[str, object]:
    """Prove rank/device membership and reductions on an initialized group."""

    if not dist.is_initialized():
        raise RuntimeError("Torch process group is not initialized")
    if dist.get_world_size() != runtime.world_size:
        raise RuntimeError("Initialized process-group world size changed")
    if dist.get_rank() != runtime.rank:
        raise RuntimeError("Initialized process-group rank changed")
    module_id = os.environ.get("HLS_MODULE_ID")
    visible_modules = os.environ.get("HABANA_VISIBLE_MODULES")
    expected_module_ids = (
        visible_modules.split(",")[: runtime.world_size]
        if visible_modules is not None
        else [str(index) for index in range(runtime.world_size)]
    )
    module_binding_ok = module_id == expected_module_ids[runtime.local_rank]
    device = torch.device(runtime.device)
    rank_membership = torch.zeros(
        runtime.world_size, dtype=torch.float32, device=device
    )
    local_membership = torch.zeros_like(rank_membership)
    rank_membership[runtime.rank] = 1.0
    local_membership[runtime.local_rank] = 1.0
    probe = torch.tensor(
        [float(runtime.rank + 1), float(runtime.local_rank + 1)],
        dtype=torch.float32,
        device=device,
    )
    dist.all_reduce(rank_membership, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_membership, op=dist.ReduceOp.SUM)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM)

    expected_sum = runtime.world_size * (runtime.world_size + 1) / 2
    rank_values = rank_membership.cpu().tolist()
    device_values = local_membership.cpu().tolist()
    reduction_values = probe.cpu().tolist()
    expected_membership = [1.0] * runtime.world_size
    rank_mapping_ok = rank_values == expected_membership
    device_mapping_ok = device_values == expected_membership
    reduction_ok = reduction_values == [expected_sum, expected_sum]

    # Ensure every process reaches all checks before any process reports a
    # failure and tears down the communicator.
    local_ok = torch.tensor(
        [
            float(
                rank_mapping_ok
                and device_mapping_ok
                and reduction_ok
                and module_binding_ok
            )
        ],
        dtype=torch.float32,
        device=device,
    )
    dist.all_reduce(local_ok, op=dist.ReduceOp.SUM)
    dist.barrier()
    if float(local_ok.item()) != float(runtime.world_size):
        raise RuntimeError(
            "Distributed contract failed: "
            f"rank_mapping={rank_mapping_ok}, "
            f"device_mapping={device_mapping_ok}, "
            f"reduction={reduction_ok}, module_binding={module_binding_ok}"
        )

    return {
        "backend": str(dist.get_backend()),
        "device_binding": "initialize_distributed_hpu -> HLS_MODULE_ID",
        "devices": [f"hpu:{index}" for index in range(runtime.world_size)],
        "expected_reduction_sum": expected_sum,
        "module_ids": expected_module_ids,
        "rank_membership": rank_values,
        "device_membership": device_values,
        "reduction_values": reduction_values,
        "status": "ok",
        "tensor_device": runtime.device,
        "world_size": runtime.world_size,
    }


def broadcast_trainable_parameters(
    model: Any,
    dist: Any,
    *,
    source_rank: int = 0,
) -> dict[str, int]:
    """Make the live trainable adapter byte-identical to the source rank."""

    tensors = 0
    parameters = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        dist.broadcast(parameter.data, src=source_rank)
        tensors += 1
        parameters += int(parameter.numel())
    if tensors == 0 or parameters == 0:
        raise RuntimeError("Model has no trainable parameters to broadcast")
    dist.barrier()
    return {
        "trainable_parameter_tensors": tensors,
        "trainable_parameters": parameters,
    }


def validate_hccl_collectives(
    runtime: DistributedRuntime,
    *,
    timeout_seconds: int = 120,
) -> dict[str, object]:
    """Initialize HCCL, validate collectives, and cleanly tear it down.

    No model is loaded and no parameter is updated.  The caller is responsible
    for launching exactly one process per HPU with ``torchrun``.
    """

    torch, dist = initialize_hccl_process_group(
        runtime,
        timeout_seconds=timeout_seconds,
    )
    try:
        return validate_initialized_hccl(runtime, torch, dist)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def validate_distributed_optimizer_math(
    runtime: DistributedRuntime,
    *,
    timeout_seconds: int = 120,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    parameter_count: int = 16,
) -> dict[str, object]:
    """Average one synthetic adapter gradient and take one identical AdamW step.

    This is the trainer collective-math gate, not model training.  It loads no
    checkpoint and uses one small synthetic parameter vector.  Each rank starts
    with a distinct deterministic gradient, synchronizes gradients once after
    the local objective boundary, clips the shared gradient, and proves exact
    parameter and optimizer-state equality after one step.
    """

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")

    torch, dist = initialize_hccl_process_group(
        runtime,
        timeout_seconds=timeout_seconds,
    )
    try:
        runtime_contract = validate_initialized_hccl(runtime, torch, dist)
        device = torch.device(runtime.device)
        initial_values = torch.linspace(
            -0.5,
            0.5,
            parameter_count,
            dtype=torch.float32,
            device=device,
        )
        parameter = torch.nn.Parameter(initial_values.clone())
        dist.broadcast(parameter.data, src=0)
        initial_parameter = parameter.detach().clone()
        optimizer = torch.optim.AdamW(
            [parameter],
            lr=learning_rate,
            weight_decay=0.0,
        )

        parameter.grad = torch.tensor(
            synthetic_gradient_values(runtime.rank, parameter_count),
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(runtime.world_size))
        expected_gradient = torch.tensor(
            expected_synthetic_average_gradient(
                runtime.world_size,
                parameter_count,
            ),
            dtype=torch.float32,
            device=device,
        )
        average_error = torch.max(torch.abs(parameter.grad - expected_gradient))
        dist.all_reduce(average_error, op=dist.ReduceOp.MAX)

        pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter],
            max_grad_norm,
        )
        optimizer.step()

        def global_source_difference(tensor: Any) -> float:
            reference = tensor.detach().clone()
            dist.broadcast(reference, src=0)
            difference = torch.max(torch.abs(tensor.detach() - reference))
            dist.all_reduce(difference, op=dist.ReduceOp.MAX)
            return float(difference.cpu().item())

        parameter_difference = global_source_difference(parameter)
        optimizer_state = optimizer.state[parameter]
        first_moment_difference = global_source_difference(
            optimizer_state["exp_avg"]
        )
        second_moment_difference = global_source_difference(
            optimizer_state["exp_avg_sq"]
        )
        step_value = torch.tensor(
            [float(optimizer_state["step"].detach().cpu().item())],
            dtype=torch.float32,
            device=device,
        )
        step_difference = global_source_difference(step_value)
        parameter_change = float(
            torch.max(torch.abs(parameter.detach() - initial_parameter)).cpu().item()
        )
        norm_value = torch.tensor(
            [float(pre_clip_grad_norm.detach().cpu().item())],
            dtype=torch.float32,
            device=device,
        )
        grad_norm_difference = global_source_difference(norm_value)
        optimizer.zero_grad(set_to_none=True)
        gradients_cleared = parameter.grad is None

        average_error_value = float(average_error.cpu().item())
        local_ok = torch.tensor(
            [
                float(
                    average_error_value == 0.0
                    and parameter_difference == 0.0
                    and first_moment_difference == 0.0
                    and second_moment_difference == 0.0
                    and step_difference == 0.0
                    and grad_norm_difference == 0.0
                    and parameter_change > 0.0
                    and gradients_cleared
                )
            ],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(local_ok, op=dist.ReduceOp.SUM)
        dist.barrier()
        if float(local_ok.cpu().item()) != float(runtime.world_size):
            raise RuntimeError(
                "Distributed optimizer-math contract failed: "
                f"average_error={average_error_value}, "
                f"parameter_difference={parameter_difference}, "
                f"first_moment_difference={first_moment_difference}, "
                f"second_moment_difference={second_moment_difference}, "
                f"step_difference={step_difference}, "
                f"grad_norm_difference={grad_norm_difference}, "
                f"parameter_change={parameter_change}, "
                f"gradients_cleared={gradients_cleared}"
            )

        return {
            "global_max_abs_average_gradient_error": average_error_value,
            "global_max_abs_first_moment_difference": first_moment_difference,
            "global_max_abs_grad_norm_difference": grad_norm_difference,
            "global_max_abs_parameter_difference": parameter_difference,
            "global_max_abs_second_moment_difference": second_moment_difference,
            "global_max_abs_step_difference": step_difference,
            "gradient_sync_phases": 1,
            "gradient_tensors_reduced": 1,
            "gradients_cleared": gradients_cleared,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "optimizer": "AdamW",
            "optimizer_state_entries": len(optimizer.state),
            "optimizer_steps": 1,
            "parameter_change": parameter_change,
            "parameter_count": parameter_count,
            "pre_clip_grad_norm": float(pre_clip_grad_norm.detach().cpu().item()),
            "runtime_contract": runtime_contract,
            "status": "ok",
            "world_size": runtime.world_size,
        }
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
