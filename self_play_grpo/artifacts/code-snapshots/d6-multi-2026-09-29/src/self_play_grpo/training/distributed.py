"""Contracts shared by the replicated four-rank trainer validation path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_play_grpo.rollouts.pilot import (
    PilotManifest,
    canonical_sha256,
    read_and_replay_pilot_match,
)
from self_play_grpo.rollouts.schema import MatchRecord, ensure_single_policy_version


DISTRIBUTED_UPDATE_SCHEMA_VERSION = 1


def trainer_match_indices(
    rank: int,
    world_size: int,
    total_matches: int,
) -> tuple[int, ...]:
    """Return an equal contiguous shard without splitting a match group."""

    if world_size < 2:
        raise ValueError("Distributed training requires at least two ranks")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside [0, {world_size})")
    if total_matches <= 0:
        raise ValueError("total_matches must be positive")
    if total_matches % world_size:
        raise ValueError(
            f"{total_matches} matches cannot be divided equally across "
            f"{world_size} trainer ranks"
        )
    per_rank = total_matches // world_size
    start = rank * per_rank
    return tuple(range(start, start + per_rank))


def validate_distributed_training_manifest(
    manifest: PilotManifest,
    *,
    config: Mapping[str, Any],
    world_size: int,
) -> int:
    """Validate one complete, single-policy batch for equal trainer sharding."""

    manifest.validate()
    if canonical_sha256(config) != manifest.config_sha256:
        raise ValueError("Training configuration differs from rollout manifest")
    if len(manifest.matches) != manifest.target_games:
        raise ValueError("Distributed training requires a complete rollout manifest")
    rollout = config.get("rollout")
    if not isinstance(rollout, Mapping):
        raise ValueError("Training configuration is missing rollout settings")
    configured_games = int(rollout.get("games_per_update", 0))
    if configured_games != len(manifest.matches):
        raise ValueError(
            f"Manifest has {len(manifest.matches)} matches, but games_per_update "
            f"is {configured_games}"
        )
    indices = trainer_match_indices(0, world_size, len(manifest.matches))
    policy_versions = {entry.policy_version for entry in manifest.matches}
    if policy_versions != {manifest.policy_version}:
        raise ValueError("Distributed training manifest mixes policy versions")
    return len(indices)


def load_trainer_match_shard(
    root: str | Path,
    manifest: PilotManifest,
    indices: Sequence[int],
) -> list[MatchRecord]:
    """Load, hash-check, and engine-replay one rank's complete matches."""

    matches: list[MatchRecord] = []
    for index in indices:
        if not 0 <= index < len(manifest.matches):
            raise ValueError(f"Trainer match index {index} is outside the manifest")
        entry = manifest.matches[index]
        if entry.index != index:
            raise ValueError("Pilot manifest entries are not index aligned")
        match, digest = read_and_replay_pilot_match(root, manifest, index)
        if digest != entry.sha256:
            raise ValueError(f"Pilot artifact digest changed: {entry.path}")
        matches.append(match)
    if not matches:
        raise ValueError("Trainer shard contains no matches")
    if ensure_single_policy_version(matches) != manifest.policy_version:
        raise ValueError("Trainer shard uses a different behavior policy")
    return matches


def average_trainable_gradients(
    model: Any,
    dist: Any,
    *,
    world_size: int,
) -> dict[str, int]:
    """Synchronize every trainable gradient at one post-backward boundary."""

    tensors = 0
    parameters = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError("A trainable parameter is missing its local gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))
        tensors += 1
        parameters += int(parameter.numel())
    if tensors == 0:
        raise RuntimeError("The model has no synchronized trainable gradients")
    return {
        "gradient_sync_phases": 1,
        "gradient_tensors_reduced": tensors,
        "trainable_parameters": parameters,
    }


def global_source_max_abs_difference(
    torch: Any,
    dist: Any,
    tensor: Any,
) -> float:
    """Compare every rank with rank zero and return the global maximum error."""

    if tensor.numel() <= 0:
        raise ValueError("Cannot compare an empty tensor")
    reference = tensor.detach().clone()
    dist.broadcast(reference, src=0)
    difference = torch.max(torch.abs(tensor.detach() - reference))
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    return float(difference.cpu().item())


def verify_optimizer_replicas(
    torch: Any,
    dist: Any,
    parameters: Sequence[Any],
    optimizer: Any,
    *,
    device: Any,
) -> dict[str, float | int]:
    """Require exact parameters and populated AdamW state on every rank."""

    if not parameters:
        raise ValueError("No trainable parameters were supplied")
    parameter_vector = torch.cat(
        [parameter.detach().reshape(-1) for parameter in parameters]
    )
    first_moments = []
    second_moments = []
    steps = []
    state_tensors = 0
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state:
            raise RuntimeError("AdamW state is missing for a trainable parameter")
        first_moments.append(state["exp_avg"].detach().reshape(-1))
        second_moments.append(state["exp_avg_sq"].detach().reshape(-1))
        steps.append(float(state["step"].detach().cpu().item()))
        state_tensors += 3
    first_moment_vector = torch.cat(first_moments)
    second_moment_vector = torch.cat(second_moments)
    step_vector = torch.tensor(steps, dtype=torch.float32, device=device)
    return {
        "global_max_abs_first_moment_difference": global_source_max_abs_difference(
            torch, dist, first_moment_vector
        ),
        "global_max_abs_parameter_difference": global_source_max_abs_difference(
            torch, dist, parameter_vector
        ),
        "global_max_abs_second_moment_difference": global_source_max_abs_difference(
            torch, dist, second_moment_vector
        ),
        "global_max_abs_step_difference": global_source_max_abs_difference(
            torch, dist, step_vector
        ),
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_state_tensors": state_tensors,
    }


def write_rank_update_report(
    root: str | Path,
    rank: int,
    report: Mapping[str, Any],
) -> Path:
    """Atomically persist one trainer-rank validation report."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    root_path = Path(root)
    target = root_path / "ranks" / f"rank-{rank:03d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": DISTRIBUTED_UPDATE_SCHEMA_VERSION,
        **dict(report),
    }
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target

