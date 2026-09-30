"""CPU-only D5 admission of one complete, immutable trainer shard.

No model is constructed and no optimizer is mutated here. A trainer rank must
complete this gate before it loads the policy or starts any backward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from self_play_grpo.config import ExperimentConfig
from self_play_grpo.rollouts.pilot import (
    canonical_sha256, file_sha256, read_pilot_manifest,
)
from self_play_grpo.rollouts.schema import MatchRecord
from self_play_grpo.training.coordinator import (
    BatchReceipt, PolicyDescriptor, verify_completed_batch,
)
from self_play_grpo.training.distributed import (
    load_trainer_match_shard, trainer_match_indices,
)


@dataclass(frozen=True)
class TrainerShardAdmission:
    rank: int
    match_indices: tuple[int, ...]
    matches: tuple[MatchRecord, ...]
    receipt: BatchReceipt
    replay_tolerance: float


def admit_trainer_shard(
    root: str | Path,
    *,
    config: ExperimentConfig,
    policy: PolicyDescriptor,
    rank: int,
    expected_manifest_sha256: str,
) -> TrainerShardAdmission:
    """Verify all 64 games, then load only this rank's 16 immutable games."""

    if not 0 <= rank < 4:
        raise ValueError("D5 trainer rank must be in [0, 4)")
    if config.rollout.games_per_update != 64 or config.rollout.parallel_games_per_rank != 2:
        raise ValueError("D5 trainer requires the 64-game, two-active-game profile")
    if (config.training.reward_mode != "outcome" or config.training.group_size != 4
            or config.environment.players != 4
            or config.training.optimizer_epochs_per_batch != 1):
        raise ValueError("D5 trainer requires the four-seat outcome baseline")
    if policy.run_kind != "production":
        raise ValueError("D5 trainer requires a production policy")
    if policy.config_sha256 != canonical_sha256(config.to_dict()):
        raise ValueError("D5 trainer configuration differs from the policy")
    if policy.model_revision != config.model.revision:
        raise ValueError("D5 trainer model revision differs from the policy")
    root_path = Path(root)
    descriptor = PolicyDescriptor(**json.loads(
        (root_path / "policy_descriptor.json").read_text(encoding="utf-8")
    ))
    if descriptor != policy:
        raise ValueError("D5 rollout policy descriptor differs from trainer policy")
    manifest_path = root_path / "manifest.json"
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("D5 rollout manifest differs from coordinator receipt")
    receipt = verify_completed_batch(
        root_path, config=config.to_dict(), policy=policy, expected_games=64,
    )
    if receipt.manifest_sha256 != expected_manifest_sha256:
        raise ValueError("D5 verified batch differs from coordinator receipt")
    manifest = read_pilot_manifest(root_path)
    indices = trainer_match_indices(rank, 4, 64)
    if receipt.match_indices_by_rank[rank] != indices:
        raise ValueError("D5 trainer rank shard differs from coordinator receipt")
    matches = tuple(load_trainer_match_shard(root_path, manifest, indices))
    if len(matches) != 16:
        raise ValueError("D5 trainer shard must contain 16 complete matches")
    if sum(len(match.turns) for match in matches) != receipt.turns_by_rank[rank]:
        raise ValueError("D5 trainer shard turn count differs from receipt")
    if sum(manifest.matches[index].owned_tokens for index in indices) != receipt.owned_tokens_by_rank[rank]:
        raise ValueError("D5 trainer shard owned-token count differs from receipt")
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("D5 rollout manifest changed during trainer admission")
    return TrainerShardAdmission(
        rank=rank, match_indices=indices, matches=matches, receipt=receipt,
        replay_tolerance=manifest.replay_tolerance,
    )
