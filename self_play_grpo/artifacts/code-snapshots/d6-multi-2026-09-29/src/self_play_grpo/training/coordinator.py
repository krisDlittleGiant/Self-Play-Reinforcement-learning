"""Fail-closed, accelerator-free D5 cycle contracts.

This module coordinates *evidence*, not processes.  A later launcher supplies
four rollout workers and a separate four-rank trainer group; no eight-rank
gradient collective is created here.  The complete rollout is checked before
any trainer acknowledgement can advance the cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from self_play_grpo.rewards.outcome import outcome_advantages
from self_play_grpo.rollouts.pilot import (
    PilotManifest,
    canonical_sha256,
    directory_sha256,
    file_sha256,
    make_match_entry,
    read_and_replay_pilot_match,
    read_pilot_manifest,
)
from self_play_grpo.training.distributed import (
    trainer_match_indices,
    validate_distributed_training_manifest,
)


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


class Phase(str, Enum):
    READY = "ready"
    COLLECTING = "collecting"
    BATCH_COMMITTED = "batch_committed"
    VERIFIED = "verified"
    UPDATED = "updated"
    REFRESHING = "refreshing"
    FAILED = "failed"


class CycleError(RuntimeError):
    """A cycle cannot safely advance after this protocol failure."""


@dataclass(frozen=True)
class RoleLayout:
    rollout_modules: tuple[int, int, int, int]
    trainer_modules: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        modules = self.rollout_modules + self.trainer_modules
        if (
            len(self.rollout_modules) != 4
            or len(self.trainer_modules) != 4
            or any(module < 0 for module in modules)
            or len(set(modules)) != 8
        ):
            raise ValueError("D5 requires eight distinct non-negative HPU modules")


@dataclass(frozen=True)
class PolicyDescriptor:
    version: str
    update_index: int
    adapter_sha256: str
    config_sha256: str
    model_revision: str
    tokenizer_sha256: str
    grammar_sha256: str
    run_kind: str

    def __post_init__(self) -> None:
        if not self.version or self.update_index < 0 or not self.model_revision:
            raise ValueError("Policy descriptor requires version, index, and revision")
        if self.run_kind not in {"production", "validation"}:
            raise ValueError("Policy run_kind must be production or validation")
        for name in (
            "adapter_sha256", "config_sha256", "tokenizer_sha256", "grammar_sha256"
        ):
            _sha256(getattr(self, name), name)


@dataclass(frozen=True)
class BatchReceipt:
    manifest_sha256: str
    policy_version: str
    adapter_sha256: str
    config_sha256: str
    game_ids: tuple[str, ...]
    match_indices_by_rank: tuple[tuple[int, ...], ...]
    turns_by_rank: tuple[int, ...]
    owned_tokens_by_rank: tuple[int, ...]
    max_replay_error: float

    def __post_init__(self) -> None:
        _sha256(self.manifest_sha256, "manifest_sha256")
        _sha256(self.adapter_sha256, "adapter_sha256")
        _sha256(self.config_sha256, "config_sha256")
        if len(self.match_indices_by_rank) != 4:
            raise ValueError("D5 requires four trainer shards")
        if len(self.turns_by_rank) != 4 or len(self.owned_tokens_by_rank) != 4:
            raise ValueError("Batch coverage requires four trainer counts")
        if any(count <= 0 for count in self.turns_by_rank + self.owned_tokens_by_rank):
            raise ValueError("Every trainer shard must contain turns and owned tokens")
        if not math.isfinite(self.max_replay_error) or self.max_replay_error < 0:
            raise ValueError("Batch replay error must be finite and non-negative")
        if len(self.game_ids) != len(set(self.game_ids)):
            raise ValueError("Rollout batch has duplicate game IDs")
        indices = [index for shard in self.match_indices_by_rank for index in shard]
        if sorted(indices) != list(range(len(self.game_ids))):
            raise ValueError("Trainer shards must cover every match exactly once")


def verify_completed_batch(
    root: str | Path,
    *,
    config: Mapping[str, Any],
    policy: PolicyDescriptor,
    expected_games: int,
) -> BatchReceipt:
    """Hash-check, engine-replay, and credit-check a complete rollout batch.

    The returned receipt is suitable for a coordinator handoff only after all
    checks pass.  This path is model-free; the trainer separately verifies
    behavior probabilities before updating.
    """

    root_path = Path(root)
    manifest = read_pilot_manifest(root_path)
    validate_distributed_training_manifest(manifest, config=config, world_size=4)
    if manifest.target_games != expected_games:
        raise ValueError("Rollout game count differs from the D5 cycle")
    if manifest.policy_version != policy.version:
        raise ValueError("Rollout policy version is stale")
    if manifest.adapter_sha256 != policy.adapter_sha256:
        raise ValueError("Rollout adapter identity differs from the ready policy")
    if manifest.config_sha256 != policy.config_sha256:
        raise ValueError("Rollout configuration identity differs from the ready policy")
    if directory_sha256(root_path / "policy_adapter") != manifest.adapter_sha256:
        raise ValueError("Rollout adapter files differ from the manifest")

    game_ids: list[str] = []
    for entry in manifest.matches:
        match, digest = read_and_replay_pilot_match(root_path, manifest, entry.index)
        if digest != entry.sha256:
            raise ValueError(f"Rollout match digest differs: {entry.path}")
        if make_match_entry(
            index=entry.index,
            match=match,
            path=root_path / entry.path,
            max_abs_log_prob_error=entry.max_abs_log_prob_error,
        ) != entry:
            raise ValueError(f"Rollout match metadata differs: {entry.path}")
        expected_advantages = outcome_advantages(match.final_results)
        for turn in match.turns:
            expected = expected_advantages[turn.seat]
            if turn.credit.outcome_advantage is None or not math.isclose(
                turn.credit.outcome_advantage, expected, abs_tol=1e-9
            ):
                raise ValueError(f"Outcome credit differs: {entry.path} turn {turn.joint_step}")
            if turn.credit.training_advantage is None or not math.isclose(
                turn.credit.training_advantage, expected, abs_tol=1e-9
            ):
                raise ValueError(f"Training credit differs: {entry.path} turn {turn.joint_step}")
        game_ids.append(match.game_id)

    shards = tuple(
        trainer_match_indices(rank, 4, len(manifest.matches)) for rank in range(4)
    )
    receipt = BatchReceipt(
        manifest_sha256=file_sha256(root_path / "manifest.json"),
        policy_version=manifest.policy_version,
        adapter_sha256=manifest.adapter_sha256,
        config_sha256=manifest.config_sha256,
        game_ids=tuple(game_ids),
        match_indices_by_rank=shards,
        turns_by_rank=tuple(sum(manifest.matches[i].turns for i in shard) for shard in shards),
        owned_tokens_by_rank=tuple(
            sum(manifest.matches[i].owned_tokens for i in shard) for shard in shards
        ),
        max_replay_error=max(
            entry.max_abs_log_prob_error for entry in manifest.matches
        ),
    )
    if receipt.max_replay_error > manifest.replay_tolerance:
        raise ValueError("Rollout replay error exceeds the frozen tolerance")
    return receipt


@dataclass
class CycleCoordinator:
    """One-update lifecycle; invalid acknowledgements fail closed."""

    layout: RoleLayout
    policy: PolicyDescriptor
    config: Mapping[str, Any]
    expected_games: int
    phase: Phase = Phase.READY
    ready: set[tuple[str, int]] = field(default_factory=set)
    verified: set[int] = field(default_factory=set)
    updated: dict[int, tuple[str, str, str, int]] = field(default_factory=dict)
    refreshed: set[int] = field(default_factory=set)
    batch: BatchReceipt | None = None
    next_policy: PolicyDescriptor | None = None
    checkpoint_sha256: str | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.expected_games <= 0 or self.expected_games % 4:
            raise ValueError("D5 game count must be positive and divisible by four")
        if canonical_sha256(self.config) != self.policy.config_sha256:
            raise ValueError("Coordinator config differs from its policy descriptor")
        rollout = self.config.get("rollout")
        if not isinstance(rollout, Mapping) or rollout.get("games_per_update") != self.expected_games:
            raise ValueError("Coordinator game count differs from the rollout config")

    def _require(self, phase: Phase) -> None:
        if self.phase != phase:
            self._fail(f"Expected phase {phase.value}, got {self.phase.value}")

    def _fail(self, message: str) -> None:
        self.failure = message
        self.phase = Phase.FAILED
        raise CycleError(message)

    def acknowledge_ready(self, role: str, rank: int, policy: PolicyDescriptor) -> None:
        self._require(Phase.READY)
        key = (role, rank)
        if role not in {"rollout", "trainer"} or rank not in range(4):
            self._fail("Unknown D5 worker role or rank")
        if key in self.ready or policy != self.policy:
            self._fail("Duplicate or stale ready acknowledgement")
        self.ready.add(key)

    def begin_collection(self) -> None:
        self._require(Phase.READY)
        required = {(role, rank) for role in ("rollout", "trainer") for rank in range(4)}
        if self.ready != required:
            self._fail("Collection requires all eight matching ready acknowledgements")
        self.phase = Phase.COLLECTING

    def commit_batch(self, root: str | Path) -> BatchReceipt:
        self._require(Phase.COLLECTING)
        try:
            receipt = verify_completed_batch(
                root, config=self.config, policy=self.policy,
                expected_games=self.expected_games,
            )
        except Exception as exc:
            self._fail(f"Complete rollout verification failed: {exc}")
        self.batch = receipt
        self.phase = Phase.BATCH_COMMITTED
        return receipt

    def acknowledge_verified(
        self,
        rank: int,
        *,
        manifest_sha256: str,
        match_indices: tuple[int, ...],
        turns: int,
        owned_tokens: int,
        max_replay_error: float,
    ) -> None:
        self._require(Phase.BATCH_COMMITTED)
        assert self.batch is not None
        if rank not in range(4) or rank in self.verified:
            self._fail("Duplicate or invalid trainer verification rank")
        if (
            manifest_sha256 != self.batch.manifest_sha256
            or match_indices != self.batch.match_indices_by_rank[rank]
            or turns != self.batch.turns_by_rank[rank]
            or owned_tokens != self.batch.owned_tokens_by_rank[rank]
            or not math.isfinite(max_replay_error)
            or max_replay_error < 0
            or max_replay_error > self.batch.max_replay_error
        ):
            self._fail("Trainer verification differs from the committed batch")
        self.verified.add(rank)
        if len(self.verified) == 4:
            self.phase = Phase.VERIFIED

    def acknowledge_update(
        self,
        rank: int,
        *,
        manifest_sha256: str,
        next_policy: PolicyDescriptor,
        parameter_sha256: str,
        optimizer_sha256: str,
        optimizer_steps: int,
        gradient_sync_phases: int,
        changed_tensors: int,
    ) -> None:
        self._require(Phase.VERIFIED)
        assert self.batch is not None
        if rank not in range(4) or rank in self.updated:
            self._fail("Duplicate or invalid trainer update rank")
        if manifest_sha256 != self.batch.manifest_sha256:
            self._fail("Trainer updated against a different rollout batch")
        if (
            next_policy.update_index != self.policy.update_index + 1
            or next_policy.version == self.policy.version
            or next_policy.config_sha256 != self.policy.config_sha256
            or next_policy.model_revision != self.policy.model_revision
            or next_policy.tokenizer_sha256 != self.policy.tokenizer_sha256
            or next_policy.grammar_sha256 != self.policy.grammar_sha256
            or next_policy.run_kind != self.policy.run_kind
            or optimizer_steps != next_policy.update_index
            or gradient_sync_phases != 1
            or changed_tensors < 0
        ):
            self._fail("Trainer update metadata is incompatible with the current policy")
        try:
            _sha256(parameter_sha256, "parameter_sha256")
            _sha256(optimizer_sha256, "optimizer_sha256")
        except ValueError as exc:
            self._fail(str(exc))
        if self.next_policy is not None and next_policy != self.next_policy:
            self._fail("Trainer ranks reported different next policies")
        self.next_policy = next_policy
        self.updated[rank] = (
            parameter_sha256, optimizer_sha256,
            next_policy.adapter_sha256, changed_tensors,
        )
        if len(self.updated) == 4:
            if len(set(self.updated.values())) != 1:
                self._fail("Trainer parameter, optimizer, or adapter replicas differ")
            if changed_tensors > 0 and next_policy.adapter_sha256 == self.policy.adapter_sha256:
                self._fail("Trainer changed tensors but published the old adapter identity")
            self.phase = Phase.UPDATED

    def commit_checkpoint(
        self, *, manifest_sha256: str, checkpoint_sha256: str,
        policy: PolicyDescriptor,
    ) -> None:
        self._require(Phase.UPDATED)
        assert self.batch is not None
        if manifest_sha256 != self.batch.manifest_sha256 or policy != self.next_policy:
            self._fail("Checkpoint does not identify the committed update")
        try:
            self.checkpoint_sha256 = _sha256(checkpoint_sha256, "checkpoint_sha256")
        except ValueError as exc:
            self._fail(str(exc))
        self.phase = Phase.REFRESHING

    def acknowledge_refreshed(
        self, rank: int, *, policy: PolicyDescriptor, probe_error: float,
        probe_tolerance: float,
    ) -> None:
        self._require(Phase.REFRESHING)
        if rank not in range(4) or rank in self.refreshed or policy != self.next_policy:
            self._fail("Duplicate or stale rollout refresh acknowledgement")
        if (
            not math.isfinite(probe_error) or probe_error < 0
            or not math.isfinite(probe_tolerance) or probe_tolerance < 0
            or probe_error > probe_tolerance
        ):
            self._fail("Rollout refresh probability probe failed")
        self.refreshed.add(rank)
        if len(self.refreshed) == 4:
            assert self.next_policy is not None
            self.policy = self.next_policy
            self.next_policy = None
            self.batch = None
            self.ready.clear()
            self.verified.clear()
            self.updated.clear()
            self.refreshed.clear()
            self.phase = Phase.READY

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "policy_version": self.policy.version,
            "update_index": self.policy.update_index,
            "ready_acknowledgements": len(self.ready),
            "verified_trainers": len(self.verified),
            "updated_trainers": len(self.updated),
            "refreshed_rollouts": len(self.refreshed),
            "batch_sha256": self.batch.manifest_sha256 if self.batch else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "failure": self.failure,
        }
