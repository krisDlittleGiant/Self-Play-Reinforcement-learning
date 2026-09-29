"""Build a production format-3 checkpoint template from a committed D5 batch.

This is metadata preparation only. The four ranks must separately stage their
HPU RNG records before rank zero calls ``SynchronousTrainer.save_checkpoint``.
That saver fills the file inventory and publishes the checkpoint atomically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from self_play_grpo.rollouts.pilot import canonical_sha256, file_sha256, read_pilot_manifest
from self_play_grpo.training.coordinator import BatchReceipt, PolicyDescriptor
from self_play_grpo.training.distributed import (
    trainer_match_indices, validate_distributed_training_manifest,
)
from self_play_grpo.training.distributed_checkpoint import (
    DistributedCheckpointManifest, TrainerRankState,
)
from self_play_grpo.training.distributed_resume_gate import (
    _adapter_schema, _optimizer_schema,
)


_CODE_FILES = (
    "distributed.py",
    "policies/llm.py",
    "training/coordinator.py",
    "training/d5_preflight.py",
    "training/distributed.py",
    "training/distributed_checkpoint.py",
    "training/distributed_resume_gate.py",
    "training/identity.py",
    "training/initial_policy.py",
    "training/loop.py",
    "training/loss.py",
    "training/process_supervisor.py",
    "training/phase_supervisor.py",
    "training/policy_refresh.py",
    "training/refresh_gate.py",
    "training/initial_cycle.py",
    "training/production_checkpoint.py",
    "training/rollout_worker.py",
    "training/trainer_handoff.py",
    "training/trainer_update.py",
    "training/trainer_worker.py",
)


def production_code_identity() -> str:
    """Fingerprint the exact core source files that define a D5 update."""

    package = Path(__file__).resolve().parents[1]
    return "source-sha256:" + canonical_sha256({
        name: file_sha256(package / name) for name in _CODE_FILES
    })


def build_production_checkpoint_template(
    *,
    run_id: str,
    rollout_root: str | Path,
    receipt: BatchReceipt,
    policy: PolicyDescriptor,
    trainer: Any,
    trainer_module_ids: Sequence[int],
    runtime_identity: Mapping[str, str],
) -> DistributedCheckpointManifest:
    """Bind the next checkpoint to one exact, already-verified 64-game batch."""

    if policy.run_kind != "production":
        raise ValueError("Production checkpoint requires a production policy")
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("Production run_id must be a single safe name")
    if (len(trainer_module_ids) != 4 or len(set(trainer_module_ids)) != 4
            or any(type(module_id) is not int or module_id < 0 for module_id in trainer_module_ids)):
        raise ValueError("Production checkpoint needs four distinct trainer module IDs")
    if (trainer.update_index != policy.update_index + 1
            or trainer.policy_version != f"policy-{trainer.update_index:06d}"):
        raise ValueError("Trainer has not advanced exactly one policy version")
    config = trainer.config
    if canonical_sha256(config.to_dict()) != policy.config_sha256:
        raise ValueError("Checkpoint config differs from source policy")
    if (config.model.id == "" or config.model.revision != policy.model_revision
            or config.rollout.games_per_update != 64):
        raise ValueError("Checkpoint model or batch profile differs")
    root = Path(rollout_root)
    if file_sha256(root / "manifest.json") != receipt.manifest_sha256:
        raise ValueError("Checkpoint source rollout digest differs from receipt")
    manifest = read_pilot_manifest(root)
    validate_distributed_training_manifest(manifest, config=config.to_dict(), world_size=4)
    if (manifest.policy_version != policy.version
            or manifest.adapter_sha256 != policy.adapter_sha256
            or manifest.config_sha256 != policy.config_sha256
            or receipt.policy_version != policy.version
            or receipt.adapter_sha256 != policy.adapter_sha256
            or receipt.config_sha256 != policy.config_sha256):
        raise ValueError("Checkpoint source batch differs from policy")
    expected_runtime = {"python", "torch", "transformers", "peft", "habana"}
    if set(runtime_identity) != expected_runtime or any(not value for value in runtime_identity.values()):
        raise ValueError("Checkpoint runtime identity is incomplete")

    ranks = []
    for rank in range(4):
        indices = trainer_match_indices(rank, 4, len(manifest.matches))
        if receipt.match_indices_by_rank[rank] != indices:
            raise ValueError("Checkpoint trainer shard differs from receipt")
        entries = tuple(manifest.matches[index] for index in indices)
        if tuple(entry.game_id for entry in entries) != tuple(receipt.game_ids[index] for index in indices):
            raise ValueError("Checkpoint match IDs differ from receipt")
        record = TrainerRankState(
            rank=rank, local_rank=rank, module_id=str(trainer_module_ids[rank]),
            device=f"hpu:{rank}", rng_path=f"distributed/rng/rank-{rank:03d}.pt",
            match_indices=indices,
            match_ids=tuple(entry.game_id for entry in entries),
            match_sha256=tuple(entry.sha256 for entry in entries),
        )
        record.validate()
        ranks.append(record)
    if file_sha256(root / "manifest.json") != receipt.manifest_sha256:
        raise ValueError("Checkpoint source rollout changed during template construction")
    template = DistributedCheckpointManifest(
        run_kind="production", run_id=run_id,
        policy_version=trainer.policy_version, update_index=trainer.update_index,
        config_sha256=policy.config_sha256,
        model_id=config.model.id, model_revision=config.model.revision,
        adapter_schema_sha256=_adapter_schema(trainer),
        optimizer_schema_sha256=_optimizer_schema(trainer),
        tokenizer_sha256=policy.tokenizer_sha256,
        grammar_sha256=policy.grammar_sha256,
        code_identity=production_code_identity(),
        runtime_identity=tuple(sorted(runtime_identity.items())),
        attention_backend=str(getattr(trainer.policy.model.config, "_attn_implementation", "unreported")),
        dtype=config.model.dtype, backend="hccl", trainer_world_size=4,
        source_rollout_manifest_sha256=receipt.manifest_sha256,
        ranks=tuple(ranks), files=(),
    )
    # The missing file inventory is intentional until save_checkpoint stages
    # adapter, optimizer, trainer state and all four rank RNG files.
    return template
