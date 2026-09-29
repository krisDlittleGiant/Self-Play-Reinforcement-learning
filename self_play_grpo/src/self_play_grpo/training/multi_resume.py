"""Fail-closed cold-process restore of the committed production trainer update.

The caller owns the four-rank HCCL lifecycle and constructs a fresh
``SynchronousTrainer`` on each rank. This module does not acquire an HPU at
import time or take an optimizer step. It restores the format-3 checkpoint
only after validating the committed D5 identity against the active runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from self_play_grpo.rollouts.pilot import canonical_sha256
from self_play_grpo.training.multi_recovery import RecoveryPlan, _json_file
from self_play_grpo.training.distributed_checkpoint import (
    DistributedCheckpointManifest, read_distributed_manifest,
    validate_resume_identity, verify_checkpoint_files,
)
from self_play_grpo.training.distributed_resume_gate import (
    _adapter_schema, _expected_identity, _optimizer_schema,
)
from self_play_grpo.training.identity import (
    optimizer_state_sha256, trainable_parameter_sha256,
)
from self_play_grpo.training.ledger import read_commits
from self_play_grpo.training.multi_identity import checkpoint_code_identity


def production_resume_expectation(
    trainer: Any, manifest: DistributedCheckpointManifest, *,
    plan: RecoveryPlan, rank: int, trainer_module_ids: Sequence[int],
    runtime_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Construct an independent expectation before adapter/optimizer mutation."""

    if plan.status != "ready_for_next_collection":
        raise RuntimeError("D6 trainer resume requires four verified refresh ranks")
    if plan.committed_update < 2 or plan.next_update != plan.committed_update + 1:
        raise ValueError("Multi-update continuation starts from update 2 or later")
    if (len(trainer_module_ids) != 4 or len(set(trainer_module_ids)) != 4
            or any(type(item) is not int or item < 0 for item in trainer_module_ids)):
        raise ValueError("D6 resume requires four distinct trainer module IDs")
    if type(rank) is not int or rank not in range(4):
        raise ValueError("D6 resume rank must be in [0, 4)")
    if (trainer.update_index != 0 or trainer.policy_version != "policy-000000"
            or getattr(trainer, "evaluator", None) is not None):
        raise ValueError("D6 resume requires a fresh outcome-only trainer")
    config = trainer.config
    expected_runtime_keys = {"python", "torch", "transformers", "peft", "habana"}
    if set(runtime_identity) != expected_runtime_keys or any(not value for value in runtime_identity.values()):
        raise ValueError("D6 runtime identity is incomplete")
    expected = _expected_identity(manifest)
    expected.update({
        "run_id": plan.run_id,
        "policy_version": plan.policy_version,
        "update_index": plan.committed_update,
        "config_sha256": canonical_sha256(config.to_dict()),
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "adapter_schema_sha256": _adapter_schema(trainer),
        "optimizer_schema_sha256": _optimizer_schema(trainer),
        "code_identity": checkpoint_code_identity(plan.committed_update),
        "runtime_identity": dict(runtime_identity),
        "attention_backend": str(getattr(trainer.policy.model.config, "_attn_implementation", "unreported")),
        "dtype": config.model.dtype,
        "backend": "hccl",
        "trainer_world_size": 4,
        "source_rollout_manifest_sha256": plan.source_manifest_sha256,
        "rank_bindings": [
            {"rank": item, "local_rank": item,
             "module_id": str(trainer_module_ids[item]), "device": f"hpu:{item}"}
            for item in range(4)
        ],
    })
    return expected


def restore_production_update(
    trainer: Any, *, root: str | Path, plan: RecoveryPlan,
    rank: int, trainer_module_ids: Sequence[int],
    runtime_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Restore policy, optimizer and this rank's CPU/HPU RNG from the ledger head.

    The existing distributed loader validates all checkpoint files, then
    restores the adapter, optimizer and per-rank RNG. This wrapper adds the
    production ledger/refresh and active-runtime identity checks, plus a
    post-restore tensor/optimizer fingerprint comparison. It performs no step.
    """

    root = Path(root)
    records = read_commits(root)
    if not records:
        raise ValueError("D6 restore requires a committed production ledger")
    head = records[-1]
    if (head.run_id != plan.run_id
            or head.update_index != plan.committed_update
            or head.policy_version != plan.policy_version
            or head.checkpoint_path != plan.checkpoint
            or head.checkpoint_manifest_sha256 != plan.checkpoint_manifest_sha256
            or head.source_manifest_sha256 != plan.source_manifest_sha256):
        raise ValueError("D6 restore plan differs from the authoritative ledger")
    checkpoint = root / plan.checkpoint
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("D6 committed checkpoint is missing or symlinked")
    manifest = read_distributed_manifest(checkpoint)
    verify_checkpoint_files(checkpoint, manifest)
    expected = production_resume_expectation(
        trainer, manifest, plan=plan, rank=rank,
        trainer_module_ids=trainer_module_ids,
        runtime_identity=runtime_identity,
    )
    validate_resume_identity(manifest, expected=expected, allow_validation=False)
    summary = _json_file(root / f"refresh-{plan.committed_update:06d}" / "refresh_summary.json")
    if (summary.get("status") != "refresh_reports_verified"
            or summary.get("policy_version") != plan.policy_version
            or summary.get("checkpoint_manifest_sha256") != plan.checkpoint_manifest_sha256
            or summary.get("source_manifest_sha256") != plan.source_manifest_sha256):
        raise ValueError("D6 refresh summary differs from committed restore boundary")
    expected_parameter_digest = summary.get("parameter_sha256")
    expected_optimizer_digest = summary.get("optimizer_sha256")
    if any(not isinstance(value, str) or len(value) != 64 or
           any(character not in "0123456789abcdef" for character in value)
           for value in (expected_parameter_digest, expected_optimizer_digest)):
        raise ValueError("D6 refresh summary has invalid state fingerprints")
    trainer.load_checkpoint(
        checkpoint, allow_validation=False,
        distributed_expected=expected, distributed_rank=rank,
    )
    if (trainer.update_index != plan.committed_update
            or trainer.policy_version != plan.policy_version):
        raise RuntimeError("D6 trainer restored the wrong update")
    parameter_digest = trainable_parameter_sha256(trainer.policy.model)
    optimizer_digest = optimizer_state_sha256(trainer.optimizer)
    if parameter_digest != expected_parameter_digest or optimizer_digest != expected_optimizer_digest:
        raise RuntimeError("D6 restored parameters or optimizer differ from committed evidence")
    return {
        "status": "restored_no_optimizer_step",
        "rank": rank,
        "module_id": str(trainer_module_ids[rank]),
        "run_id": plan.run_id,
        "policy_version": trainer.policy_version,
        "update_index": trainer.update_index,
        "checkpoint_manifest_sha256": plan.checkpoint_manifest_sha256,
        "parameter_sha256": parameter_digest,
        "optimizer_sha256": optimizer_digest,
    }

