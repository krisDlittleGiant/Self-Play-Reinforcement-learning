"""CPU-only aggregation of four post-update rollout refresh reports.

This verifies evidence, but does not commit the update ledger or start the
next rollout. A D5 cycle driver must do those steps explicitly afterwards.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from self_play_grpo.config import ExperimentConfig, expected_gradient_sync_phases
from self_play_grpo.rollouts.pilot import (
    canonical_sha256, directory_sha256, file_sha256, read_pilot_manifest,
)
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout
from self_play_grpo.training.distributed_checkpoint import (
    read_distributed_manifest, verify_checkpoint_files,
)
from self_play_grpo.training.policy_refresh import validate_probe
from self_play_grpo.training.production_checkpoint import production_code_identity


_REPORT_KEYS = {
    "status", "rank", "module_id", "policy_version", "source_manifest_sha256",
    "checkpoint_manifest_sha256", "adapter_sha256", "parameter_sha256",
    "probe_max_abs_error",
}


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"D5 {label} must be a SHA-256 digest")
    return value


def validate_refresh_reports(
    reports: Mapping[int, Any], *, layout: RoleLayout,
    source_manifest_sha256: str, checkpoint_manifest_sha256: str,
    adapter_sha256: str, parameter_sha256: str, tolerance: float,
) -> float:
    """Reject missing, duplicated, stale, or mismatched rollout replicas."""

    for name, value in (
        ("source manifest", source_manifest_sha256),
        ("checkpoint manifest", checkpoint_manifest_sha256),
        ("adapter", adapter_sha256), ("parameters", parameter_sha256),
    ):
        _digest(value, name)
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("D5 refresh tolerance must be finite and positive")
    if set(reports) != set(range(4)):
        raise ValueError("D5 refresh requires exactly four rollout rank reports")
    errors = []
    for rank in range(4):
        row = reports[rank]
        if not isinstance(row, dict) or set(row) != _REPORT_KEYS:
            raise ValueError(f"D5 refresh rank {rank} report schema differs")
        if (row["status"] != "refresh_verified" or type(row["rank"]) is not int
                or row["rank"] != rank or type(row["module_id"]) is not int
                or row["module_id"] != layout.rollout_modules[rank]
                or row["policy_version"] != "policy-000001"
                or row["source_manifest_sha256"] != source_manifest_sha256
                or row["checkpoint_manifest_sha256"] != checkpoint_manifest_sha256
                or row["adapter_sha256"] != adapter_sha256
                or row["parameter_sha256"] != parameter_sha256):
            raise ValueError(f"D5 refresh rank {rank} identity differs")
        error = row["probe_max_abs_error"]
        if (type(error) not in (int, float) or not math.isfinite(error)
                or error < 0 or error > tolerance):
            raise ValueError(f"D5 refresh rank {rank} probability probe failed")
        errors.append(float(error))
    return max(errors)


def aggregate_refresh_evidence(
    *, rollout_root: str | Path, trainer_output: str | Path,
    refresh_output: str | Path, config: ExperimentConfig,
    layout: RoleLayout,
) -> tuple[PolicyDescriptor, dict[str, Any]]:
    """Validate source, checkpoint, all trainer reports and all refresh ranks."""

    source = Path(rollout_root)
    trained = Path(trainer_output)
    refreshed = Path(refresh_output)
    source_digest = file_sha256(source / "manifest.json")
    source_manifest = read_pilot_manifest(source)
    if (len(source_manifest.matches) != 64
            or source_manifest.config_sha256 != canonical_sha256(config.to_dict())):
        raise ValueError("D5 refresh source is not the current complete 64-game batch")
    prior = PolicyDescriptor(**json.loads((source / "policy_descriptor.json").read_text(encoding="utf-8")))
    if (prior.run_kind != "production" or prior.update_index != 0
            or source_manifest.policy_version != prior.version
            or source_manifest.adapter_sha256 != prior.adapter_sha256
            or prior.config_sha256 != source_manifest.config_sha256):
        raise ValueError("D5 refresh source policy descriptor differs")
    summary = json.loads((trained / "trainer_summary.json").read_text(encoding="utf-8"))
    checkpoint = trained / "trainer" / "checkpoints" / "policy-000001"
    if (summary.get("status") != "updated_checkpoint_committed_refresh_pending"
            or Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or summary.get("source_manifest_sha256") != source_digest):
        raise ValueError("D5 trainer summary source/checkpoint differs")
    manifest_digest = _digest(summary.get("checkpoint_manifest_sha256"), "checkpoint manifest")
    adapter_digest = _digest(summary.get("checkpoint_adapter_sha256"), "checkpoint adapter")
    parameters_digest = _digest(summary.get("parameter_sha256"), "trainer parameters")
    validate_probe(summary.get("policy_probe"))
    if checkpoint.is_symlink() or file_sha256(checkpoint / "distributed" / "manifest.json") != manifest_digest:
        raise ValueError("D5 checkpoint manifest digest differs")
    checkpoint_manifest = read_distributed_manifest(checkpoint)
    verify_checkpoint_files(checkpoint, checkpoint_manifest)
    if (checkpoint_manifest.run_kind != "production"
            or checkpoint_manifest.policy_version != "policy-000001"
            or checkpoint_manifest.update_index != 1
            or checkpoint_manifest.source_rollout_manifest_sha256 != source_digest
            or checkpoint_manifest.config_sha256 != prior.config_sha256
            or checkpoint_manifest.model_id != config.model.id
            or checkpoint_manifest.model_revision != prior.model_revision
            or checkpoint_manifest.tokenizer_sha256 != prior.tokenizer_sha256
            or checkpoint_manifest.grammar_sha256 != prior.grammar_sha256
            or checkpoint_manifest.code_identity != production_code_identity()
            or checkpoint_manifest.trainer_world_size != 4):
        raise ValueError("D5 checkpoint identity differs from the production run")
    if directory_sha256(checkpoint / "adapter") != adapter_digest:
        raise ValueError("D5 checkpoint adapter digest differs")
    trainer_reports = []
    for rank in range(4):
        row = json.loads((trained / "ranks" / f"rank-{rank:03d}.json").read_text(encoding="utf-8"))
        update = row.get("update", {})
        if (row.get("status") != "updated_checkpoint_committed_refresh_pending"
                or row.get("rank") != rank
                or row.get("module_id") != str(layout.trainer_modules[rank])
                or row.get("source_manifest_sha256") != source_digest
                or row.get("checkpoint_manifest_sha256") != manifest_digest
                or row.get("checkpoint_adapter_sha256") != adapter_digest
                or update.get("parameter_sha256") != parameters_digest
                or update.get("next_policy_version") != "policy-000001"
                or update.get("consumed_manifest_sha256") != source_digest
                or update.get("optimizer_steps") != 1
                or update.get("gradient_sync_phases") != expected_gradient_sync_phases(config)):
            raise ValueError(f"D5 trainer rank {rank} report differs")
        trainer_reports.append(row)
    optimizer_digests = {row["update"].get("optimizer_sha256") for row in trainer_reports}
    if len(optimizer_digests) != 1:
        raise ValueError("D5 trainer optimizer replicas differ")
    _digest(next(iter(optimizer_digests)), "optimizer")
    reports = {
        rank: json.loads((refreshed / f"rank-{rank:03d}.json").read_text(encoding="utf-8"))
        for rank in range(4)
    }
    error = validate_refresh_reports(
        reports, layout=layout, source_manifest_sha256=source_digest,
        checkpoint_manifest_sha256=manifest_digest, adapter_sha256=adapter_digest,
        parameter_sha256=parameters_digest, tolerance=source_manifest.replay_tolerance,
    )
    next_policy = PolicyDescriptor(
        version="policy-000001", update_index=1, adapter_sha256=adapter_digest,
        config_sha256=prior.config_sha256, model_revision=prior.model_revision,
        tokenizer_sha256=prior.tokenizer_sha256, grammar_sha256=prior.grammar_sha256,
        run_kind="production",
    )
    evidence = {
        "status": "refresh_reports_verified",
        "policy_version": next_policy.version,
        "source_manifest_sha256": source_digest,
        "checkpoint_manifest_sha256": manifest_digest,
        "adapter_sha256": adapter_digest,
        "parameter_sha256": parameters_digest,
        "optimizer_sha256": next(iter(optimizer_digests)),
        "refresh_ranks": 4,
        "max_abs_probe_error": error,
    }
    return next_policy, evidence


def publish_refresh_evidence(output: str | Path, evidence: Mapping[str, Any]) -> Path:
    """Publish verified evidence exclusively, before the cycle ledger write."""

    if evidence.get("status") != "refresh_reports_verified":
        raise ValueError("D5 refresh evidence is not verified")
    root = Path(output)
    if root.is_symlink():
        raise ValueError("D5 refresh evidence root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    target = root / "refresh_summary.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(dict(evidence), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target
