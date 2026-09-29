"""Refresh four rollout replicas after the resumed multi-update checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from self_play_grpo.config import ExperimentConfig, expected_gradient_sync_phases, load_config
from self_play_grpo.policies.llm import ConstrainedLLMPolicy
from self_play_grpo.rollouts.pilot import (
    canonical_sha256, directory_sha256, file_sha256,
    read_pilot_manifest, restore_initial_adapter,
)
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout
from self_play_grpo.training.d5_preflight import verify_d5_launch_prerequisites
from self_play_grpo.training.multi_identity import checkpoint_code_identity
from self_play_grpo.training.distributed_checkpoint import (
    read_distributed_manifest, verify_checkpoint_files,
)
from self_play_grpo.training.identity import trainable_parameter_sha256
from self_play_grpo.training.policy_refresh import compare_probe, policy_probe, validate_probe
from self_play_grpo.training.refresh_gate import _REPORT_KEYS, _digest, publish_refresh_evidence
from self_play_grpo.training.rollout_worker import _worker_binding
from self_play_grpo.training.trainer_worker import _modules


def _source_and_checkpoint(
    rollout_root: Path, trainer_output: Path, config: ExperimentConfig,
    *, run_id: str,
) -> tuple[PolicyDescriptor, dict[str, Any], Path, str]:
    source_manifest = read_pilot_manifest(rollout_root)
    source_digest = file_sha256(rollout_root / "manifest.json")
    descriptor = PolicyDescriptor(**json.loads(
        (rollout_root / "policy_descriptor.json").read_text(encoding="utf-8")
    ))
    if (len(source_manifest.matches) != 64 or descriptor.run_kind != "production"
            or descriptor.version != f"policy-{descriptor.update_index:06d}" or descriptor.update_index < 1
            or source_manifest.policy_version != descriptor.version
            or source_manifest.adapter_sha256 != descriptor.adapter_sha256
            or source_manifest.config_sha256 != canonical_sha256(config.to_dict())
            or descriptor.config_sha256 != source_manifest.config_sha256):
        raise ValueError("D6 refresh source is not a complete policy-000001 batch")
    summary = json.loads((trainer_output / "trainer_summary.json").read_text(encoding="utf-8"))
    checkpoint = trainer_output / "trainer" / "checkpoints" / f"policy-{descriptor.update_index + 1:06d}"
    if (summary.get("status") != "updated_checkpoint_committed_refresh_pending"
            or summary.get("run_id") != run_id
            or Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or summary.get("source_manifest_sha256") != source_digest
            or summary.get("source_policy_version") != descriptor.version):
        raise ValueError("D6 trainer summary source/checkpoint differs")
    checkpoint_digest = _digest(summary.get("checkpoint_manifest_sha256"), "checkpoint")
    adapter_digest = _digest(summary.get("checkpoint_adapter_sha256"), "adapter")
    _digest(summary.get("parameter_sha256"), "parameters")
    validate_probe(summary.get("policy_probe"))
    if (checkpoint.is_symlink()
            or file_sha256(checkpoint / "distributed" / "manifest.json") != checkpoint_digest):
        raise ValueError("D6 checkpoint manifest digest differs")
    manifest = read_distributed_manifest(checkpoint)
    verify_checkpoint_files(checkpoint, manifest)
    if (manifest.run_kind != "production" or manifest.run_id != run_id
            or manifest.policy_version != f"policy-{descriptor.update_index + 1:06d}" or manifest.update_index != descriptor.update_index + 1
            or manifest.source_rollout_manifest_sha256 != source_digest
            or manifest.config_sha256 != descriptor.config_sha256
            or manifest.model_id != config.model.id
            or manifest.model_revision != descriptor.model_revision
            or manifest.tokenizer_sha256 != descriptor.tokenizer_sha256
            or manifest.grammar_sha256 != descriptor.grammar_sha256
            or manifest.code_identity != checkpoint_code_identity(manifest.update_index)
            or manifest.trainer_world_size != 4):
        raise ValueError("D6 checkpoint identity differs from resumed production update")
    if directory_sha256(checkpoint / "adapter") != adapter_digest:
        raise ValueError("D6 checkpoint adapter digest differs")
    return descriptor, summary, checkpoint, source_digest


def refresh_rank(args: Any) -> dict[str, Any]:
    """Load the committed update on one rollout HPU and verify its recorded-shape probe."""

    config = load_config(args.config)
    rollout_modules, trainer_modules = _modules(args.rollout_modules), _modules(args.trainer_modules)
    layout = RoleLayout(rollout_modules, trainer_modules)
    verify_d5_launch_prerequisites(
        config_path=args.config, two_rank_d4_summary=args.d4_two_summary,
        four_rank_d4_summary=args.d4_four_summary, layout=layout,
        worker_role="rollout",
    )
    module_id = _worker_binding(args.rank)
    if module_id != rollout_modules[args.rank]:
        raise RuntimeError("D6 refresh module differs from requested layout")
    if args.output.resolve() in (args.rollout_root.resolve(), args.trainer_output.resolve()):
        raise ValueError("D6 refresh output must be separate")
    target = args.output / f"rank-{args.rank:03d}.json"
    if target.exists():
        raise FileExistsError("D6 refresh rank report already exists")
    descriptor, summary, checkpoint, source_digest = _source_and_checkpoint(
        args.rollout_root, args.trainer_output, config, run_id=args.run_id,
    )
    expected_probe = validate_probe(summary["policy_probe"])
    adapter_digest = summary["checkpoint_adapter_sha256"]
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    restore_initial_adapter(policy.model, checkpoint / "adapter", adapter_digest)
    policy.model.eval()
    parameter_digest = trainable_parameter_sha256(policy.model)
    if parameter_digest != summary["parameter_sha256"]:
        raise RuntimeError("D6 refreshed rollout adapter tensor values differ from trainer")
    actual_probe = policy_probe(policy.model, args.rollout_root)
    tolerance = read_pilot_manifest(args.rollout_root).replay_tolerance
    error = compare_probe(expected_probe, actual_probe, tolerance)
    report = {
        "status": "refresh_verified", "rank": args.rank, "module_id": module_id,
        "policy_version": f"policy-{descriptor.update_index + 1:06d}", "source_manifest_sha256": source_digest,
        "checkpoint_manifest_sha256": summary["checkpoint_manifest_sha256"],
        "adapter_sha256": adapter_digest, "parameter_sha256": parameter_digest,
        "probe_max_abs_error": error,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return report


def aggregate_refresh(
    *, rollout_root: Path, trainer_output: Path, refresh_output: Path,
    config: ExperimentConfig, layout: RoleLayout, run_id: str,
) -> tuple[PolicyDescriptor, dict[str, Any]]:
    """CPU-only verification of trainer and all four multi-update refresh reports."""

    prior, summary, checkpoint, source_digest = _source_and_checkpoint(
        rollout_root, trainer_output, config, run_id=run_id,
    )
    checkpoint_digest = summary["checkpoint_manifest_sha256"]
    adapter_digest = summary["checkpoint_adapter_sha256"]
    parameter_digest = summary["parameter_sha256"]
    trainer_reports = []
    for rank in range(4):
        row = json.loads((trainer_output / "ranks" / f"rank-{rank:03d}.json").read_text(encoding="utf-8"))
        update = row.get("update", {})
        if (row.get("status") != "updated_checkpoint_committed_refresh_pending"
                or row.get("rank") != rank
                or row.get("module_id") != str(layout.trainer_modules[rank])
                or row.get("source_manifest_sha256") != source_digest
                or row.get("checkpoint_manifest_sha256") != checkpoint_digest
                or row.get("checkpoint_adapter_sha256") != adapter_digest
                or row.get("source_policy_version") != prior.version
                or update.get("source_policy_version") != prior.version
                or update.get("next_policy_version") != f"policy-{prior.update_index + 1:06d}"
                or update.get("parameter_sha256") != parameter_digest
                or update.get("consumed_manifest_sha256") != source_digest
                or update.get("optimizer_steps") != prior.update_index + 1
                or update.get("gradient_sync_phases") != expected_gradient_sync_phases(config)):
            raise ValueError(f"D6 trainer rank {rank} evidence differs")
        trainer_reports.append(row)
    optimizer_digests = {row["update"].get("optimizer_sha256") for row in trainer_reports}
    if len(optimizer_digests) != 1:
        raise ValueError("D6 trainer optimizer replicas differ")
    optimizer_digest = _digest(next(iter(optimizer_digests)), "optimizer")
    errors = []
    tolerance = read_pilot_manifest(rollout_root).replay_tolerance
    for rank in range(4):
        row = json.loads((refresh_output / f"rank-{rank:03d}.json").read_text(encoding="utf-8"))
        if not isinstance(row, dict) or set(row) != _REPORT_KEYS:
            raise ValueError(f"D6 refresh rank {rank} report schema differs")
        if (row["status"] != "refresh_verified" or type(row["rank"]) is not int
                or row["rank"] != rank or type(row["module_id"]) is not int
                or row["module_id"] != layout.rollout_modules[rank]
                or row["policy_version"] != f"policy-{prior.update_index + 1:06d}"
                or row["source_manifest_sha256"] != source_digest
                or row["checkpoint_manifest_sha256"] != checkpoint_digest
                or row["adapter_sha256"] != adapter_digest
                or row["parameter_sha256"] != parameter_digest):
            raise ValueError(f"D6 refresh rank {rank} identity differs")
        error = row["probe_max_abs_error"]
        if (type(error) not in (float, int) or not math.isfinite(error)
                or error < 0 or error > tolerance):
            raise ValueError(f"D6 refresh rank {rank} probe failed")
        errors.append(float(error))
    next_policy = PolicyDescriptor(
        version=f"policy-{prior.update_index + 1:06d}", update_index=prior.update_index + 1,
        adapter_sha256=adapter_digest, config_sha256=prior.config_sha256,
        model_revision=prior.model_revision,
        tokenizer_sha256=prior.tokenizer_sha256,
        grammar_sha256=prior.grammar_sha256, run_kind="production",
    )
    evidence = {
        "status": "refresh_reports_verified",
        "policy_version": next_policy.version,
        "source_manifest_sha256": source_digest,
        "checkpoint_manifest_sha256": checkpoint_digest,
        "adapter_sha256": adapter_digest,
        "parameter_sha256": parameter_digest,
        "optimizer_sha256": optimizer_digest,
        "refresh_ranks": 4,
        "max_abs_probe_error": max(errors),
    }
    return next_policy, evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D6 multi-update rollout refresh rank")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--trainer-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--rollout-modules", required=True)
    parser.add_argument("--trainer-modules", required=True)
    parser.add_argument("--d4-two-summary", required=True, type=Path)
    parser.add_argument("--d4-four-summary", required=True, type=Path)
    print(json.dumps(refresh_rank(parser.parse_args(argv)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

