"""Opt-in, fail-closed first D5 production update on a 4+4 Gaudi node.

This is a single update, not the five-update restart driver. Importing this
module does not initialize HPUs or launch child processes. The caller must
provide a recorded pilot root, real D4 evidence,
and an allocation of all eight assigned HPUs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from self_play_grpo.config import ExperimentConfig, load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256
from self_play_grpo.training.coordinator import (
    BatchReceipt, CycleCoordinator, Phase, PolicyDescriptor, RoleLayout,
    verify_completed_batch,
)
from self_play_grpo.training.d5_preflight import verify_d5_launch_prerequisites
from self_play_grpo.training.distributed_checkpoint import (
    read_distributed_manifest, verify_checkpoint_files,
)
from self_play_grpo.training.handoff import (
    acknowledge_refresh_with_identity, publish_cycle_checkpoint,
)
from self_play_grpo.training.ledger import read_commits
from self_play_grpo.training.initial_policy import prepare_initial_policy
from self_play_grpo.training.phase_supervisor import supervise_role_phase
from self_play_grpo.training.process_supervisor import WorkerCommand
from self_play_grpo.training.production_checkpoint import production_code_identity
from self_play_grpo.training.refresh_gate import (
    aggregate_refresh_evidence, publish_refresh_evidence,
)
from self_play_grpo.training.rollout_worker import (
    aggregate_rank_shards, prepare_frozen_batch,
)


def _modules(value: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("D5 module list is malformed") from exc
    if len(result) != 4 or len(set(result)) != 4 or any(item < 0 for item in result):
        raise ValueError("D5 module list must have four distinct non-negative IDs")
    return result  # type: ignore[return-value]


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"D5 {name} is not a SHA-256 digest")
    return value


def _initial_policy(
    descriptor_path: Path, adapter: Path, config: ExperimentConfig,
) -> PolicyDescriptor:
    descriptor = PolicyDescriptor(**json.loads(descriptor_path.read_text(encoding="utf-8")))
    if (descriptor.run_kind != "production" or descriptor.version != "policy-000000"
            or descriptor.update_index != 0
            or descriptor.config_sha256 != canonical_sha256(config.to_dict())
            or descriptor.model_revision != config.model.revision):
        raise ValueError("D5 initial policy descriptor differs from the production config")
    if adapter.is_symlink() or not adapter.is_dir() or any(path.is_symlink() for path in adapter.rglob("*")):
        raise ValueError("D5 initial adapter must be a real directory without symlinks")
    if directory_sha256(adapter) != descriptor.adapter_sha256:
        raise ValueError("D5 initial adapter digest differs from its descriptor")
    return descriptor


def _commands(role: str, modules: tuple[int, int, int, int], arguments: list[str]) -> tuple[WorkerCommand, ...]:
    return tuple(
        WorkerCommand(role, rank, modules[rank], tuple([
            sys.executable, "-m", arguments[0], *arguments[1:], "--rank", str(rank),
        ]))
        for rank in range(4)
    )


def _trainer_evidence(
    output: Path, *, source_digest: str, config: ExperimentConfig,
    prior: PolicyDescriptor, layout: RoleLayout, run_id: str,
) -> tuple[PolicyDescriptor, dict[str, Any], tuple[dict[str, Any], ...]]:
    summary = json.loads((output / "trainer_summary.json").read_text(encoding="utf-8"))
    checkpoint = output / "trainer" / "checkpoints" / "policy-000001"
    if (summary.get("status") != "updated_checkpoint_committed_refresh_pending"
            or summary.get("run_id") != run_id
            or Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or summary.get("source_manifest_sha256") != source_digest):
        raise ValueError("D5 trainer summary does not identify the intended update")
    manifest_digest = _digest(summary.get("checkpoint_manifest_sha256"), "checkpoint")
    adapter_digest = _digest(summary.get("checkpoint_adapter_sha256"), "adapter")
    parameter_digest = _digest(summary.get("parameter_sha256"), "parameters")
    if checkpoint.is_symlink() or file_sha256(checkpoint / "distributed" / "manifest.json") != manifest_digest:
        raise ValueError("D5 checkpoint digest differs from trainer summary")
    manifest = read_distributed_manifest(checkpoint)
    verify_checkpoint_files(checkpoint, manifest)
    if (manifest.run_kind != "production" or manifest.run_id != run_id
            or manifest.policy_version != "policy-000001" or manifest.update_index != 1
            or manifest.source_rollout_manifest_sha256 != source_digest
            or manifest.config_sha256 != prior.config_sha256
            or manifest.model_id != config.model.id
            or manifest.model_revision != prior.model_revision
            or manifest.tokenizer_sha256 != prior.tokenizer_sha256
            or manifest.grammar_sha256 != prior.grammar_sha256
            or manifest.code_identity != production_code_identity()
            or manifest.trainer_world_size != 4
            or tuple(rank.module_id for rank in manifest.ranks)
            != tuple(str(module) for module in layout.trainer_modules)):
        raise ValueError("D5 checkpoint identity differs from the production run")
    if directory_sha256(checkpoint / "adapter") != adapter_digest:
        raise ValueError("D5 checkpoint adapter digest differs")
    reports = []
    for rank in range(4):
        report = json.loads((output / "ranks" / f"rank-{rank:03d}.json").read_text(encoding="utf-8"))
        update = report.get("update", {})
        if (report.get("status") != "updated_checkpoint_committed_refresh_pending"
                or report.get("rank") != rank
                or report.get("module_id") != str(layout.trainer_modules[rank])
                or report.get("source_manifest_sha256") != source_digest
                or report.get("checkpoint_manifest_sha256") != manifest_digest
                or report.get("checkpoint_adapter_sha256") != adapter_digest
                or update.get("rank") != rank
                or update.get("source_policy_version") != prior.version
                or update.get("next_policy_version") != "policy-000001"
                or update.get("consumed_manifest_sha256") != source_digest
                or update.get("parameter_sha256") != parameter_digest
                or update.get("optimizer_steps") != 1
                or update.get("gradient_sync_phases") != 1):
            raise ValueError(f"D5 trainer rank {rank} evidence differs")
        _digest(update.get("optimizer_sha256"), f"optimizer rank {rank}")
        reports.append(report)
    if len({row["update"]["optimizer_sha256"] for row in reports}) != 1:
        raise ValueError("D5 trainer optimizer replicas differ")
    next_policy = PolicyDescriptor(
        version="policy-000001", update_index=1, adapter_sha256=adapter_digest,
        config_sha256=prior.config_sha256, model_revision=prior.model_revision,
        tokenizer_sha256=prior.tokenizer_sha256,
        grammar_sha256=prior.grammar_sha256, run_kind="production",
    )
    return next_policy, summary, tuple(reports)


def _reconstruct_coordinator(
    *, layout: RoleLayout, prior: PolicyDescriptor,
    config: ExperimentConfig, rollout_root: Path, receipt: BatchReceipt,
    next_policy: PolicyDescriptor, reports: tuple[dict[str, Any], ...],
) -> CycleCoordinator:
    """Reconstruct the lifecycle from completed, identity-checked evidence."""

    coordinator = CycleCoordinator(layout, prior, config.to_dict(), 64)
    for role in ("rollout", "trainer"):
        for rank in range(4):
            coordinator.acknowledge_ready(role, rank, prior)
    coordinator.begin_collection()
    committed = coordinator.commit_batch(rollout_root)
    if committed != receipt:
        raise ValueError("D5 coordinator batch differs from phase receipt")
    for rank, report in enumerate(reports):
        update = report["update"]
        coordinator.acknowledge_verified(
            rank, manifest_sha256=receipt.manifest_sha256,
            match_indices=receipt.match_indices_by_rank[rank],
            turns=int(update["turns"]), owned_tokens=int(update["owned_tokens"]),
            max_replay_error=float(update["max_abs_log_prob_error"]),
        )
    for rank, report in enumerate(reports):
        update = report["update"]
        coordinator.acknowledge_update(
            rank, manifest_sha256=receipt.manifest_sha256,
            next_policy=next_policy, parameter_sha256=update["parameter_sha256"],
            optimizer_sha256=update["optimizer_sha256"],
            optimizer_steps=update["optimizer_steps"],
            gradient_sync_phases=update["gradient_sync_phases"],
            changed_tensors=update["changed_parameter_tensors"],
        )
    if coordinator.phase is not Phase.UPDATED:
        raise RuntimeError("D5 coordinator did not accept all trainer reports")
    return coordinator


def run_initial_cycle(args: Any) -> dict[str, Any]:
    """Run one complete production update and stop after four refresh probes."""

    config = load_config(args.config)
    layout = RoleLayout(_modules(args.rollout_modules), _modules(args.trainer_modules))
    verify_d5_launch_prerequisites(
        config_path=args.config, two_rank_d4_summary=args.d4_two_summary,
        four_rank_d4_summary=args.d4_four_summary, layout=layout,
    )
    if (args.seed < 0 or not math.isfinite(args.replay_tolerance)
            or args.replay_tolerance <= 0 or any(
                timeout <= 0 for timeout in (
                    args.rollout_timeout_seconds, args.trainer_timeout_seconds,
                    args.refresh_timeout_seconds,
                )
            )):
        raise ValueError("D5 seeds, tolerance and phase deadlines must be valid")
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id or args.run_id in {".", ".."}:
        raise ValueError("D5 run ID must be a safe single name")
    if not 1 <= args.trainer_master_port <= 65535:
        raise ValueError("D5 trainer rendezvous port is outside [1, 65535]")
    root = args.output
    if root.exists() or root.is_symlink():
        raise FileExistsError("D5 one-update output must be new")
    root.mkdir(parents=True)
    initial_bundle = root / "initial-policy"
    rollout_root = root / "rollout-000000"
    trainer_output = root / "trainer-000001"
    refresh_output = root / "refresh-000001"
    try:
        prepare_initial_policy(
            pilot_root=args.pilot_root, config=config, output=initial_bundle,
        )
        prior = _initial_policy(
            initial_bundle / "policy_descriptor.json", initial_bundle / "policy_adapter", config,
        )
        prepare_frozen_batch(
            rollout_root, config=config, source_adapter=initial_bundle / "policy_adapter",
            policy=prior, base_seed=args.seed,
            replay_tolerance=args.replay_tolerance,
        )
        rollout_args = [
            "self_play_grpo.training.rollout_worker",
            "--config", str(args.config), "--root", str(rollout_root),
            "--policy-descriptor", str(rollout_root / "policy_descriptor.json"),
        ]
        supervise_role_phase(
            _commands("rollout", layout.rollout_modules, rollout_args),
            layout, root / "logs-rollout-000000", role="rollout",
            trainer_master_port=args.trainer_master_port,
            timeout_seconds=args.rollout_timeout_seconds,
        )
        aggregate_rank_shards(
            rollout_root, config=config, policy=prior,
            expected_modules=layout.rollout_modules,
        )
        receipt = verify_completed_batch(
            rollout_root, config=config.to_dict(), policy=prior,
            expected_games=64,
        )
        trainer_args = [
            "self_play_grpo.training.trainer_worker",
            "--config", str(args.config), "--rollout-root", str(rollout_root),
            "--policy-descriptor", str(rollout_root / "policy_descriptor.json"),
            "--expected-manifest-sha256", receipt.manifest_sha256,
            "--output", str(trainer_output), "--run-id", args.run_id,
            "--rollout-modules", args.rollout_modules,
            "--trainer-modules", args.trainer_modules,
            "--d4-two-summary", str(args.d4_two_summary),
            "--d4-four-summary", str(args.d4_four_summary),
            "--timeout-seconds", str(args.trainer_timeout_seconds),
        ]
        supervise_role_phase(
            _commands("trainer", layout.trainer_modules, trainer_args),
            layout, root / "logs-trainer-000001", role="trainer",
            trainer_master_port=args.trainer_master_port,
            timeout_seconds=args.trainer_timeout_seconds,
        )
        next_policy, trainer_summary, trainer_reports = _trainer_evidence(
            trainer_output, source_digest=receipt.manifest_sha256,
            config=config, prior=prior, layout=layout, run_id=args.run_id,
        )
        coordinator = _reconstruct_coordinator(
            layout=layout, prior=prior, config=config, rollout_root=rollout_root,
            receipt=receipt, next_policy=next_policy, reports=trainer_reports,
        )
        record = publish_cycle_checkpoint(
            coordinator, run_root=root,
            checkpoint_path="trainer-000001/trainer/checkpoints/policy-000001",
            run_id=args.run_id,
        )
        refresh_args = [
            "self_play_grpo.training.policy_refresh",
            "--config", str(args.config), "--rollout-root", str(rollout_root),
            "--trainer-output", str(trainer_output),
            "--output", str(refresh_output),
            "--rollout-modules", args.rollout_modules,
            "--trainer-modules", args.trainer_modules,
            "--d4-two-summary", str(args.d4_two_summary),
            "--d4-four-summary", str(args.d4_four_summary),
        ]
        supervise_role_phase(
            _commands("rollout", layout.rollout_modules, refresh_args),
            layout, root / "logs-refresh-000001", role="rollout",
            trainer_master_port=args.trainer_master_port,
            timeout_seconds=args.refresh_timeout_seconds,
        )
        refreshed_policy, refresh_evidence = aggregate_refresh_evidence(
            rollout_root=rollout_root, trainer_output=trainer_output,
            refresh_output=refresh_output, config=config, layout=layout,
        )
        if refreshed_policy != next_policy:
            raise ValueError("D5 refreshed policy differs from committed checkpoint")
        for rank in range(4):
            row = json.loads((refresh_output / f"rank-{rank:03d}.json").read_text(encoding="utf-8"))
            acknowledge_refresh_with_identity(
                coordinator, rank, policy=next_policy,
                loaded_parameter_sha256=row["parameter_sha256"],
                probe_error=row["probe_max_abs_error"],
                probe_tolerance=args.replay_tolerance,
            )
        if coordinator.phase is not Phase.READY or coordinator.policy != next_policy:
            raise RuntimeError("D5 coordinator did not reach the next ready policy")
        publish_refresh_evidence(refresh_output, refresh_evidence)
        if read_commits(root) != (record,):
            raise RuntimeError("D5 production ledger changed after refresh")
        result = {
            "status": "one_update_complete", "run_id": args.run_id,
            "games": 64, "policy_version": next_policy.version,
            "source_manifest_sha256": receipt.manifest_sha256,
            "checkpoint_manifest_sha256": record.checkpoint_manifest_sha256,
            "checkpoint": record.checkpoint_path,
            "refresh_ranks": 4,
            "max_abs_probe_error": refresh_evidence["max_abs_probe_error"],
            "trainer_metrics": trainer_summary["metrics"],
            "coordinator": coordinator.snapshot(),
        }
        with (root / "cycle_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return result
    except BaseException as exc:
        try:
            ledger_records = len(read_commits(root))
        except Exception:
            ledger_records = None
        failure = {
            "status": "failed", "run_id": args.run_id,
            "error_type": type(exc).__name__, "error": str(exc),
            "ledger_records": ledger_records,
        }
        (root / "cycle_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D5 guarded first 4+4 production update")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rollout-modules", required=True)
    parser.add_argument("--trainer-modules", required=True)
    parser.add_argument("--d4-two-summary", required=True, type=Path)
    parser.add_argument("--d4-four-summary", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--replay-tolerance", type=float, default=2e-4)
    parser.add_argument("--trainer-master-port", type=int, required=True)
    parser.add_argument("--rollout-timeout-seconds", type=int, default=7200)
    parser.add_argument("--trainer-timeout-seconds", type=int, default=10800)
    parser.add_argument("--refresh-timeout-seconds", type=int, default=1800)
    print(json.dumps(run_initial_cycle(parser.parse_args(argv)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
