"""Opt-in, fail-closed update-2 continuation on the existing 4+4 Gaudi run.

This is one complete *resumed* update, not the five-update D6 driver. It does
not start workers on import. The caller must own eight assigned modules and
provide the committed, refresh-complete D5 run root. Partial update-2 output
is preserved on failure; it is never silently retrained or overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from self_play_grpo.config import ExperimentConfig, load_config
from self_play_grpo.rollouts.pilot import (
    canonical_sha256, directory_sha256, file_sha256, read_pilot_manifest,
)
from self_play_grpo.training.coordinator import (
    Phase, PolicyDescriptor, RoleLayout, verify_completed_batch,
)
from self_play_grpo.training.d5_preflight import verify_d5_launch_prerequisites
from self_play_grpo.training.d6_code_identity import d6_code_identity
from self_play_grpo.training.d6_collection import (
    descriptor_from_commit, prepare_next_collection,
)
from self_play_grpo.training.d6_recovery import audit_recovery
from self_play_grpo.training.d6_refresh_worker import aggregate_refresh
from self_play_grpo.training.distributed_checkpoint import (
    read_distributed_manifest, verify_checkpoint_files,
)
from self_play_grpo.training.handoff import (
    acknowledge_refresh_with_identity, publish_cycle_checkpoint,
)
from self_play_grpo.training.initial_cycle import (
    _commands, _digest, _modules, _reconstruct_coordinator,
)
from self_play_grpo.training.ledger import read_commits
from self_play_grpo.training.phase_supervisor import supervise_role_phase
from self_play_grpo.training.refresh_gate import publish_refresh_evidence
from self_play_grpo.training.rollout_worker import (
    _validate_prepared, aggregate_rank_shards,
)


def _trainer_evidence(
    output: Path, *, run_root: Path, rollout_root: Path,
    source_digest: str, config: ExperimentConfig,
    prior: PolicyDescriptor, layout: RoleLayout, run_id: str,
) -> tuple[PolicyDescriptor, dict[str, Any], tuple[dict[str, Any], ...]]:
    """Validate the update-2 checkpoint and four synchronized rank reports."""

    summary_path = output / "trainer_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ValueError("D6 trainer summary is missing or symlinked")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = output / "trainer" / "checkpoints" / "policy-000002"
    if (summary.get("status") != "updated_checkpoint_committed_refresh_pending"
            or summary.get("run_id") != run_id
            or Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or summary.get("source_manifest_sha256") != source_digest
            or summary.get("source_policy_version") != prior.version):
        raise ValueError("D6 trainer summary does not identify the intended update")
    checkpoint_digest = _digest(summary.get("checkpoint_manifest_sha256"), "checkpoint")
    adapter_digest = _digest(summary.get("checkpoint_adapter_sha256"), "adapter")
    parameter_digest = _digest(summary.get("parameter_sha256"), "parameters")
    if (checkpoint.is_symlink()
            or file_sha256(checkpoint / "distributed" / "manifest.json") != checkpoint_digest):
        raise ValueError("D6 checkpoint digest differs from trainer summary")
    manifest = read_distributed_manifest(checkpoint)
    verify_checkpoint_files(checkpoint, manifest)
    if (manifest.run_kind != "production" or manifest.run_id != run_id
            or manifest.policy_version != "policy-000002" or manifest.update_index != 2
            or manifest.source_rollout_manifest_sha256 != source_digest
            or manifest.config_sha256 != prior.config_sha256
            or manifest.model_id != config.model.id
            or manifest.model_revision != prior.model_revision
            or manifest.tokenizer_sha256 != prior.tokenizer_sha256
            or manifest.grammar_sha256 != prior.grammar_sha256
            or manifest.code_identity != d6_code_identity()
            or manifest.trainer_world_size != 4
            or tuple(item.module_id for item in manifest.ranks)
            != tuple(str(module) for module in layout.trainer_modules)):
        raise ValueError("D6 checkpoint identity differs from the active production run")
    if directory_sha256(checkpoint / "adapter") != adapter_digest:
        raise ValueError("D6 checkpoint adapter digest differs")
    previous = read_commits(run_root)
    if len(previous) != 1 or previous[0].policy_version != prior.version:
        raise ValueError("D6 resumed trainer source is not the committed ledger head")
    if source_digest == previous[0].source_manifest_sha256:
        raise ValueError("D6 resumed trainer reused the consumed batch")
    reports = []
    for rank in range(4):
        row = json.loads((output / "ranks" / f"rank-{rank:03d}.json").read_text(encoding="utf-8"))
        update = row.get("update", {})
        restored = row.get("restored", {})
        if (row.get("status") != "updated_checkpoint_committed_refresh_pending"
                or row.get("rank") != rank
                or row.get("module_id") != str(layout.trainer_modules[rank])
                or row.get("source_manifest_sha256") != source_digest
                or row.get("source_policy_version") != prior.version
                or row.get("checkpoint_manifest_sha256") != checkpoint_digest
                or row.get("checkpoint_adapter_sha256") != adapter_digest
                or restored.get("status") != "restored_no_optimizer_step"
                or restored.get("policy_version") != prior.version
                or restored.get("checkpoint_manifest_sha256")
                != previous[0].checkpoint_manifest_sha256
                or update.get("rank") != rank
                or update.get("source_policy_version") != prior.version
                or update.get("next_policy_version") != "policy-000002"
                or update.get("consumed_manifest_sha256") != source_digest
                or update.get("parameter_sha256") != parameter_digest
                or update.get("optimizer_steps") != 2
                or update.get("gradient_sync_phases") != 1):
            raise ValueError(f"D6 trainer rank {rank} evidence differs")
        _digest(update.get("optimizer_sha256"), f"optimizer rank {rank}")
        reports.append(row)
    if len({row["update"]["optimizer_sha256"] for row in reports}) != 1:
        raise ValueError("D6 trainer optimizer replicas differ")
    next_policy = PolicyDescriptor(
        version="policy-000002", update_index=2,
        adapter_sha256=adapter_digest, config_sha256=prior.config_sha256,
        model_revision=prior.model_revision,
        tokenizer_sha256=prior.tokenizer_sha256,
        grammar_sha256=prior.grammar_sha256, run_kind="production",
    )
    return next_policy, summary, tuple(reports)


def run_second_cycle(args: Any) -> dict[str, Any]:
    """Collect a fresh batch, restore four trainers, update, commit, refresh."""

    config = load_config(args.config)
    layout = RoleLayout(_modules(args.rollout_modules), _modules(args.trainer_modules))
    verify_d5_launch_prerequisites(
        config_path=args.config, two_rank_d4_summary=args.d4_two_summary,
        four_rank_d4_summary=args.d4_four_summary, layout=layout,
    )
    if (type(args.seed) is not int or args.seed < 0
            or not math.isfinite(args.replay_tolerance) or args.replay_tolerance <= 0
            or any(value <= 0 for value in (
                args.rollout_timeout_seconds, args.trainer_timeout_seconds,
                args.refresh_timeout_seconds,
            ))):
        raise ValueError("D6 seed, tolerance or phase deadline is invalid")
    if not 1 <= args.trainer_master_port <= 65535:
        raise ValueError("D6 trainer master port is outside [1, 65535]")
    root = args.run_root
    plan = audit_recovery(
        root, config_path=args.config, experiment_seed=args.seed,
        expected_run_id=args.run_id,
    )
    if plan.status != "ready_for_next_collection" or plan.committed_update != 1:
        raise RuntimeError("D6 update-2 driver requires refresh-complete update 1")
    initial = read_pilot_manifest(root / "rollout-000000")
    if initial.base_seed != args.seed:
        raise ValueError("D6 experiment seed differs from the initial collection")
    rollout_root = root / "rollout-000001"
    trainer_output = root / "trainer-000002"
    refresh_output = root / "refresh-000002"
    if trainer_output.exists() or refresh_output.exists():
        raise FileExistsError("D6 update-2 output already exists; inspect recovery state before retry")
    prior = descriptor_from_commit(root, plan, config=config)
    try:
        reuse_complete_batch = False
        if rollout_root.exists():
            manifest = read_pilot_manifest(rollout_root)
            if (manifest.base_seed != plan.next_base_seed
                    or manifest.replay_tolerance != args.replay_tolerance):
                raise ValueError("D6 existing rollout seed or tolerance differs")
            if len(manifest.matches) == 64:
                recorded = PolicyDescriptor(**json.loads(
                    (rollout_root / "policy_descriptor.json").read_text(encoding="utf-8")
                ))
                if (recorded != prior or
                        directory_sha256(rollout_root / "policy_adapter") != prior.adapter_sha256):
                    raise ValueError("D6 complete rollout differs from the committed policy")
                reuse_complete_batch = True
            elif not manifest.matches:
                _validate_prepared(rollout_root, config, prior)
                if (any((rollout_root / "ranks").glob("*")) or
                        any((rollout_root / "matches").glob("*"))):
                    raise FileExistsError("D6 partial rollout exists; no automatic overwrite")
            else:
                raise FileExistsError("D6 partial rollout manifest exists; no automatic recollection")
        else:
            prepare_next_collection(
                root, config_path=args.config, experiment_seed=args.seed,
                expected_run_id=args.run_id, replay_tolerance=args.replay_tolerance,
            )
        if not reuse_complete_batch:
            rollout_args = [
                "self_play_grpo.training.rollout_worker",
                "--config", str(args.config), "--root", str(rollout_root),
                "--policy-descriptor", str(rollout_root / "policy_descriptor.json"),
            ]
            supervise_role_phase(
                _commands("rollout", layout.rollout_modules, rollout_args),
                layout, root / "logs-rollout-000001", role="rollout",
                trainer_master_port=args.trainer_master_port,
                timeout_seconds=args.rollout_timeout_seconds,
            )
            aggregate_rank_shards(
                rollout_root, config=config, policy=prior,
                expected_modules=layout.rollout_modules,
            )
        receipt = verify_completed_batch(
            rollout_root, config=config.to_dict(), policy=prior, expected_games=64,
        )
        if receipt.manifest_sha256 in plan.consumed_batches:
            raise ValueError("D6 rollout batch was already consumed")
        trainer_args = [
            "self_play_grpo.training.d6_trainer_worker",
            "--config", str(args.config), "--run-root", str(root),
            "--rollout-root", str(rollout_root),
            "--expected-manifest-sha256", receipt.manifest_sha256,
            "--output", str(trainer_output), "--run-id", args.run_id,
            "--experiment-seed", str(args.seed),
            "--rollout-modules", args.rollout_modules,
            "--trainer-modules", args.trainer_modules,
            "--d4-two-summary", str(args.d4_two_summary),
            "--d4-four-summary", str(args.d4_four_summary),
            "--timeout-seconds", str(args.trainer_timeout_seconds),
        ]
        supervise_role_phase(
            _commands("trainer", layout.trainer_modules, trainer_args),
            layout, root / "logs-trainer-000002", role="trainer",
            trainer_master_port=args.trainer_master_port,
            timeout_seconds=args.trainer_timeout_seconds,
        )
        next_policy, trainer_summary, trainer_reports = _trainer_evidence(
            trainer_output, run_root=root, rollout_root=rollout_root,
            source_digest=receipt.manifest_sha256, config=config,
            prior=prior, layout=layout, run_id=args.run_id,
        )
        coordinator = _reconstruct_coordinator(
            layout=layout, prior=prior, config=config, rollout_root=rollout_root,
            receipt=receipt, next_policy=next_policy, reports=trainer_reports,
        )
        record = publish_cycle_checkpoint(
            coordinator, run_root=root,
            checkpoint_path="trainer-000002/trainer/checkpoints/policy-000002",
            run_id=args.run_id,
        )
        refresh_args = [
            "self_play_grpo.training.d6_refresh_worker",
            "--config", str(args.config), "--rollout-root", str(rollout_root),
            "--trainer-output", str(trainer_output),
            "--output", str(refresh_output), "--run-id", args.run_id,
            "--rollout-modules", args.rollout_modules,
            "--trainer-modules", args.trainer_modules,
            "--d4-two-summary", str(args.d4_two_summary),
            "--d4-four-summary", str(args.d4_four_summary),
        ]
        supervise_role_phase(
            _commands("rollout", layout.rollout_modules, refresh_args),
            layout, root / "logs-refresh-000002", role="rollout",
            trainer_master_port=args.trainer_master_port,
            timeout_seconds=args.refresh_timeout_seconds,
        )
        refreshed_policy, refresh_evidence = aggregate_refresh(
            rollout_root=rollout_root, trainer_output=trainer_output,
            refresh_output=refresh_output, config=config,
            layout=layout, run_id=args.run_id,
        )
        if refreshed_policy != next_policy:
            raise ValueError("D6 refreshed policy differs from committed checkpoint")
        for rank in range(4):
            row = json.loads((refresh_output / f"rank-{rank:03d}.json").read_text(encoding="utf-8"))
            acknowledge_refresh_with_identity(
                coordinator, rank, policy=next_policy,
                loaded_parameter_sha256=row["parameter_sha256"],
                probe_error=row["probe_max_abs_error"],
                probe_tolerance=args.replay_tolerance,
            )
        if coordinator.phase is not Phase.READY or coordinator.policy != next_policy:
            raise RuntimeError("D6 coordinator did not reach the next ready policy")
        publish_refresh_evidence(refresh_output, refresh_evidence)
        commits = read_commits(root)
        if (len(commits) != 2 or commits[-1] != record or
                commits[0].checkpoint_manifest_sha256 != plan.checkpoint_manifest_sha256):
            raise RuntimeError("D6 ledger changed after refresh")
        result = {
            "status": "second_update_complete", "run_id": args.run_id,
            "games": 64, "policy_version": next_policy.version,
            "source_manifest_sha256": receipt.manifest_sha256,
            "checkpoint_manifest_sha256": record.checkpoint_manifest_sha256,
            "checkpoint": record.checkpoint_path,
            "refresh_ranks": 4,
            "max_abs_probe_error": refresh_evidence["max_abs_probe_error"],
            "trainer_metrics": trainer_summary["metrics"],
            "coordinator": coordinator.snapshot(),
        }
        with (root / "cycle_summary_000002.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return result
    except BaseException as exc:
        try:
            ledger_count = len(read_commits(root))
        except Exception:
            ledger_count = None
        failure = {
            "status": "failed", "run_id": args.run_id,
            "error_type": type(exc).__name__, "error": str(exc),
            "ledger_records": ledger_count,
        }
        (root / "cycle_failure_000002.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D6 guarded update-2 continuation on 4+4 HPUs")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
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
    print(json.dumps(run_second_cycle(parser.parse_args(argv)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
