"""Bounded synchronous 4+4 training with recovery at committed updates.

Run through an absolute update number with --until-update. Reissuing the same
command recovers interrupted work. No worker or accelerator starts on import.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, read_pilot_manifest
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout, verify_completed_batch
from self_play_grpo.training.d5_preflight import verify_d5_launch_prerequisites
from self_play_grpo.training.d6_collection import descriptor_from_commit
from self_play_grpo.training.d6_recovery import _json_file
from self_play_grpo.training.distributed_checkpoint import read_distributed_manifest
from self_play_grpo.training.handoff import publish_cycle_checkpoint
from self_play_grpo.training.initial_cycle import _commands, _modules, _reconstruct_coordinator
from self_play_grpo.training.multi_evidence import finalize_checkpoint, publish_json
from self_play_grpo.training.multi_identity import multi_code_identity
from self_play_grpo.training.multi_recovery import audit_recovery
from self_play_grpo.training.multi_refresh import aggregate_refresh
from self_play_grpo.training.multi_supervisor import supervise_role_phase
from self_play_grpo.training.refresh_gate import publish_refresh_evidence
from self_play_grpo.training.rollout_worker import prepare_frozen_batch, aggregate_rank_shards


def event(name: str, **values: Any) -> None:
    print(json.dumps({"event": name, **values}, sort_keys=True, allow_nan=False), flush=True)


@contextmanager
def run_lock(root: Path):
    """Workers inherit this descriptor, so parent death cannot permit a rival run."""
    path = root / ".multi_run.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This run still has a live coordinator or worker; wait for its exit") from exc
        yield fd
    finally:
        # Do not explicitly LOCK_UN: live children must retain the lock.
        os.close(fd)


def archive(root: Path, path: Path) -> None:
    """Move only an explicitly selected interrupted artifact; preserve evidence."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.parent.resolve() not in (root.resolve(), (root / "commits").resolve()):
        raise ValueError(f"Unsafe recovery artifact: {path}")
    target = root / "recovery_archive"
    if target.is_symlink():
        raise ValueError("Symlinked recovery archive")
    target.mkdir(exist_ok=True)
    destination = target / f"{path.name}-{uuid.uuid4().hex}"
    path.rename(destination)
    event("archived_interrupted_artifact", source=str(path), archive=str(destination))


def clean_pending_ledger(root: Path) -> None:
    """A killed publisher can leave its unpublished hard-link staging file."""
    directory = root / "commits"
    if directory.is_symlink():
        raise ValueError("Symlinked commit directory")
    if directory.exists():
        for path in directory.iterdir():
            if re.fullmatch(r"\.update-\d{6}\.json\.\d+\.pending", path.name):
                archive(root, path)


def launch(args: Any, layout: RoleLayout, lock_fd: int, role: str,
           phase: str, update: int, worker_args: list[str], timeout: int) -> None:
    logs = args.run_root / f"logs-multi-{update:06d}-{phase}-{uuid.uuid4().hex[:12]}"
    modules = layout.rollout_modules if role == "rollout" else layout.trainer_modules
    event("phase_started", update=update, phase=phase, logs=str(logs))
    start = time.monotonic()
    supervise_role_phase(
        _commands(role, modules, worker_args), layout, logs, role=role,
        trainer_master_port=args.trainer_master_port, timeout_seconds=timeout, lock_fd=lock_fd,
    )
    duration = time.monotonic() - start
    publish_json(logs / "timing.json", {"update": update, "phase": phase, "seconds": duration})
    event("phase_complete", update=update, phase=phase, seconds=duration)


def common_worker_args(args: Any) -> list[str]:
    return ["--config", str(args.config), "--run-id", args.run_id,
            "--rollout-modules", args.rollout_modules, "--trainer-modules", args.trainer_modules,
            "--d4-two-summary", str(args.d4_two_summary), "--d4-four-summary", str(args.d4_four_summary)]


def collect_next(args: Any, config: Any, layout: RoleLayout, plan: Any,
                 lock_fd: int) -> tuple[Path, PolicyDescriptor, Any]:
    prior = descriptor_from_commit(args.run_root, plan, config=config)
    output = args.run_root / f"rollout-{plan.next_collection_index:06d}"
    if output.is_symlink():
        raise ValueError("Symlinked rollout output")
    if output.exists():
        manifest = read_pilot_manifest(output)
        if (manifest.base_seed != plan.next_base_seed or manifest.replay_tolerance != args.replay_tolerance
                or manifest.policy_version != prior.version or manifest.adapter_sha256 != prior.adapter_sha256
                or manifest.config_sha256 != prior.config_sha256
                or PolicyDescriptor(**_json_file(output / "policy_descriptor.json")) != prior
                or directory_sha256(output / "policy_adapter") != prior.adapter_sha256):
            raise ValueError("Existing collection differs from the committed policy or seed")
        if len(manifest.matches) == 64:
            receipt = verify_completed_batch(output, config=config.to_dict(), policy=prior, expected_games=64)
            if receipt.manifest_sha256 in plan.consumed_batches:
                raise ValueError("Collection was already consumed")
            event("reusing_complete_batch", update=plan.next_update, batch_sha256=receipt.manifest_sha256)
            return output, prior, receipt
        # The last manifest is unpublished. Recollect from the same frozen
        # policy and seed in a new directory; retain all interrupted shards.
        archive(args.run_root, output)
    prepare_frozen_batch(
        output, config=config, source_adapter=args.run_root / plan.checkpoint / "adapter",
        policy=prior, base_seed=plan.next_base_seed, replay_tolerance=args.replay_tolerance,
    )
    launch(args, layout, lock_fd, "rollout", "collection", plan.next_update, [
        "self_play_grpo.training.rollout_worker", "--config", str(args.config),
        "--root", str(output), "--policy-descriptor", str(output / "policy_descriptor.json"),
    ], args.rollout_timeout_seconds)
    aggregate_rank_shards(output, config=config, policy=prior, expected_modules=layout.rollout_modules)
    receipt = verify_completed_batch(output, config=config.to_dict(), policy=prior, expected_games=64)
    if receipt.manifest_sha256 in plan.consumed_batches:
        raise ValueError("Collection was already consumed")
    return output, prior, receipt


def update_next(args: Any, config: Any, layout: RoleLayout, plan: Any, lock_fd: int,
                rollout: Path, prior: PolicyDescriptor, receipt: Any) -> Any:
    update = plan.next_update
    output = args.run_root / f"trainer-{update:06d}"
    checkpoint = output / "trainer" / "checkpoints" / f"policy-{update:06d}"
    if output.is_symlink():
        raise ValueError("Symlinked trainer output")
    if not checkpoint.exists() and not checkpoint.is_symlink():
        if output.exists():
            if list((output / "trainer" / "checkpoints").glob("policy-*")):
                raise ValueError("Unexpected published checkpoint; inspect before retry")
            archive(args.run_root, output)
        launch(args, layout, lock_fd, "trainer", "training", update, [
            "self_play_grpo.training.multi_trainer", *common_worker_args(args),
            "--run-root", str(args.run_root), "--rollout-root", str(rollout),
            "--expected-manifest-sha256", receipt.manifest_sha256, "--output", str(output),
            "--experiment-seed", str(args.seed), "--timeout-seconds", str(args.trainer_timeout_seconds),
        ], args.trainer_timeout_seconds)
    else:
        event("recovering_published_checkpoint", update=update, checkpoint=str(checkpoint))
    summary = finalize_checkpoint(root=args.run_root, output=output, rollout_root=rollout,
                                  config=config, layout=layout, run_id=args.run_id, update=update)
    next_policy = PolicyDescriptor(
        version=f"policy-{update:06d}", update_index=update,
        adapter_sha256=summary["checkpoint_adapter_sha256"], config_sha256=prior.config_sha256,
        model_revision=prior.model_revision, tokenizer_sha256=prior.tokenizer_sha256,
        grammar_sha256=prior.grammar_sha256, run_kind="production",
    )
    reports = tuple(_json_file(output / "ranks" / f"rank-{rank:03d}.json") for rank in range(4))
    coordinator = _reconstruct_coordinator(layout=layout, prior=prior, config=config, rollout_root=rollout,
                                         receipt=receipt, next_policy=next_policy, reports=reports)
    commit = publish_cycle_checkpoint(coordinator, run_root=args.run_root,
                                     checkpoint_path=str(checkpoint.relative_to(args.run_root)), run_id=args.run_id)
    event("checkpoint_committed", update=update, policy_version=next_policy.version,
          checkpoint_manifest_sha256=commit.checkpoint_manifest_sha256, trainer_metrics=summary["metrics"])
    return commit


def finish_refresh(args: Any, config: Any, layout: RoleLayout, plan: Any, lock_fd: int) -> None:
    update = plan.committed_update
    rollout = args.run_root / f"rollout-{update - 1:06d}"
    trainer = args.run_root / f"trainer-{update:06d}"
    output = args.run_root / f"refresh-{update:06d}"
    if update >= 3:
        finalize_checkpoint(root=args.run_root, output=trainer, rollout_root=rollout,
                            config=config, layout=layout, run_id=args.run_id, update=update)
    if output.is_symlink():
        raise ValueError("Symlinked refresh output")
    if output.exists():
        archive(args.run_root, output)
    launch(args, layout, lock_fd, "rollout", "refresh", update, [
        "self_play_grpo.training.multi_refresh", *common_worker_args(args),
        "--rollout-root", str(rollout), "--trainer-output", str(trainer), "--output", str(output),
    ], args.refresh_timeout_seconds)
    policy, evidence = aggregate_refresh(rollout_root=rollout, trainer_output=trainer,
                                        refresh_output=output, config=config, layout=layout, run_id=args.run_id)
    if policy.version != plan.policy_version:
        raise ValueError("Refreshed policy differs from ledger head")
    publish_refresh_evidence(output, evidence)
    summary = _json_file(trainer / "trainer_summary.json")
    publish_json(args.run_root / f"multi_cycle_summary_{update:06d}.json", {
        "status": "update_complete", "update": update, "policy_version": policy.version,
        "checkpoint": plan.checkpoint, "checkpoint_manifest_sha256": plan.checkpoint_manifest_sha256,
        "source_manifest_sha256": plan.source_manifest_sha256, "refresh_ranks": 4,
        "max_abs_probe_error": evidence["max_abs_probe_error"], "trainer_metrics": summary["metrics"],
    })
    event("update_complete", update=update, policy_version=policy.version)


def run(args: Any) -> dict[str, Any]:
    if args.run_root.is_symlink() or not args.run_root.is_dir():
        raise ValueError("Run root must be an existing real directory")
    args.run_root = args.run_root.resolve()
    args.config = args.config.resolve()
    config = load_config(args.config)
    if args.audit:
        return asdict(audit_recovery(args.run_root, config_path=args.config,
                                    experiment_seed=args.seed, expected_run_id=args.run_id))
    if (type(args.until_update) is not int or not 3 <= args.until_update <= 999999
            or not math.isfinite(args.replay_tolerance) or not 0 < args.replay_tolerance <= 2e-4
            or not 1 <= args.trainer_master_port <= 65535
            or min(args.rollout_timeout_seconds, args.trainer_timeout_seconds, args.refresh_timeout_seconds) <= 0):
        raise ValueError("Invalid target update, replay tolerance, port or phase timeout")
    layout = RoleLayout(_modules(args.rollout_modules), _modules(args.trainer_modules))
    verify_d5_launch_prerequisites(config_path=args.config, two_rank_d4_summary=args.d4_two_summary,
                                  four_rank_d4_summary=args.d4_four_summary, layout=layout)
    with run_lock(args.run_root) as lock_fd:
        clean_pending_ledger(args.run_root)
        plan = audit_recovery(args.run_root, config_path=args.config, experiment_seed=args.seed,
                              expected_run_id=args.run_id)
        if plan.committed_update < 2:
            raise ValueError("Complete update 2 before using this driver")
        manifest = read_distributed_manifest(args.run_root / plan.checkpoint)
        if tuple(row.module_id for row in manifest.ranks) != tuple(map(str, layout.trainer_modules)):
            raise ValueError("Trainer topology differs from committed checkpoint")
        publish_json(args.run_root / "multi_run_contract.json", {
            "run_id": args.run_id, "seed": args.seed, "config_sha256": canonical_sha256(config.to_dict()),
            "code_identity": multi_code_identity(), "rollout_modules": list(layout.rollout_modules),
            "trainer_modules": list(layout.trainer_modules), "replay_tolerance": args.replay_tolerance,
        })
        while True:
            if plan.status == "refresh_required":
                finish_refresh(args, config, layout, plan, lock_fd)
                plan = audit_recovery(args.run_root, config_path=args.config, experiment_seed=args.seed,
                                      expected_run_id=args.run_id)
            if plan.status != "ready_for_next_collection":
                raise RuntimeError("Policy refresh has not completed")
            if plan.committed_update >= args.until_update:
                return {"status": "target_reached", "committed_update": plan.committed_update,
                        "policy_version": plan.policy_version, "target_update": args.until_update,
                        "checkpoint": plan.checkpoint, "logical_training_games": plan.committed_update * 64}
            event("update_started", update=plan.next_update, source_policy=plan.policy_version,
                  collection_seed=plan.next_base_seed)
            rollout, prior, receipt = collect_next(args, config, layout, plan, lock_fd)
            update_next(args, config, layout, plan, lock_fd, rollout, prior, receipt)
            plan = audit_recovery(args.run_root, config_path=args.config, experiment_seed=args.seed,
                                  expected_run_id=args.run_id)


def main(argv: list[str] | None = None) -> int:
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--until-update", type=int, default=5, help="Absolute target update, not additional steps")
    parser.add_argument("--audit", action="store_true", help="Read-only CPU audit; starts no workers")
    parser.add_argument("--rollout-modules", default="0,1,2,3")
    parser.add_argument("--trainer-modules", default="4,5,6,7")
    parser.add_argument("--d4-two-summary", type=Path)
    parser.add_argument("--d4-four-summary", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--replay-tolerance", type=float, default=2e-4)
    parser.add_argument("--trainer-master-port", type=int, default=29642)
    parser.add_argument("--rollout-timeout-seconds", type=int, default=7200)
    parser.add_argument("--trainer-timeout-seconds", type=int, default=10800)
    parser.add_argument("--refresh-timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if not args.audit and (args.d4_two_summary is None or args.d4_four_summary is None):
        parser.error("Training requires both D4 summary paths")
    try:
        print(json.dumps(run(args), sort_keys=True, allow_nan=False), flush=True)
    except BaseException as exc:
        event("run_stopped", error_type=type(exc).__name__, error=str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
