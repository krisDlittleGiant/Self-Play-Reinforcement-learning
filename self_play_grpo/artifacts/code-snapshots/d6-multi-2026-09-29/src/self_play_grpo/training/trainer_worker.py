"""Guarded production one-update trainer rank for the D5 4+4 topology.

This module is opt-in. It never starts HPUs at import time. Four invocations
must run under the supervisor's explicit trainer rank/module environments.
Rank zero publishes one format-3 checkpoint after all ranks finish the same
64-game outcome update; rollout refresh and the cycle ledger happen later.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from self_play_grpo.config import load_config
from self_play_grpo.distributed import (
    discover_torchrun_runtime, initialize_hccl_process_group,
    validate_initialized_hccl,
)
from self_play_grpo.policies.llm import ConstrainedLLMPolicy
from self_play_grpo.rollouts.pilot import (
    directory_sha256, file_sha256, restore_initial_adapter,
)
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout
from self_play_grpo.training.d5_preflight import verify_d5_launch_prerequisites
from self_play_grpo.training.distributed import global_source_max_abs_difference, write_rank_update_report
from self_play_grpo.training.distributed_checkpoint import (
    read_distributed_manifest, validate_resume_identity,
    verify_checkpoint_files, write_rank_rng_record,
)
from self_play_grpo.training.distributed_resume_gate import (
    _expected_identity, _runtime_identity,
)
from self_play_grpo.training.loop import SynchronousTrainer, UpdateMetrics
from self_play_grpo.training.policy_refresh import policy_probe
from self_play_grpo.training.production_checkpoint import build_production_checkpoint_template
from self_play_grpo.training.trainer_handoff import admit_trainer_shard
from self_play_grpo.training.trainer_update import (
    SynchronizedUpdateResult, _collective_phase, synchronized_trainer_update,
)


def _modules(raw: str) -> tuple[int, int, int, int]:
    parts = raw.split(",")
    if len(parts) != 4:
        raise ValueError("D5 worker module list must contain four IDs")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("D5 worker module IDs must be integers") from exc
    if len(set(values)) != 4 or any(value < 0 for value in values):
        raise ValueError("D5 worker modules must be distinct and non-negative")
    return values  # type: ignore[return-value]


def _binding(rank: int, trainer_modules: tuple[int, int, int, int]) -> None:
    if rank not in range(4):
        raise ValueError("D5 trainer rank must be in [0, 4)")
    expected = str(trainer_modules[rank])
    if (os.environ.get("SP_GRPO_ROLE") != "trainer"
            or os.environ.get("SP_GRPO_RANK") != str(rank)
            or os.environ.get("SP_GRPO_MODULE_ID") != expected
            or os.environ.get("HLS_MODULE_ID") != expected
            or os.environ.get("HABANA_VISIBLE_MODULES") != ",".join(map(str, trainer_modules))):
        raise RuntimeError("D5 trainer rank/physical-module binding differs")


def _global_metrics(
    result: SynchronizedUpdateResult, *, trainer: SynchronousTrainer,
    torch: Any, dist: Any, device: Any,
) -> UpdateMetrics:
    """Reduce the exact local diagnostics across all four admitted shards."""

    values = torch.tensor([
        result.loss, result.policy_loss, result.kl_loss,
        result.mean_ratio * result.owned_tokens,
        result.clip_fraction * result.owned_tokens,
        float(result.owned_tokens), float(result.turns), 16.0,
    ], dtype=torch.float32, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total = values.detach().cpu().tolist()
    error = torch.tensor([result.max_abs_log_prob_error], dtype=torch.float32, device=device)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    if (not all(math.isfinite(float(value)) for value in total)
            or total[5] <= 0 or int(round(total[7])) != 64):
        raise RuntimeError("D5 global metrics are non-finite or omit games/tokens")
    return UpdateMetrics(
        update=trainer.update_index, policy_version=trainer.policy_version,
        games=64, turns=int(round(total[6])), owned_tokens=int(round(total[5])),
        loss=float(total[0] / 4), policy_loss=float(total[1] / 4),
        kl_loss=float(total[2] / 4),
        mean_ratio_before_step=float(total[3] / total[5]),
        clip_fraction_before_step=float(total[4] / total[5]),
        replay_max_abs_error=float(error.cpu().item()),
        grad_norm=result.grad_norm, optimizer_steps=1,
    )


def run_initial_trainer_rank(args: Any) -> dict[str, Any]:
    """Execute exactly one initial-policy update, ending before rollout refresh."""

    config = load_config(args.config)
    rollout_modules = _modules(args.rollout_modules)
    trainer_modules = _modules(args.trainer_modules)
    layout = RoleLayout(rollout_modules, trainer_modules)
    verify_d5_launch_prerequisites(
        config_path=args.config, two_rank_d4_summary=args.d4_two_summary,
        four_rank_d4_summary=args.d4_four_summary, layout=layout,
        worker_role="trainer",
    )
    _binding(args.rank, trainer_modules)
    if args.timeout_seconds <= 0:
        raise ValueError("D5 trainer timeout must be positive")
    descriptor = PolicyDescriptor(**json.loads(
        args.policy_descriptor.read_text(encoding="utf-8")
    ))
    if descriptor.run_kind != "production" or descriptor.update_index != 0:
        raise ValueError("D5 initial trainer worker requires policy-000000 production source")
    if args.output.resolve() == args.rollout_root.resolve():
        raise ValueError("D5 trainer output must differ from immutable rollout root")
    if (args.output / "ranks" / f"rank-{args.rank:03d}.json").exists():
        raise FileExistsError("D5 trainer rank report already exists")
    admission = admit_trainer_shard(
        args.rollout_root, config=config, policy=descriptor,
        rank=args.rank, expected_manifest_sha256=args.expected_manifest_sha256,
    )
    runtime = discover_torchrun_runtime(os.environ, expected_world_size=4)
    if runtime.rank != args.rank:
        raise RuntimeError("D5 trainer CLI rank differs from process-group rank")
    torch, dist = initialize_hccl_process_group(
        runtime, timeout_seconds=args.timeout_seconds,
    )
    device = torch.device(runtime.device)
    try:
        runtime_report = validate_initialized_hccl(runtime, torch, dist)
        _collective_phase(
            torch, dist, device, "prepare trainer output",
            lambda: args.output.mkdir(parents=True, exist_ok=True)
            if runtime.rank == 0 else None,
        )
        torch.manual_seed(config.seed + runtime.rank)
        torch.hpu.manual_seed_all(config.seed + runtime.rank)

        def load_trainer() -> SynchronousTrainer:
            policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
            restore_initial_adapter(
                policy.model, args.rollout_root / "policy_adapter",
                descriptor.adapter_sha256,
            )
            return SynchronousTrainer(config, policy, args.output / "trainer")

        trainer = _collective_phase(torch, dist, device, "load frozen policy", load_trainer)
        result = synchronized_trainer_update(
            trainer, admission, descriptor, runtime=runtime, torch=torch, dist=dist,
        )
        metrics = _global_metrics(result, trainer=trainer, torch=torch, dist=dist, device=device)
        probe = _collective_phase(
            torch, dist, device, "post-update policy probe",
            lambda: policy_probe(trainer.policy.model, args.rollout_root),
        )
        probe_values = torch.tensor(probe["log_probs"], dtype=torch.float32, device=device)
        probe_disagreement = global_source_max_abs_difference(torch, dist, probe_values)
        if not math.isfinite(probe_disagreement) or probe_disagreement > 2e-4:
            raise RuntimeError("D5 trainer ranks disagree on post-update policy probabilities")
        template = _collective_phase(
            torch, dist, device, "build production checkpoint template",
            lambda: build_production_checkpoint_template(
                run_id=args.run_id, rollout_root=args.rollout_root,
                receipt=admission.receipt, policy=descriptor, trainer=trainer,
                trainer_module_ids=trainer_modules,
                runtime_identity=dict(_runtime_identity(torch)),
            ),
        )
        staging = args.output / "rank_rng_staging"
        _collective_phase(
            torch, dist, device, "stage rank RNG",
            lambda: write_rank_rng_record(staging, template.ranks[runtime.rank]),
        )
        checkpoint = args.output / "trainer" / "checkpoints" / trainer.policy_version
        _collective_phase(
            torch, dist, device, "publish production checkpoint",
            lambda: trainer.save_checkpoint(
                metrics, validation_only=False,
                distributed_manifest=template, rank_rng_staging=staging,
            ) if runtime.rank == 0 else None,
        )

        def verify_checkpoint() -> str:
            published = read_distributed_manifest(checkpoint)
            verify_checkpoint_files(checkpoint, published)
            validate_resume_identity(
                published, expected=_expected_identity(template),
                allow_validation=False,
            )
            return file_sha256(checkpoint / "distributed" / "manifest.json")

        checkpoint_digest = _collective_phase(
            torch, dist, device, "verify production checkpoint", verify_checkpoint,
        )
        checkpoint_adapter_sha256 = directory_sha256(checkpoint / "adapter")
        report = {
            "status": "updated_checkpoint_committed_refresh_pending",
            "rank": runtime.rank,
            "module_id": str(trainer_modules[runtime.rank]),
            "source_manifest_sha256": admission.receipt.manifest_sha256,
            "source_policy_version": descriptor.version,
            "checkpoint_manifest_sha256": checkpoint_digest,
            "checkpoint_adapter_sha256": checkpoint_adapter_sha256,
            "runtime_contract": runtime_report,
            "update": asdict(result),
        }
        _collective_phase(
            torch, dist, device, "write trainer rank report",
            lambda: write_rank_update_report(args.output, runtime.rank, report),
        )
        if runtime.rank == 0:
            summary = {
                "status": "updated_checkpoint_committed_refresh_pending",
                "run_id": args.run_id, "checkpoint": str(checkpoint),
                "checkpoint_manifest_sha256": checkpoint_digest,
                "checkpoint_adapter_sha256": checkpoint_adapter_sha256,
                "parameter_sha256": result.parameter_sha256,
                "source_manifest_sha256": admission.receipt.manifest_sha256,
                "policy_probe": probe,
                "metrics": asdict(metrics),
            }
        else:
            summary = report
        def publish_summary() -> None:
            if runtime.rank == 0:
                target = args.output / "trainer_summary.json"
                if target.exists():
                    raise FileExistsError("D5 trainer summary already exists")
                temporary = args.output / ".trainer_summary.json.tmp"
                temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary.replace(target)
        _collective_phase(torch, dist, device, "trainer summary", publish_summary)
        return summary
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D5 initial four-rank production trainer")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--policy-descriptor", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--rollout-modules", required=True)
    parser.add_argument("--trainer-modules", required=True)
    parser.add_argument("--d4-two-summary", required=True, type=Path)
    parser.add_argument("--d4-four-summary", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    result = run_initial_trainer_rank(parser.parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
