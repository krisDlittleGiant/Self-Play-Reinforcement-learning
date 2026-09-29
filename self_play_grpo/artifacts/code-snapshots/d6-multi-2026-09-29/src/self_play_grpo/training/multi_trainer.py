"""Four-rank continuation from the latest committed production checkpoint.

Importing this module starts no HPU. The supervisor must bind four ranks to
the specified trainer modules and provide one verified fresh 64-game batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from self_play_grpo.config import load_config
from self_play_grpo.distributed import (
    discover_torchrun_runtime, initialize_hccl_process_group,
    validate_initialized_hccl,
)
from self_play_grpo.policies.llm import ConstrainedLLMPolicy
from self_play_grpo.rollouts.pilot import directory_sha256, file_sha256
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout
from self_play_grpo.training.d5_preflight import verify_d5_launch_prerequisites
from self_play_grpo.training.multi_identity import multi_code_identity
from self_play_grpo.training.d6_collection import descriptor_from_commit
from self_play_grpo.training.multi_recovery import audit_recovery
from self_play_grpo.training.multi_resume import restore_production_update
from self_play_grpo.training.distributed import (
    global_source_max_abs_difference, write_rank_update_report,
)
from self_play_grpo.training.distributed_checkpoint import (
    read_distributed_manifest, validate_resume_identity,
    verify_checkpoint_files, write_rank_rng_record,
)
from self_play_grpo.training.distributed_resume_gate import _expected_identity, _runtime_identity
from self_play_grpo.training.loop import SynchronousTrainer
from self_play_grpo.training.policy_refresh import policy_probe
from self_play_grpo.training.production_checkpoint import build_production_checkpoint_template
from self_play_grpo.training.trainer_handoff import admit_trainer_shard
from self_play_grpo.training.trainer_update import _collective_phase, _require_equal_digest, synchronized_trainer_update
from self_play_grpo.training.trainer_worker import _binding, _global_metrics, _modules


def run_resumed_trainer_rank(args: Any) -> dict[str, Any]:
    """Restore the ledger head, consume a new batch once, and publish the next update."""

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
        raise ValueError("D6 trainer timeout must be positive")
    plan = audit_recovery(
        args.run_root, config_path=args.config,
        experiment_seed=args.experiment_seed, expected_run_id=args.run_id,
    )
    if plan.status != "ready_for_next_collection" or plan.next_update < 3:
        raise RuntimeError("D6 multi-update worker requires committed/refresh-complete update 1")
    if args.output.resolve() != (args.run_root / f"trainer-{plan.next_update:06d}").resolve():
        raise ValueError("Trainer output differs from the next update")
    source = descriptor_from_commit(args.run_root, plan, config=config)
    descriptor_path = args.rollout_root / "policy_descriptor.json"
    descriptor = PolicyDescriptor(**json.loads(descriptor_path.read_text(encoding="utf-8")))
    if descriptor != source:
        raise ValueError("D6 rollout source differs from committed checkpoint policy")
    if (args.rollout_root.resolve() !=
            (args.run_root / f"rollout-{plan.next_collection_index:06d}").resolve()):
        raise ValueError("D6 rollout path differs from the expected collection index")
    if args.expected_manifest_sha256 in plan.consumed_batches:
        raise ValueError("D6 rollout batch was consumed by a previous update")
    if args.output.resolve() == args.rollout_root.resolve() or args.output.resolve() == args.run_root.resolve():
        raise ValueError("D6 trainer output must be separate from rollout/run roots")
    if (args.output / "ranks" / f"rank-{args.rank:03d}.json").exists():
        raise FileExistsError("D6 trainer rank report already exists")
    admission = admit_trainer_shard(
        args.rollout_root, config=config, policy=descriptor,
        rank=args.rank, expected_manifest_sha256=args.expected_manifest_sha256,
    )
    runtime = discover_torchrun_runtime(os.environ, expected_world_size=4)
    if runtime.rank != args.rank:
        raise RuntimeError("D6 trainer CLI rank differs from process-group rank")
    torch, dist = initialize_hccl_process_group(runtime, timeout_seconds=args.timeout_seconds)
    device = torch.device(runtime.device)
    try:
        runtime_report = validate_initialized_hccl(runtime, torch, dist)
        _collective_phase(
            torch, dist, device, "prepare D6 trainer output",
            lambda: args.output.mkdir(parents=True, exist_ok=True)
            if runtime.rank == 0 else None,
        )
        torch.manual_seed(config.seed + runtime.rank)
        torch.hpu.manual_seed_all(config.seed + runtime.rank)

        def load_and_restore() -> tuple[SynchronousTrainer, dict[str, Any]]:
            policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
            trainer = SynchronousTrainer(config, policy, args.output / "trainer")
            report = restore_production_update(
                trainer, root=args.run_root, plan=plan,
                rank=runtime.rank, trainer_module_ids=trainer_modules,
                runtime_identity=dict(_runtime_identity(torch)),
            )
            return trainer, report

        trainer, restore_report = _collective_phase(
            torch, dist, device, "restore committed production trainer", load_and_restore,
        )
        _require_equal_digest(
            torch, dist, device, restore_report["parameter_sha256"],
            "restored trainable parameters",
        )
        _require_equal_digest(
            torch, dist, device, restore_report["optimizer_sha256"],
            "restored optimizer state",
        )
        result = synchronized_trainer_update(
            trainer, admission, descriptor, runtime=runtime, torch=torch, dist=dist,
        )
        result = replace(result, optimizer_steps=trainer.update_index)
        metrics = _global_metrics(result, trainer=trainer, torch=torch, dist=dist, device=device)
        probe = _collective_phase(
            torch, dist, device, "D6 post-update policy probe",
            lambda: policy_probe(trainer.policy.model, args.rollout_root),
        )
        values = torch.tensor(probe["log_probs"], dtype=torch.float32, device=device)
        disagreement = global_source_max_abs_difference(torch, dist, values)
        metrics = replace(metrics, optimizer_steps=trainer.update_index)
        if not math.isfinite(disagreement) or disagreement > 2e-4:
            raise RuntimeError("D6 trainer ranks disagree on updated policy probabilities")
        template = _collective_phase(
            torch, dist, device, "build D6 production checkpoint template",
            lambda: replace(build_production_checkpoint_template(
                run_id=args.run_id, rollout_root=args.rollout_root,
                receipt=admission.receipt, policy=descriptor, trainer=trainer,
                trainer_module_ids=trainer_modules,
                runtime_identity=dict(_runtime_identity(torch)),
            ), code_identity=multi_code_identity()),
        )
        from self_play_grpo.training.multi_evidence import CommittedMetrics, publish_json, finalize_checkpoint
        prepared = {
            "rank": runtime.rank, "module_id": str(trainer_modules[runtime.rank]),
            "restored": restore_report, "runtime_contract": runtime_report,
            "update": asdict(result),
        }
        _collective_phase(
            torch, dist, device, "persist pre-checkpoint trainer evidence",
            lambda: publish_json(args.output / "prepared" / f"rank-{runtime.rank:03d}.json", prepared),
        )
        def bind_evidence():
            rows = [json.loads((args.output / "prepared" / f"rank-{rank:03d}.json").read_text())
                    for rank in range(4)]
            if any(row["rank"] != rank or row["update"]["parameter_sha256"] != result.parameter_sha256
                   or row["update"]["optimizer_sha256"] != result.optimizer_sha256
                   for rank, row in enumerate(rows)):
                raise ValueError("Pre-checkpoint trainer evidence differs")
            return CommittedMetrics(**asdict(metrics), continuation_evidence={
                "rank_reports": rows, "policy_probe": probe,
            })
        metrics = _collective_phase(torch, dist, device, "bind checkpoint evidence", bind_evidence)
        staging = args.output / "rank_rng_staging"
        _collective_phase(
            torch, dist, device, "stage D6 rank RNG",
            lambda: write_rank_rng_record(staging, template.ranks[runtime.rank]),
        )
        checkpoint = args.output / "trainer" / "checkpoints" / trainer.policy_version
        _collective_phase(
            torch, dist, device, "publish D6 production checkpoint",
            lambda: trainer.save_checkpoint(
                metrics, validation_only=False,
                distributed_manifest=template, rank_rng_staging=staging,
            ) if runtime.rank == 0 else None,
        )

        def verify_checkpoint() -> str:
            published = read_distributed_manifest(checkpoint)
            verify_checkpoint_files(checkpoint, published)
            validate_resume_identity(
                published, expected=_expected_identity(template), allow_validation=False,
            )
            return file_sha256(checkpoint / "distributed" / "manifest.json")

        checkpoint_digest = _collective_phase(
            torch, dist, device, "verify D6 checkpoint", verify_checkpoint,
        )
        summary = _collective_phase(
            torch, dist, device, "finalize checkpoint evidence",
            lambda: finalize_checkpoint(
                root=args.run_root, output=args.output, rollout_root=args.rollout_root,
                config=config, layout=layout, run_id=args.run_id, update=trainer.update_index,
            ) if runtime.rank == 0 else {"rank": runtime.rank, "checkpoint_manifest_sha256": checkpoint_digest},
        )
        return summary
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D6 guarded four-rank multi-update trainer")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-seed", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--rollout-modules", required=True)
    parser.add_argument("--trainer-modules", required=True)
    parser.add_argument("--d4-two-summary", required=True, type=Path)
    parser.add_argument("--d4-four-summary", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    print(json.dumps(run_resumed_trainer_rank(parser.parse_args(argv)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

