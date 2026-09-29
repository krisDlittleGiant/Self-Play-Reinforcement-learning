"""Short validation utilities and explicit, guarded experiment entry points."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Sequence

from self_play_grpo.config import ExperimentConfig, load_config
from self_play_grpo.envs.quoridor import QuoridorEnv
from self_play_grpo.evaluation.tournament import TournamentRunner, write_evaluation_jsonl
from self_play_grpo.policies.bots import RandomPolicy, ShortestPathPolicy, WallAwarePolicy, _bot_sample
from self_play_grpo.rewards.progress import path_distances


def command_validate_distributed_runtime(args: argparse.Namespace) -> int:
    """Prove a single-node torchrun/HCCL rank-to-HPU assignment."""

    from self_play_grpo.distributed import (
        discover_torchrun_runtime,
        validate_hccl_collectives,
    )

    runtime = discover_torchrun_runtime(
        os.environ,
        expected_world_size=args.expected_world_size,
    )
    result = validate_hccl_collectives(
        runtime,
        timeout_seconds=args.timeout_seconds,
    )
    if runtime.rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_validate_distributed_optimizer_math(args: argparse.Namespace) -> int:
    """Prove one synchronized synthetic AdamW step across trainer HPUs."""

    from self_play_grpo.distributed import (
        discover_torchrun_runtime,
        validate_distributed_optimizer_math,
    )

    runtime = discover_torchrun_runtime(
        os.environ,
        expected_world_size=args.expected_world_size,
    )
    result = validate_distributed_optimizer_math(
        runtime,
        timeout_seconds=args.timeout_seconds,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        parameter_count=args.parameter_count,
    )
    if runtime.rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_validate_distributed_checkpoint_resume(args: argparse.Namespace) -> int:
    """Run the bounded validation-only D4 save/reload continuation gate."""

    from self_play_grpo.training.distributed_resume_gate import (
        run_distributed_checkpoint_resume_gate,
    )

    return run_distributed_checkpoint_resume_gate(args)


def command_validate_distributed_policy_update(args: argparse.Namespace) -> int:
    """Run one real, validation-only replicated LoRA update on saved matches."""

    if args.replay_tolerance <= 0.0:
        raise ValueError("--replay-tolerance must be positive")
    if args.output.exists():
        raise FileExistsError(
            f"Distributed update output must be a new directory: {args.output}"
        )

    from self_play_grpo.distributed import (
        broadcast_trainable_parameters,
        discover_torchrun_runtime,
        initialize_hccl_process_group,
        validate_initialized_hccl,
    )
    from self_play_grpo.policies.llm import (
        ConstrainedLLMPolicy,
        checkpoint_qwen3_attention,
        constrained_log_probs_batched_shape,
    )
    from self_play_grpo.rollouts.pilot import (
        read_pilot_manifest,
        restore_initial_adapter,
        validate_manifest_request,
    )
    from self_play_grpo.training.distributed import (
        average_trainable_gradients,
        global_source_max_abs_difference,
        load_trainer_match_shard,
        trainer_match_indices,
        validate_distributed_training_manifest,
        verify_optimizer_replicas,
        write_rank_update_report,
    )
    from self_play_grpo.training.loop import SynchronousTrainer, UpdateMetrics
    from self_play_grpo.training.loss import backward_training_loss

    config = load_config(args.config)
    manifest = read_pilot_manifest(args.input)
    validate_manifest_request(
        manifest,
        config=config.to_dict(),
        base_seed=manifest.base_seed,
        replay_tolerance=args.replay_tolerance,
    )
    runtime = discover_torchrun_runtime(
        os.environ,
        expected_world_size=args.expected_world_size,
    )
    games_per_rank = validate_distributed_training_manifest(
        manifest,
        config=config.to_dict(),
        world_size=runtime.world_size,
    )
    indices = trainer_match_indices(
        runtime.rank,
        runtime.world_size,
        len(manifest.matches),
    )

    torch, dist = initialize_hccl_process_group(
        runtime,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        runtime_contract = validate_initialized_hccl(runtime, torch, dist)
        if runtime.rank == 0:
            args.output.mkdir(parents=True)
        dist.barrier()

        matches = load_trainer_match_shard(args.input, manifest, indices)
        policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
        restore_initial_adapter(
            policy.model,
            args.input / "policy_adapter",
            manifest.adapter_sha256,
        )
        adapter_sync = broadcast_trainable_parameters(policy.model, dist)
        trainer = SynchronousTrainer(config, policy, args.output)
        parameters = [
            parameter
            for parameter in policy.model.parameters()
            if parameter.requires_grad
        ]
        initial_parameters = [parameter.detach().clone() for parameter in parameters]

        policy.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        device = torch.device(runtime.device)
        local_action_total = sum(len(match.turns) for match in matches)
        memory_metrics: dict[str, int] = {}
        memory_supported = (
            hasattr(torch, "hpu")
            and hasattr(torch.hpu, "synchronize")
            and hasattr(torch.hpu, "memory_allocated")
            and hasattr(torch.hpu, "max_memory_allocated")
            and hasattr(torch.hpu, "reset_peak_memory_stats")
        )
        if memory_supported:
            torch.hpu.synchronize()
            memory_metrics["baseline_hpu_memory_bytes"] = int(
                torch.hpu.memory_allocated()
            )
            torch.hpu.reset_peak_memory_stats()

        def report_action_progress(completed_actions: int) -> None:
            if not (
                completed_actions == 1
                or completed_actions % 25 == 0
                or completed_actions == local_action_total
            ):
                return
            progress = {
                "completed_actions": completed_actions,
                "event": "trainer_backward_progress",
                "rank": runtime.rank,
                "total_actions": local_action_total,
            }
            if memory_supported:
                torch.hpu.synchronize()
                progress["hpu_memory_allocated_bytes"] = int(
                    torch.hpu.memory_allocated()
                )
                progress["peak_hpu_memory_bytes"] = int(
                    torch.hpu.max_memory_allocated()
                )
            print(json.dumps(progress, sort_keys=True), flush=True)

        with checkpoint_qwen3_attention(policy.model):
            local_output = backward_training_loss(
                policy.model,
                matches,
                clip_epsilon=config.training.clip_epsilon,
                loss_normalizer_per_game=config.training.loss_normalizer_per_game,
                kl_beta=config.training.kl_beta,
                reference_model=trainer.reference_model,
                action_progress_callback=report_action_progress,
                log_prob_function=constrained_log_probs_batched_shape,
                replay_tolerance=args.replay_tolerance,
            )
        if memory_supported:
            torch.hpu.synchronize()
            memory_metrics["post_backward_hpu_memory_bytes"] = int(
                torch.hpu.memory_allocated()
            )
            memory_metrics["peak_hpu_memory_bytes"] = int(
                torch.hpu.max_memory_allocated()
            )

        replay_error = torch.tensor(
            [local_output.max_abs_log_prob_error],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(replay_error, op=dist.ReduceOp.MAX)
        global_replay_error = float(replay_error.cpu().item())
        if global_replay_error > args.replay_tolerance:
            trainer.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Global behavior replay mismatch {global_replay_error:.6g} exceeds "
                f"{args.replay_tolerance:.6g}"
            )

        gradient_sync = average_trainable_gradients(
            policy.model,
            dist,
            world_size=runtime.world_size,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            config.training.max_grad_norm,
        )
        grad_norm_value = float(grad_norm.detach().cpu().item())
        grad_norm_tensor = torch.tensor(
            [grad_norm_value],
            dtype=torch.float32,
            device=device,
        )
        grad_norm_difference = global_source_max_abs_difference(
            torch,
            dist,
            grad_norm_tensor,
        )
        if not torch.isfinite(grad_norm).item() or grad_norm_value <= 0.0:
            trainer.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Distributed update requires a finite positive gradient norm, "
                f"got {grad_norm_value}"
            )

        trainer.optimizer.step()
        replica_equality = verify_optimizer_replicas(
            torch,
            dist,
            parameters,
            trainer.optimizer,
            device=device,
        )
        change_vector = torch.stack(
            [
                torch.max(torch.abs(parameter.detach() - initial))
                for parameter, initial in zip(parameters, initial_parameters)
            ]
        )
        parameter_change = float(torch.max(change_vector).cpu().item())
        changed_parameter_tensors = int((change_vector > 0).sum().cpu().item())
        changed_count_tensor = torch.tensor(
            [float(changed_parameter_tensors)],
            dtype=torch.float32,
            device=device,
        )
        changed_count_difference = global_source_max_abs_difference(
            torch,
            dist,
            changed_count_tensor,
        )
        trainer.optimizer.zero_grad(set_to_none=True)
        gradients_cleared = all(parameter.grad is None for parameter in parameters)

        equality_errors = [
            float(value)
            for key, value in replica_equality.items()
            if key.startswith("global_max_abs_")
        ]
        local_contract_ok = (
            grad_norm_difference == 0.0
            and changed_count_difference == 0.0
            and all(error == 0.0 for error in equality_errors)
            and parameter_change > 0.0
            and changed_parameter_tensors > 0
            and gradients_cleared
        )
        contract_ok = torch.tensor(
            [float(local_contract_ok)],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(contract_ok, op=dist.ReduceOp.SUM)
        if float(contract_ok.cpu().item()) != float(runtime.world_size):
            raise RuntimeError(
                "Distributed real-update replica contract failed: "
                f"grad_norm_difference={grad_norm_difference}, "
                f"changed_count_difference={changed_count_difference}, "
                f"replica_equality={replica_equality}, "
                f"parameter_change={parameter_change}, "
                f"changed_parameter_tensors={changed_parameter_tensors}, "
                f"gradients_cleared={gradients_cleared}"
            )

        local_owned_tokens = local_output.owned_tokens
        local_turns = sum(len(match.turns) for match in matches)
        aggregate = torch.tensor(
            [
                float(local_output.loss.detach().cpu().item()),
                float(local_output.policy_loss.detach().cpu().item()),
                float(local_output.kl_loss.detach().cpu().item()),
                local_output.mean_ratio * local_owned_tokens,
                local_output.clip_fraction * local_owned_tokens,
                float(local_owned_tokens),
                float(len(matches)),
                float(local_turns),
            ],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
        values = aggregate.cpu().tolist()
        global_owned_tokens = int(round(values[5]))
        global_games = int(round(values[6]))
        global_turns = int(round(values[7]))
        global_loss = values[0] / runtime.world_size
        global_policy_loss = values[1] / runtime.world_size
        global_kl_loss = values[2] / runtime.world_size
        global_mean_ratio = values[3] / global_owned_tokens
        global_clip_fraction = values[4] / global_owned_tokens

        rank_report = {
            "adapter_sha256": manifest.adapter_sha256,
            "behavior_policy_version": manifest.policy_version,
            "changed_parameter_tensors": changed_parameter_tensors,
            "games": len(matches),
            "grad_norm": grad_norm_value,
            "indices": list(indices),
            "local_max_abs_log_prob_error": local_output.max_abs_log_prob_error,
            "local_rank": runtime.local_rank,
            "owned_tokens": local_owned_tokens,
            "rank": runtime.rank,
            "status": "ok",
            "turns": local_turns,
            **memory_metrics,
        }
        report_path = write_rank_update_report(
            args.output,
            runtime.rank,
            rank_report,
        )
        print(
            json.dumps(
                {
                    "event": "trainer_rank_complete",
                    "games": len(matches),
                    "indices": list(indices),
                    "rank": runtime.rank,
                    "report": str(report_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        dist.barrier()

        checkpoint = None
        checkpoint_bytes = None
        summary = None
        if runtime.rank == 0:
            trainer.update_index = 1
            trainer.policy_version = "policy-000001"
            metrics = UpdateMetrics(
                update=0,
                policy_version=manifest.policy_version,
                games=global_games,
                turns=global_turns,
                owned_tokens=global_owned_tokens,
                loss=global_loss,
                policy_loss=global_policy_loss,
                kl_loss=global_kl_loss,
                mean_ratio_before_step=global_mean_ratio,
                clip_fraction_before_step=global_clip_fraction,
                replay_max_abs_error=global_replay_error,
                grad_norm=grad_norm_value,
                optimizer_steps=1,
            )
            checkpoint = trainer.save_checkpoint(metrics, validation_only=True)
            checkpoint_bytes = sum(
                path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
            )
            rank_reports = [
                json.loads(
                    (args.output / "ranks" / f"rank-{rank:03d}.json").read_text(
                        encoding="utf-8"
                    )
                )
                for rank in range(runtime.world_size)
            ]
            summary = {
                **adapter_sync,
                **gradient_sync,
                **replica_equality,
                "behavior_adapter_sha256": manifest.adapter_sha256,
                "behavior_policy_version": manifest.policy_version,
                "changed_parameter_count_difference": changed_count_difference,
                "changed_parameter_tensors": changed_parameter_tensors,
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint_bytes,
                "clip_fraction_before_step": global_clip_fraction,
                "games": global_games,
                "games_per_rank": games_per_rank,
                "global_max_abs_grad_norm_difference": grad_norm_difference,
                "global_max_abs_log_prob_error": global_replay_error,
                "grad_norm": grad_norm_value,
                "gradients_cleared": gradients_cleared,
                "loss": global_loss,
                "mean_ratio_before_step": global_mean_ratio,
                "new_policy_version": trainer.policy_version,
                "optimizer_steps": 1,
                "owned_tokens": global_owned_tokens,
                "parameter_change": parameter_change,
                "rank_reports": rank_reports,
                "runtime_contract": runtime_contract,
                "status": "ok",
                "turns": global_turns,
                "validation_only": True,
                "world_size": runtime.world_size,
            }
            if memory_supported:
                summary["max_baseline_hpu_memory_bytes"] = max(
                    int(report["baseline_hpu_memory_bytes"])
                    for report in rank_reports
                )
                summary["max_peak_hpu_memory_bytes"] = max(
                    int(report["peak_hpu_memory_bytes"])
                    for report in rank_reports
                )
                summary["max_post_backward_hpu_memory_bytes"] = max(
                    int(report["post_backward_hpu_memory_bytes"])
                    for report in rank_reports
                )
            summary_path = args.output / "summary.json"
            temporary = args.output / ".summary.json.tmp"
            temporary.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(summary_path)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def command_collect_distributed_policy_pilot(args: argparse.Namespace) -> int:
    """Collect one frozen-policy gameplay batch across a single HPU node."""

    from self_play_grpo.distributed import (
        broadcast_trainable_parameters,
        discover_torchrun_runtime,
        initialize_hccl_process_group,
        validate_initialized_hccl,
    )
    from self_play_grpo.policies.llm import (
        ConstrainedLLMPolicy,
        max_behavior_replay_error,
    )
    from self_play_grpo.rollouts.collector import BatchedMatchCollector, MatchCollector
    from self_play_grpo.rollouts.distributed import (
        DistributedRankReport,
        ordered_report_entries,
        rank_match_indices,
        read_rank_report,
        write_rank_report,
    )
    from self_play_grpo.rollouts.pilot import (
        initialize_pilot_root,
        make_match_entry,
        model_samples,
        pilot_game_id,
        read_and_replay_pilot_match,
        read_pilot_manifest,
        restore_initial_adapter,
        validate_registered_matches,
        write_pilot_manifest,
        write_pilot_match,
    )

    if args.games_per_rank <= 0:
        raise ValueError("--games-per-rank must be positive")
    if args.parallel_games is not None and args.parallel_games <= 0:
        raise ValueError("--parallel-games must be positive")
    if args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be positive")
    if args.output.exists():
        raise FileExistsError(
            f"Distributed rollout output must be a new directory: {args.output}"
        )

    config = load_config(args.config)
    runtime = discover_torchrun_runtime(
        os.environ,
        expected_world_size=args.expected_world_size,
    )
    total_games = runtime.world_size * args.games_per_rank
    if total_games != config.rollout.games_per_update:
        raise ValueError(
            f"world_size * games_per_rank is {total_games}, but the configured "
            f"games_per_update is {config.rollout.games_per_update}"
        )
    parallel_games = (
        config.rollout.parallel_games_per_rank
        if args.parallel_games is None
        else args.parallel_games
    )
    if parallel_games > args.games_per_rank:
        raise ValueError("--parallel-games cannot exceed --games-per-rank")
    base_seed = config.seed if args.seed is None else args.seed

    torch, dist = initialize_hccl_process_group(
        runtime,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        runtime_contract = validate_initialized_hccl(runtime, torch, dist)
        policy = ConstrainedLLMPolicy.load(config.model, config.rollout)

        if runtime.rank == 0:
            initialize_pilot_root(
                args.output,
                model=policy.model,
                config=config.to_dict(),
                model_revision=config.model.revision,
                base_seed=base_seed,
                target_games=total_games,
                replay_tolerance=args.replay_tolerance,
            )
        dist.barrier()
        manifest = read_pilot_manifest(args.output)
        if runtime.rank != 0:
            restore_initial_adapter(
                policy.model,
                args.output / "policy_adapter",
                manifest.adapter_sha256,
            )
        adapter_sync = broadcast_trainable_parameters(policy.model, dist)

        indices = rank_match_indices(
            runtime.rank,
            runtime.world_size,
            args.games_per_rank,
        )
        collector = MatchCollector(
            collect_progress=True,
            proxy_temperature=config.process_extension.proxy_temperature,
        )
        batched_collector = BatchedMatchCollector(
            collect_progress=True,
            proxy_temperature=config.process_extension.proxy_temperature,
        )
        local_entries = []
        for offset in range(0, len(indices), parallel_games):
            batch_indices = indices[offset : offset + parallel_games]
            seeds = tuple(base_seed + index for index in batch_indices)
            game_ids = tuple(
                pilot_game_id(manifest.policy_version, index, seed)
                for index, seed in zip(batch_indices, seeds)
            )
            if len(batch_indices) == 1:
                matches = (
                    collector.collect(
                        QuoridorEnv(config.environment),
                        policy,
                        game_id=game_ids[0],
                        seed=seeds[0],
                        policy_version=manifest.policy_version,
                    ),
                )
            else:
                matches = batched_collector.collect(
                    tuple(QuoridorEnv(config.environment) for _ in batch_indices),
                    policy,
                    game_ids=game_ids,
                    seeds=seeds,
                    policy_version=manifest.policy_version,
                )
            for index, match in zip(batch_indices, matches):
                path = write_pilot_match(args.output, index, match)
                restored, _ = read_and_replay_pilot_match(
                    args.output,
                    manifest,
                    index,
                )
                if restored.to_json() != match.to_json():
                    raise RuntimeError("Saved distributed match did not round-trip exactly")
                replay_error = max_behavior_replay_error(
                    policy.model,
                    model_samples(restored),
                )
                local_entries.append(
                    make_match_entry(
                        index=index,
                        match=restored,
                        path=path,
                        max_abs_log_prob_error=replay_error,
                    )
                )

        local_max_error = max(
            (entry.max_abs_log_prob_error for entry in local_entries),
            default=0.0,
        )
        global_error = torch.tensor(
            [local_max_error],
            dtype=torch.float32,
            device=torch.device(runtime.device),
        )
        dist.all_reduce(global_error, op=dist.ReduceOp.MAX)
        global_max_error = float(global_error.cpu().item())
        if global_max_error > args.replay_tolerance:
            raise RuntimeError(
                f"Global behavior replay mismatch {global_max_error:.6g} exceeds "
                f"{args.replay_tolerance:.6g}; partial artifacts preserved"
            )

        report = DistributedRankReport(
            rank=runtime.rank,
            local_rank=runtime.local_rank,
            world_size=runtime.world_size,
            games_per_rank=args.games_per_rank,
            policy_version=manifest.policy_version,
            adapter_sha256=manifest.adapter_sha256,
            matches=tuple(local_entries),
        )
        report_path = write_rank_report(args.output, report, manifest)
        print(
            json.dumps(
                {
                    "event": "rank_complete",
                    "games": len(local_entries),
                    "indices": list(indices),
                    "local_max_abs_log_prob_error": local_max_error,
                    "rank": runtime.rank,
                    "report": str(report_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        dist.barrier()

        if runtime.rank == 0:
            reports = tuple(
                read_rank_report(args.output, rank, manifest)
                for rank in range(runtime.world_size)
            )
            entries = ordered_report_entries(
                reports,
                world_size=runtime.world_size,
                games_per_rank=args.games_per_rank,
            )
            for entry in entries:
                manifest.append(entry)
            validate_registered_matches(args.output, manifest)
            write_pilot_manifest(args.output, manifest)
        dist.barrier()

        completed = read_pilot_manifest(args.output)
        if len(completed.matches) != total_games:
            raise RuntimeError("Rank-0 manifest publication is incomplete")
        if runtime.rank == 0:
            summary = {
                **completed.summary(),
                **adapter_sync,
                "adapter_sha256": completed.adapter_sha256,
                "base_seed": completed.base_seed,
                "games_per_rank": args.games_per_rank,
                "global_max_abs_log_prob_error": global_max_error,
                "manifest": str(args.output / "manifest.json"),
                "output": str(args.output),
                "parallel_games_per_rank": parallel_games,
                "policy_version": completed.policy_version,
                "probability_path": "kv_cache",
                "runtime_contract": runtime_contract,
                "status": "complete",
                "world_size": runtime.world_size,
            }
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class _RetreatPolicy:
    """Fixture-only bot that delays its own goal without placing walls."""

    name = "fixture_retreat"

    def select_action(self, env: QuoridorEnv, *, game_id: str, rng: random.Random):
        del game_id
        seat = env.current_seat
        moves = [action for action in env.legal_actions() if action.kind == "move"]
        if not moves:
            moves = list(env.legal_actions())
        scored = []
        for action in moves:
            clone = env.clone()
            scored.append((path_distances(clone.step(action))[seat], action.label))
        worst = max(score for score, _ in scored)
        labels = sorted(label for score, label in scored if score == worst)
        return _bot_sample(env, rng.choice(labels), self.name)


def _validate_winning_fixture(config: ExperimentConfig, target_seat: int) -> dict[str, object]:
    collector_policies = [_RetreatPolicy() for _ in range(4)]
    collector_policies[target_seat] = ShortestPathPolicy()
    from self_play_grpo.rollouts.collector import MatchCollector

    match = MatchCollector(collect_progress=False).collect(
        QuoridorEnv(config.environment),
        collector_policies,
        game_id=f"seat-contract-{target_seat}",
        seed=10_000 + target_seat,
        policy_version="engine-contract",
    )
    if match.final_results[target_seat] != 1.0:
        raise RuntimeError(
            f"Winning fixture for canonical seat {target_seat} did not win: "
            f"{match.final_results}"
        )
    return {
        "target_seat": target_seat,
        "results": match.final_results,
        "turns": len(match.turns),
        "termination_reason": match.termination_reason,
    }


def command_validate_engine(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    env = QuoridorEnv(config.environment)
    manifest = env.contract_manifest()
    expected_order = [0, 2, 1, 3]
    observed_order = []
    for _ in range(4):
        observed_order.append(env.seat_map.engine_player(env.current_seat))
        move = next((action for action in env.legal_actions() if action.kind == "move"), None)
        if move is None:
            raise RuntimeError("Initial seat-order fixture found no pawn move")
        env.step(move)
    if observed_order != expected_order:
        raise RuntimeError(
            f"Engine turn order {observed_order} differs from required {expected_order}"
        )
    replayed = QuoridorEnv.deserialize(env.serialize())
    if replayed.state_view() != env.state_view():
        raise RuntimeError("Serialized fixture did not reproduce the same structured state")
    winning_fixtures = [_validate_winning_fixture(config, seat) for seat in range(4)]
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": manifest,
                "observed_engine_turn_order": observed_order,
                "winning_fixtures": winning_fixtures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_bot_tournament(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    policies = {
        "random": RandomPolicy(),
        "shortest": ShortestPathPolicy(),
        "wall-aware": WallAwarePolicy(),
    }
    candidate = policies[args.candidate]
    opponents = tuple(policies[name] for name in args.opponents)
    runner = TournamentRunner(lambda: QuoridorEnv(config.environment))
    games = runner.run(
        candidate,
        opponents,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
    )
    summary = runner.summarize(games, bootstrap_seed=args.seed)
    write_evaluation_jsonl(args.output, games, summary)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def _canonical_action_labels(board_size: int) -> tuple[str, ...]:
    columns = tuple(chr(ord("A") + index) for index in range(board_size))
    moves = tuple(
        f"MOVE_{column}{row}"
        for column in columns
        for row in range(1, board_size + 1)
    )
    walls = tuple(
        f"WALL_{column}{row}{orientation}"
        for column in columns[:-1]
        for row in range(1, board_size)
        for orientation in ("H", "V")
    )
    # A forced pass is encoded by OpenSpiel as a move to the pawn's current
    # coordinate, so it is already represented by one of the MOVE labels.
    return (*moves, *walls)


def command_validate_model_assets(args: argparse.Namespace) -> int:
    """Validate pinned local files and the action tokenizer without loading weights."""

    config = load_config(args.config)
    model_config = config.model
    if model_config.local_path is None:
        raise RuntimeError("validate-model-assets requires model.local_path")
    checkpoint = Path(model_config.local_path)
    if not checkpoint.is_dir():
        raise RuntimeError(f"Local model checkpoint does not exist: {checkpoint}")

    metadata_files = sorted(
        (checkpoint / ".cache" / "huggingface" / "download").glob("*.metadata")
    )
    observed_revisions = {
        path.read_text(encoding="utf-8").splitlines()[0]
        for path in metadata_files
    }
    if observed_revisions != {model_config.revision}:
        raise RuntimeError(
            "Checkpoint metadata revisions differ from the configured pin: "
            f"{sorted(observed_revisions)}"
        )

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = ["model.safetensors"]
    missing_shards = [name for name in shard_names if not (checkpoint / name).is_file()]
    if missing_shards:
        raise RuntimeError(f"Checkpoint is missing weight shards: {missing_shards}")
    weight_bytes = sum((checkpoint / name).stat().st_size for name in shard_names)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers is required for tokenizer validation") from exc
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    token_lengths: dict[str, int] = {}
    for label in _canonical_action_labels(config.environment.board_size):
        text = f"{label}\n"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.decode(tokens, skip_special_tokens=False) != text:
            raise RuntimeError(f"Tokenizer does not round-trip action label {label!r}")
        if len(tokens) > config.rollout.max_new_tokens:
            raise RuntimeError(
                f"Action label {label!r} requires {len(tokens)} tokens; "
                f"limit is {config.rollout.max_new_tokens}"
            )
        token_lengths[label] = len(tokens)

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Choose one legal Quoridor action."},
            {"role": "user", "content": "Fixture state"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=model_config.enable_thinking,
    )
    max_action_tokens = max(token_lengths.values())
    print(
        json.dumps(
            {
                "action_labels_checked": len(token_lengths),
                "local_path": str(checkpoint),
                "max_action_tokens": max_action_tokens,
                "max_action_token_labels": sorted(
                    label for label, length in token_lengths.items()
                    if length == max_action_tokens
                ),
                "metadata_files": len(metadata_files),
                "model_id": model_config.id,
                "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                "revision": model_config.revision,
                "status": "ok",
                "tokenizer_class": type(tokenizer).__name__,
                "weight_bytes": weight_bytes,
                "weight_shards": shard_names,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    if not args.acknowledge_training:
        raise SystemExit(
            "Refusing to start training without --acknowledge-training. "
            "Run validate-engine and probability replay gates first."
        )
    from self_play_grpo.policies.llm import ConstrainedLLMPolicy
    from self_play_grpo.training.loop import SynchronousTrainer

    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    trainer = SynchronousTrainer(config, policy, args.artifact_dir)
    metrics = trainer.run_updates(args.updates)
    print(json.dumps([item.__dict__ for item in metrics], indent=2, sort_keys=True))
    return 0


def command_validate_model_load(args: argparse.Namespace) -> int:
    """Load the base model and LoRA adapter without running a forward pass."""

    from self_play_grpo.policies.llm import ConstrainedLLMPolicy

    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    parameters = tuple(policy.model.parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    devices = sorted({str(parameter.device) for parameter in parameters})
    dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    print(
        json.dumps(
            {
                "devices": devices,
                "dtypes": dtypes,
                "model_class": type(policy.model).__name__,
                "model_id": config.model.id,
                "revision": config.model.revision,
                "status": "ok",
                "tokenizer_class": type(policy.tokenizer).__name__,
                "total_parameters": total_parameters,
                "trainable_fraction": trainable_parameters / total_parameters,
                "trainable_parameters": trainable_parameters,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_validate_policy_forward(args: argparse.Namespace) -> int:
    """Sample one legal action and replay its constrained probabilities."""

    import torch

    from self_play_grpo.policies.llm import ConstrainedLLMPolicy, constrained_log_probs

    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    env = QuoridorEnv(config.environment)
    sample = policy.select_action(
        env,
        game_id="policy-forward-gate",
        rng=random.Random(config.seed),
    )
    action = env.action_for_label(sample.chosen_label)
    stepped = env.clone()
    stepped.step(action)

    policy.model.eval()
    with torch.no_grad():
        replayed = constrained_log_probs(policy.model, sample).float().cpu()
    behavior = torch.tensor(sample.behavior_log_probs, dtype=torch.float32)
    ratios = torch.exp(replayed - behavior)
    errors = torch.abs(replayed - behavior)
    print(
        json.dumps(
            {
                "allowed_token_counts": [len(tokens) for tokens in sample.allowed_token_ids],
                "behavior_log_probs": behavior.tolist(),
                "chosen_engine_action": action.engine_action,
                "chosen_label": sample.chosen_label,
                "completion_text": sample.completion_text,
                "completion_token_ids": list(sample.completion_token_ids),
                "forced_token_positions": sum(
                    len(tokens) == 1 for tokens in sample.allowed_token_ids
                ),
                "joint_actions_after_step": stepped.joint_actions,
                "max_abs_log_prob_error": float(errors.max()),
                "max_abs_ratio_error": float(torch.abs(ratios - 1.0).max()),
                "probability_ratios": ratios.tolist(),
                "replayed_log_probs": replayed.tolist(),
                "status": "ok",
                "stochastic_token_positions": sum(
                    len(tokens) > 1 for tokens in sample.allowed_token_ids
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_validate_padded_target_row_replay(args: argparse.Namespace) -> int:
    """Compare exact recorded-batch replay with lower-memory target-row replay."""

    import torch

    from self_play_grpo.policies.llm import (
        ConstrainedLLMPolicy,
        constrained_log_probs_batched_shape,
        constrained_log_probs_padded_target_row,
    )
    from self_play_grpo.rollouts.schema import read_matches_jsonl

    if args.replay_tolerance <= 0.0:
        raise ValueError("--replay-tolerance must be positive")
    matches = read_matches_jsonl(args.input)
    if len(matches) != 1:
        raise ValueError("Target-row replay gate requires exactly one saved match")
    match = matches[0]
    if not 0 <= args.joint_step < len(match.turns):
        raise ValueError(
            f"--joint-step {args.joint_step} is outside [0, {len(match.turns)})"
        )
    sample = match.turns[args.joint_step].policy_sample
    if sample.sampling_config.get("probability_path") != "batched_kv_cache":
        raise ValueError("Selected action was not sampled through a batched shape")

    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    policy.model.eval()
    with torch.no_grad():
        recorded = constrained_log_probs_batched_shape(policy.model, sample).float().cpu()
        target_row = (
            constrained_log_probs_padded_target_row(policy.model, sample)
            .float()
            .cpu()
        )
    behavior = torch.tensor(sample.behavior_log_probs, dtype=torch.float32)
    behavior_recorded_error = float(torch.max(torch.abs(behavior - recorded)))
    behavior_target_error = float(torch.max(torch.abs(behavior - target_row)))
    recorded_target_error = float(torch.max(torch.abs(recorded - target_row)))
    status = (
        "ok"
        if behavior_recorded_error <= args.replay_tolerance
        and behavior_target_error <= args.replay_tolerance
        else "mismatch"
    )
    result = {
        "behavior_log_probs": behavior.tolist(),
        "chosen_label": sample.chosen_label,
        "completion_tokens": len(sample.completion_token_ids),
        "joint_step": args.joint_step,
        "max_abs_behavior_vs_recorded_batch_error": behavior_recorded_error,
        "max_abs_behavior_vs_target_row_error": behavior_target_error,
        "max_abs_recorded_batch_vs_target_row_error": recorded_target_error,
        "prompt_tokens": len(sample.prompt_token_ids),
        "recorded_batch_log_probs": recorded.tolist(),
        "recorded_batch_row": int(sample.sampling_config["sampling_batch_row"]),
        "recorded_batch_size": int(sample.sampling_config["sampling_batch_size"]),
        "recorded_prompt_width": int(
            sample.sampling_config["sampling_prompt_width"]
        ),
        "replay_tolerance": args.replay_tolerance,
        "status": status,
        "target_row_log_probs": target_row.tolist(),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if status != "ok":
        raise RuntimeError(
            "Padded target-row replay is not behavior-equivalent: "
            f"target_error={behavior_target_error:.6g}, "
            f"tolerance={args.replay_tolerance:.6g}"
        )
    return 0


def command_validate_policy_batched_forward(args: argparse.Namespace) -> int:
    """Probe padded multi-game sampling and single-row replay equivalence."""

    import time
    import torch

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be positive")
    from self_play_grpo.policies.llm import (
        behavior_replay_diagnostics,
        ConstrainedLLMPolicy,
        constrained_log_probs_cached,
        summarize_log_prob_replay,
    )

    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    baseline_memory = None
    if (
        hasattr(torch, "hpu")
        and hasattr(torch.hpu, "memory_allocated")
        and hasattr(torch.hpu, "reset_peak_memory_stats")
    ):
        try:
            if hasattr(torch.hpu, "synchronize"):
                torch.hpu.synchronize()
            baseline_memory = int(torch.hpu.memory_allocated())
            torch.hpu.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            baseline_memory = None
    envs = []
    for row in range(args.batch_size):
        env = QuoridorEnv(config.environment)
        for _ in range(row % 4):
            move = next(
                (action for action in env.legal_actions() if action.kind == "move"),
                None,
            )
            if move is None:
                raise RuntimeError("Batch fixture could not find a legal pawn move")
            env.step(move)
        envs.append(env)
    sample_start = time.perf_counter()
    samples = policy.select_actions_batched(
        envs,
        game_ids=tuple(f"batch-probe-{row}" for row in range(args.batch_size)),
        rngs=tuple(random.Random(config.seed + row) for row in range(args.batch_size)),
    )
    if hasattr(torch, "hpu") and hasattr(torch.hpu, "synchronize"):
        try:
            torch.hpu.synchronize()
        except (AttributeError, RuntimeError):
            pass
    sampling_seconds = time.perf_counter() - sample_start
    for env, sample in zip(envs, samples):
        env.action_for_label(sample.chosen_label)
    replay_rows = behavior_replay_diagnostics(policy.model, samples)
    finite = all(bool(row["finite"]) for row in replay_rows)
    replay_error = (
        max(float(row["max_abs_error"]) for row in replay_rows)
        if finite
        else None
    )
    status = (
        "nonfinite"
        if not finite
        else "ok"
        if replay_error is not None and replay_error <= args.replay_tolerance
        else "mismatch"
    )
    single_rows = []
    policy.model.eval()
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            single = constrained_log_probs_cached(policy.model, sample).cpu().tolist()
            single_rows.append(
                {
                    "sample_index": sample_index,
                    **summarize_log_prob_replay(
                        sample.behavior_log_probs, single
                    ),
                }
            )
    single_finite = all(bool(row["finite"]) for row in single_rows)
    single_error = (
        max(float(row["max_abs_error"]) for row in single_rows)
        if single_finite
        else None
    )
    summary = {
        "batch_size": args.batch_size,
        "chosen_labels": [sample.chosen_label for sample in samples],
        "completion_token_counts": [
            len(sample.completion_token_ids) for sample in samples
        ],
        "max_abs_batch_shape_replay_error": replay_error,
        "max_abs_single_row_replay_error": single_error,
        "prompt_token_counts": [len(sample.prompt_token_ids) for sample in samples],
        "replay_tolerance": args.replay_tolerance,
        "replay_rows": replay_rows,
        "sampling_seconds": sampling_seconds,
        "single_row_replay_finite": single_finite,
        "status": status,
    }
    if baseline_memory is not None and hasattr(torch.hpu, "max_memory_allocated"):
        try:
            if hasattr(torch.hpu, "synchronize"):
                torch.hpu.synchronize()
            peak_memory = int(torch.hpu.max_memory_allocated())
            summary["baseline_hpu_memory_bytes"] = baseline_memory
            summary["peak_hpu_memory_bytes"] = peak_memory
            summary["peak_hpu_memory_increase_bytes"] = max(
                0, peak_memory - baseline_memory
            )
        except (AttributeError, RuntimeError):
            pass
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if status != "ok":
        raise RuntimeError(
            "Batched behavior replay failed: "
            f"status={status}, max_error={replay_error}, "
            f"tolerance={args.replay_tolerance:.6g}"
        )
    return 0


def command_collect_policy_match(args: argparse.Namespace) -> int:
    """Collect, persist, and replay one complete match without updating weights."""

    from self_play_grpo.policies.llm import (
        ConstrainedLLMPolicy,
        max_behavior_replay_error,
    )
    from self_play_grpo.rollouts.collector import MatchCollector
    from self_play_grpo.rollouts.replay import replay_match
    from self_play_grpo.rollouts.schema import read_matches_jsonl, write_matches_jsonl

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {args.output}")
    if args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be positive")
    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    match = MatchCollector(
        collect_progress=True,
        proxy_temperature=config.process_extension.proxy_temperature,
    ).collect(
        QuoridorEnv(config.environment),
        policy,
        game_id=f"policy-match-gate-seed-{seed}",
        seed=seed,
        policy_version=f"gate:{config.model.revision[:12]}",
    )
    write_matches_jsonl(args.output, [match])

    restored = read_matches_jsonl(args.output)
    if len(restored) != 1 or restored[0].to_json() != match.to_json():
        raise RuntimeError("Saved match JSONL did not round-trip exactly")
    replay_match(restored[0])

    samples = [
        turn.policy_sample
        for turn in match.turns
        if turn.policy_sample.completion_token_ids
    ]
    replay_error = max_behavior_replay_error(policy.model, samples)
    if replay_error > args.replay_tolerance:
        raise RuntimeError(
            f"Behavior replay mismatch {replay_error:.6g} exceeds "
            f"{args.replay_tolerance:.6g}; artifact preserved at {args.output}"
        )
    print(
        json.dumps(
            {
                "final_results": match.final_results,
                "game_id": match.game_id,
                "max_abs_log_prob_error": replay_error,
                "output": str(args.output),
                "owned_tokens": sum(
                    len(sample.completion_token_ids) for sample in samples
                ),
                "player_turns": [len(view) for view in match.player_views()],
                "policy_version": match.policy_version,
                "replay_tolerance": args.replay_tolerance,
                "seed": match.seed,
                "status": "ok",
                "termination_reason": match.termination_reason,
                "turns": len(match.turns),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_collect_policy_pilot(args: argparse.Namespace) -> int:
    """Collect a resumable frozen-policy pilot without optimizer activity."""

    from self_play_grpo.policies.llm import (
        ConstrainedLLMPolicy,
        max_behavior_replay_error,
    )
    from self_play_grpo.rollouts.collector import BatchedMatchCollector, MatchCollector
    from self_play_grpo.rollouts.pilot import (
        directory_sha256,
        discover_match_indices,
        initialize_pilot_root,
        make_match_entry,
        model_samples,
        pilot_game_id,
        pilot_match_relative_path,
        read_and_replay_pilot_match,
        read_pilot_manifest,
        restore_initial_adapter,
        validate_manifest_request,
        validate_registered_matches,
        write_pilot_manifest,
        write_pilot_match,
    )

    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.max_new_games is not None and args.max_new_games <= 0:
        raise ValueError("--max-new-games must be positive when supplied")
    if args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be positive")

    config = load_config(args.config)
    parallel_games = (
        config.rollout.parallel_games_per_rank
        if args.parallel_games is None
        else args.parallel_games
    )
    if parallel_games <= 0:
        raise ValueError("--parallel-games must be positive")
    base_seed = config.seed if args.seed is None else args.seed
    output = args.output
    policy = None

    if output.exists():
        if not output.is_dir() or not (output / "manifest.json").is_file():
            raise FileExistsError(
                f"Pilot output exists without a usable manifest: {output}"
            )
        manifest = read_pilot_manifest(output)
        validate_manifest_request(
            manifest,
            config=config.to_dict(),
            base_seed=base_seed,
            replay_tolerance=args.replay_tolerance,
        )
        if directory_sha256(output / "policy_adapter") != manifest.adapter_sha256:
            raise ValueError("Frozen pilot adapter digest does not match the manifest")
        validate_registered_matches(output, manifest)
        disk_indices = discover_match_indices(output)
        if len(disk_indices) < len(manifest.matches):
            raise RuntimeError("Pilot manifest references missing match artifacts")
        if args.games < len(disk_indices):
            raise ValueError(
                f"Requested target {args.games} is below {len(disk_indices)} "
                "persisted pilot games"
            )
        previous_target = manifest.target_games
        manifest.extend_target(args.games)
        if manifest.target_games != previous_target:
            write_pilot_manifest(output, manifest)
    else:
        policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
        manifest = initialize_pilot_root(
            output,
            model=policy.model,
            config=config.to_dict(),
            model_revision=config.model.revision,
            base_seed=base_seed,
            target_games=args.games,
            replay_tolerance=args.replay_tolerance,
        )
        disk_indices = []

    work_limit = args.max_new_games
    processed = 0
    pending_count = len(disk_indices) - len(manifest.matches)
    needs_work = pending_count > 0 or len(manifest.matches) < manifest.target_games
    if needs_work:
        if policy is None:
            policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
            restore_initial_adapter(
                policy.model,
                output / "policy_adapter",
                manifest.adapter_sha256,
            )
        collector = MatchCollector(
            collect_progress=True,
            proxy_temperature=config.process_extension.proxy_temperature,
        )
        batched_collector = BatchedMatchCollector(
            collect_progress=True,
            proxy_temperature=config.process_extension.proxy_temperature,
        )

        # A complete match can exist without a manifest entry if a process was
        # interrupted after its atomic artifact write. Validate and register it
        # rather than recollecting or overwriting it.
        for index in disk_indices[len(manifest.matches) :]:
            if work_limit is not None and processed >= work_limit:
                break
            match, _ = read_and_replay_pilot_match(output, manifest, index)
            replay_error = max_behavior_replay_error(policy.model, model_samples(match))
            if replay_error > manifest.replay_tolerance:
                raise RuntimeError(
                    f"Behavior replay mismatch {replay_error:.6g} exceeds "
                    f"{manifest.replay_tolerance:.6g}; unregistered artifact preserved"
                )
            entry = make_match_entry(
                index=index,
                match=match,
                path=output / pilot_match_relative_path(index),
                max_abs_log_prob_error=replay_error,
            )
            manifest.append(entry)
            write_pilot_manifest(output, manifest)
            processed += 1
            print(
                json.dumps(
                    {"event": "recovered_match", **entry.__dict__},
                    sort_keys=True,
                ),
                flush=True,
            )

        def commit_match(index: int, match) -> None:
            nonlocal processed
            path = write_pilot_match(output, index, match)
            restored, _ = read_and_replay_pilot_match(output, manifest, index)
            if restored.to_json() != match.to_json():
                raise RuntimeError("Saved pilot match did not round-trip exactly")
            replay_error = max_behavior_replay_error(
                policy.model, model_samples(restored)
            )
            if replay_error > manifest.replay_tolerance:
                raise RuntimeError(
                    f"Behavior replay mismatch {replay_error:.6g} exceeds "
                    f"{manifest.replay_tolerance:.6g}; artifact preserved at {path}"
                )
            entry = make_match_entry(
                index=index,
                match=restored,
                path=path,
                max_abs_log_prob_error=replay_error,
            )
            manifest.append(entry)
            write_pilot_manifest(output, manifest)
            disk_indices.append(index)
            processed += 1
            print(
                json.dumps(
                    {"event": "completed_match", **entry.__dict__},
                    sort_keys=True,
                ),
                flush=True,
            )

        while len(manifest.matches) < manifest.target_games:
            if work_limit is not None and processed >= work_limit:
                break
            index = len(manifest.matches)
            # Never skip an unregistered disk artifact merely because a work
            # limit stopped recovery during this invocation.
            if index < len(disk_indices):
                break
            remaining = manifest.target_games - index
            if work_limit is not None:
                remaining = min(remaining, work_limit - processed)
            batch_size = min(parallel_games, remaining)
            indices = tuple(range(index, index + batch_size))
            seeds = tuple(manifest.base_seed + item for item in indices)
            game_ids = tuple(
                pilot_game_id(manifest.policy_version, item, seed)
                for item, seed in zip(indices, seeds)
            )
            if batch_size == 1:
                matches = (
                    collector.collect(
                        QuoridorEnv(config.environment),
                        policy,
                        game_id=game_ids[0],
                        seed=seeds[0],
                        policy_version=manifest.policy_version,
                    ),
                )
            else:
                matches = batched_collector.collect(
                    tuple(QuoridorEnv(config.environment) for _ in indices),
                    policy,
                    game_ids=game_ids,
                    seeds=seeds,
                    policy_version=manifest.policy_version,
                )
            for match_index, match in zip(indices, matches):
                commit_match(match_index, match)

    summary = {
        **manifest.summary(),
        "adapter_sha256": manifest.adapter_sha256,
        "base_seed": manifest.base_seed,
        "manifest": str(output / "manifest.json"),
        "new_games_processed": processed,
        "output": str(output),
        "policy_version": manifest.policy_version,
        "parallel_games_per_rank": parallel_games,
        "probability_path": "kv_cache",
        "replay_tolerance": manifest.replay_tolerance,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_analyze_policy_pilot(args: argparse.Namespace) -> int:
    """Analyze persisted pilot behavior without loading OpenSpiel or the model."""

    from self_play_grpo.rollouts.analysis import analyze_policy_matches
    from self_play_grpo.rollouts.pilot import file_sha256, read_pilot_manifest
    from self_play_grpo.rollouts.schema import read_matches_jsonl

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    root = args.input
    manifest = read_pilot_manifest(root)
    matches = []
    for entry in manifest.matches:
        path = root / entry.path
        if file_sha256(path) != entry.sha256:
            raise ValueError(f"Pilot artifact hash mismatch: {path}")
        records = read_matches_jsonl(path)
        if len(records) != 1:
            raise ValueError(f"Expected one match record in {path}, found {len(records)}")
        match = records[0]
        if match.game_id != entry.game_id or match.policy_version != entry.policy_version:
            raise ValueError(f"Pilot artifact identity mismatch: {path}")
        matches.append(match)
    print(
        json.dumps(
            {
                "input": str(root),
                "status": "diagnostic",
                **analyze_policy_matches(matches, top_k=args.top_k),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_validate_policy_artifact(args: argparse.Namespace) -> int:
    """Replay a saved match artifact without recollecting or updating weights."""

    from self_play_grpo.policies.llm import (
        ConstrainedLLMPolicy,
        max_behavior_replay_error,
    )
    from self_play_grpo.rollouts.replay import replay_match
    from self_play_grpo.rollouts.schema import read_matches_jsonl

    if args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be positive")
    config = load_config(args.config)
    matches = read_matches_jsonl(args.input)
    if not matches:
        raise ValueError("The input contains no match records")
    for match in matches:
        replay_match(match)
    samples = [
        turn.policy_sample
        for match in matches
        for turn in match.turns
        if turn.policy_sample.completion_token_ids
    ]
    if not samples:
        raise ValueError("The input contains no model-generated tokens")

    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    replay_error = max_behavior_replay_error(policy.model, samples)
    if replay_error > args.replay_tolerance:
        raise RuntimeError(
            f"Behavior replay mismatch {replay_error:.6g} exceeds "
            f"{args.replay_tolerance:.6g}"
        )
    print(
        json.dumps(
            {
                "games": len(matches),
                "input": str(args.input),
                "max_abs_log_prob_error": replay_error,
                "owned_tokens": sum(
                    len(sample.completion_token_ids) for sample in samples
                ),
                "probability_path": "kv_cache",
                "replay_tolerance": args.replay_tolerance,
                "status": "ok",
                "turns": sum(len(match.turns) for match in matches),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_validate_policy_backward(args: argparse.Namespace) -> int:
    """Probe one saved action with synthetic advantage; never step an optimizer."""

    import math
    import torch

    from self_play_grpo.policies.llm import ConstrainedLLMPolicy, constrained_log_probs
    from self_play_grpo.rollouts.schema import read_matches_jsonl
    from self_play_grpo.training.loss import clipped_token_terms

    if not math.isfinite(args.replay_tolerance) or args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be finite and positive")
    matches = read_matches_jsonl(args.input)
    if len(matches) != 1:
        raise ValueError("Backward gate requires exactly one saved match")
    if not 0 <= args.joint_step < len(matches[0].turns):
        raise ValueError("--joint-step is outside the saved match")
    sample = matches[0].turns[args.joint_step].policy_sample
    if not any(len(allowed) > 1 for allowed in sample.allowed_token_ids):
        raise ValueError("Backward gate requires a stochastic action token")
    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    model = policy.model
    model.train()  # Exercise the same mode and enabled autograd as the trainer.
    model.zero_grad(set_to_none=True)
    try:
        new = constrained_log_probs(model, sample)
        old = torch.tensor(sample.behavior_log_probs, dtype=new.dtype, device=new.device)
        mask = torch.tensor(sample.loss_mask, dtype=new.dtype, device=new.device)
        error = float((new.detach() - old).abs().max().cpu())
        if not math.isfinite(error) or error > args.replay_tolerance:
            raise RuntimeError(f"Grad-enabled behavior replay mismatch: {error}")
        # The saved fixture is a draw with zero real advantage. A synthetic +1
        # probes gradient flow without modifying its recorded training credit.
        terms, ratios = clipped_token_terms(new, old, 1.0, mask, config.training.clip_epsilon)
        loss = -terms.sum() / config.training.loss_normalizer_per_game
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Non-finite backward probe loss")
        loss.backward()
        squared_norm = 0.0
        tensors_with_grad = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if not parameter.requires_grad:
                raise RuntimeError("Frozen parameter received a gradient")
            grad = parameter.grad.detach().float()
            if not bool(torch.isfinite(grad).all().item()):
                raise FloatingPointError("Non-finite parameter gradient")
            squared_norm += float(grad.square().sum().cpu())
            tensors_with_grad += 1
        norm = math.sqrt(squared_norm)
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Expected a finite nonzero gradient norm, got {norm}")
        result = {
            "status": "ok", "joint_step": args.joint_step,
            "chosen_label": sample.chosen_label, "synthetic_advantage": 1.0,
            "probability_path": "kv_cache", "max_abs_log_prob_error": error,
            "loss": float(loss.detach().cpu()), "grad_norm": norm,
            "parameter_tensors_with_grad": tensors_with_grad,
            "mean_ratio": float(ratios.detach().mean().cpu()),
            "optimizer_steps": 0,
        }
    finally:
        model.zero_grad(set_to_none=True)
    result["gradients_cleared"] = all(p.grad is None for p in model.parameters())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_validate_policy_batch_backward(args: argparse.Namespace) -> int:
    """Accumulate saved-action gradients one graph at a time without updating."""

    import copy
    import math
    import torch

    from self_play_grpo.policies.llm import ConstrainedLLMPolicy
    from self_play_grpo.rollouts.schema import read_matches_jsonl
    from self_play_grpo.training.loss import backward_training_loss

    if not math.isfinite(args.replay_tolerance) or args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be finite and positive")
    matches = read_matches_jsonl(args.input)
    if len(matches) != 1:
        raise ValueError("Batch backward gate requires exactly one saved match")
    if args.actions <= 0 or args.actions > len(matches[0].turns):
        raise ValueError("--actions must be between 1 and the saved turn count")

    probe = copy.deepcopy(matches[0])
    probe.turns = probe.turns[: args.actions]
    for turn in probe.turns:
        turn.credit.training_advantage = 1.0
    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    model = policy.model
    model.train()
    model.zero_grad(set_to_none=True)
    baseline_memory = None
    if (
        hasattr(torch, "hpu")
        and hasattr(torch.hpu, "memory_allocated")
        and hasattr(torch.hpu, "reset_peak_memory_stats")
    ):
        try:
            if hasattr(torch.hpu, "synchronize"):
                torch.hpu.synchronize()
            baseline_memory = int(torch.hpu.memory_allocated())
            torch.hpu.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            baseline_memory = None
    try:
        output = backward_training_loss(
            model,
            [probe],
            clip_epsilon=config.training.clip_epsilon,
            loss_normalizer_per_game=config.training.loss_normalizer_per_game,
        )
        if output.max_abs_log_prob_error > args.replay_tolerance:
            raise RuntimeError(
                "Grad-enabled behavior replay mismatch "
                f"{output.max_abs_log_prob_error:.6g} exceeds {args.replay_tolerance:.6g}"
            )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float("inf"))
        norm = float(grad_norm.detach().cpu())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Expected a finite nonzero gradient norm, got {norm}")
        tensors_with_grad = sum(parameter.grad is not None for parameter in trainable)
        result = {
            "actions": args.actions,
            "clip_fraction": output.clip_fraction,
            "grad_norm": norm,
            "loss": float(output.loss.cpu()),
            "max_abs_log_prob_error": output.max_abs_log_prob_error,
            "mean_ratio": output.mean_ratio,
            "optimizer_steps": 0,
            "owned_tokens": output.owned_tokens,
            "parameter_tensors_with_grad": tensors_with_grad,
            "probability_path": "kv_cache",
            "status": "ok",
            "synthetic_advantage": 1.0,
        }
        if baseline_memory is not None and hasattr(torch.hpu, "max_memory_allocated"):
            try:
                if hasattr(torch.hpu, "synchronize"):
                    torch.hpu.synchronize()
                peak_memory = int(torch.hpu.max_memory_allocated())
                result["baseline_hpu_memory_bytes"] = baseline_memory
                result["peak_hpu_memory_bytes"] = peak_memory
                result["peak_hpu_memory_increase_bytes"] = max(
                    0, peak_memory - baseline_memory
                )
            except (AttributeError, RuntimeError):
                pass
    finally:
        model.zero_grad(set_to_none=True)
    result["gradients_cleared"] = all(parameter.grad is None for parameter in model.parameters())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_show_action_history(args: argparse.Namespace) -> int:
    """Print recorded model actions without loading the model or game engine."""

    import math

    from self_play_grpo.rollouts.schema import read_matches_jsonl

    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.include_prompts and not args.json:
        raise ValueError("--include-prompts requires --json")
    matches = read_matches_jsonl(args.input)
    if not matches:
        raise ValueError("The input contains no match records")
    rows: list[dict[str, object]] = []
    for match in matches:
        for turn in match.turns:
            if turn.joint_step < args.start:
                continue
            if args.seat is not None and turn.seat != args.seat:
                continue
            sample = turn.policy_sample
            action_log_probability = sum(sample.behavior_log_probs)
            row: dict[str, object] = {
                "action_log_probability": action_log_probability,
                "action_probability": math.exp(action_log_probability),
                "allowed_token_counts": [
                    len(allowed) for allowed in sample.allowed_token_ids
                ],
                "chosen_engine_action": turn.chosen_engine_action,
                "chosen_label": sample.chosen_label,
                "chosen_notation": turn.chosen_notation,
                "completion_text": sample.completion_text,
                "completion_token_ids": list(sample.completion_token_ids),
                "game_id": match.game_id,
                "joint_step": turn.joint_step,
                "legal_action_count": len(turn.legal_actions),
                "player_local_step": turn.player_local_step,
                "seat": turn.seat,
                "stochastic_token_positions": sum(
                    len(allowed) > 1 for allowed in sample.allowed_token_ids
                ),
                "token_log_probabilities": list(sample.behavior_log_probs),
            }
            if args.include_prompts:
                row["legal_action_labels"] = [
                    action.label for action in turn.legal_actions
                ]
                row["observation"] = turn.observation
                row["prompt_text"] = sample.prompt_text
            rows.append(row)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.json:
        print(
            json.dumps(
                {
                    "actions": rows,
                    "input": str(args.input),
                    "matches": len(matches),
                    "rows": len(rows),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print("step seat local label        notation tokens stochastic action_logp")
    for row in rows:
        print(
            f"{int(row['joint_step']):4d} "
            f"{int(row['seat']):4d} "
            f"{int(row['player_local_step']):5d} "
            f"{str(row['chosen_label']):12s} "
            f"{str(row['chosen_notation']):8s} "
            f"{len(row['completion_token_ids']):6d} "
            f"{int(row['stochastic_token_positions']):10d} "
            f"{float(row['action_log_probability']):11.6f}"
        )
    for match in matches:
        print(
            f"# {match.game_id}: turns={len(match.turns)} "
            f"result={list(match.final_results)} reason={match.termination_reason}"
        )
    return 0


def command_validate_checkpoint_roundtrip(args: argparse.Namespace) -> int:
    """Save, perturb, and exactly restore an adapter-only checkpoint."""

    import math
    import torch

    from self_play_grpo.policies.llm import ConstrainedLLMPolicy, constrained_log_probs
    from self_play_grpo.rollouts.schema import read_matches_jsonl
    from self_play_grpo.training.loop import SynchronousTrainer, UpdateMetrics

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite validation output: {args.output}")
    if not math.isfinite(args.replay_tolerance) or args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be finite and positive")
    matches = read_matches_jsonl(args.input)
    if len(matches) != 1:
        raise ValueError("Checkpoint gate requires exactly one saved match")
    if not 0 <= args.joint_step < len(matches[0].turns):
        raise ValueError("--joint-step is outside the saved match")
    sample = matches[0].turns[args.joint_step].policy_sample

    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    trainer = SynchronousTrainer(config, policy, args.output)
    trainable = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Checkpoint gate found no trainable adapter parameters")
    parameter_before = trainable[0].detach().cpu().clone()
    policy.model.eval()
    with torch.no_grad():
        probabilities_before = constrained_log_probs(policy.model, sample).float().cpu()

    torch.manual_seed(20_260_920)
    metrics = UpdateMetrics(
        update=0,
        policy_version=trainer.policy_version,
        games=0,
        turns=0,
        owned_tokens=0,
        loss=0.0,
        policy_loss=0.0,
        kl_loss=0.0,
        mean_ratio_before_step=1.0,
        clip_fraction_before_step=0.0,
        replay_max_abs_error=0.0,
        grad_norm=0.0,
        optimizer_steps=0,
    )
    checkpoint = trainer.save_checkpoint(metrics, validation_only=True)
    expected_cpu_random = torch.rand(8)
    device = next(policy.model.parameters()).device
    expected_hpu_random = None
    if (
        device.type == "hpu"
        and hasattr(torch, "hpu")
        and hasattr(torch.hpu, "get_rng_state_all")
        and hasattr(torch.hpu, "set_rng_state_all")
    ):
        expected_hpu_random = torch.rand(8, device=device).cpu()

    with torch.no_grad():
        trainable[0].add_(0.125)
    torch.manual_seed(91_827)
    trainer.load_checkpoint(checkpoint, allow_validation=True)
    actual_cpu_random = torch.rand(8)
    cpu_rng_equal = bool(torch.equal(actual_cpu_random, expected_cpu_random))
    hpu_rng_equal = None
    if expected_hpu_random is not None:
        actual_hpu_random = torch.rand(8, device=device).cpu()
        hpu_rng_equal = bool(torch.equal(actual_hpu_random, expected_hpu_random))

    parameter_restored = bool(torch.equal(trainable[0].detach().cpu(), parameter_before))
    policy.model.eval()
    with torch.no_grad():
        probabilities_after = constrained_log_probs(policy.model, sample).float().cpu()
    probability_error = float((probabilities_after - probabilities_before).abs().max())
    behavior = torch.tensor(sample.behavior_log_probs, dtype=probabilities_after.dtype)
    behavior_error = float((probabilities_after - behavior).abs().max())
    if not parameter_restored:
        raise RuntimeError("Adapter parameter was not restored exactly")
    if not cpu_rng_equal or hpu_rng_equal is False:
        raise RuntimeError("Checkpoint RNG continuation was not exact")
    if probability_error != 0.0 or behavior_error > args.replay_tolerance:
        raise RuntimeError(
            "Checkpoint probability continuation failed: "
            f"before/after={probability_error}, behavior={behavior_error}"
        )
    if (checkpoint / "model_state.pt").exists():
        raise RuntimeError("Checkpoint unexpectedly contains the frozen base-model state")
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "adapter_parameter_restored": parameter_restored,
                "behavior_replay_error": behavior_error,
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": sum(path.stat().st_size for path in files),
                "checkpoint_files": [str(path.relative_to(checkpoint)) for path in files],
                "checkpoint_format": 2,
                "continuation_log_prob_error": probability_error,
                "cpu_rng_continuation_equal": cpu_rng_equal,
                "frozen_base_model_saved": False,
                "hpu_rng_continuation_equal": hpu_rng_equal,
                "optimizer_state_entries": len(trainer.optimizer.state),
                "optimizer_steps": 0,
                "policy_version": trainer.policy_version,
                "status": "ok",
                "update_index": trainer.update_index,
                "validation_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_validate_optimizer_checkpoint(args: argparse.Namespace) -> int:
    """Require exact optimizer continuation after an adapter checkpoint reload."""

    import math
    import torch

    from self_play_grpo.policies.llm import ConstrainedLLMPolicy, constrained_log_probs
    from self_play_grpo.rollouts.schema import read_matches_jsonl
    from self_play_grpo.training.loop import SynchronousTrainer, UpdateMetrics
    from self_play_grpo.training.loss import clipped_token_terms

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite validation output: {args.output}")
    if not math.isfinite(args.replay_tolerance) or args.replay_tolerance <= 0:
        raise ValueError("--replay-tolerance must be finite and positive")
    matches = read_matches_jsonl(args.input)
    if len(matches) != 1:
        raise ValueError("Optimizer checkpoint gate requires exactly one saved match")
    if not 0 <= args.joint_step < len(matches[0].turns):
        raise ValueError("--joint-step is outside the saved match")
    sample = matches[0].turns[args.joint_step].policy_sample
    config = load_config(args.config)
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    trainer = SynchronousTrainer(config, policy, args.output)
    model = policy.model
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def optimizer_step() -> dict[str, float]:
        model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        new = constrained_log_probs(model, sample)
        old = torch.tensor(sample.behavior_log_probs, dtype=new.dtype, device=new.device)
        ownership = torch.tensor(sample.loss_mask, dtype=new.dtype, device=new.device)
        replay_error = float((new.detach() - old).abs().max().cpu())
        terms, ratios = clipped_token_terms(
            new,
            old,
            1.0,
            ownership,
            config.training.clip_epsilon,
        )
        loss = -terms.sum() / config.training.loss_normalizer_per_game
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            config.training.max_grad_norm,
        )
        norm = float(grad_norm.detach().cpu())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Expected a finite nonzero gradient norm, got {norm}")
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        return {
            "clip_fraction": float(
                ((ratios < 1.0 - config.training.clip_epsilon)
                | (ratios > 1.0 + config.training.clip_epsilon))
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "grad_norm": norm,
            "loss": float(loss.detach().cpu()),
            "mean_ratio": float(ratios.detach().mean().cpu()),
            "replay_error": replay_error,
        }

    def optimizer_tensor_snapshot() -> list[tuple[str, object]]:
        snapshot: list[tuple[str, object]] = []
        for parameter_index, parameter in enumerate(trainable):
            for key, value in sorted(trainer.optimizer.state[parameter].items()):
                if torch.is_tensor(value):
                    snapshot.append(
                        (f"{parameter_index}:{key}", value.detach().cpu().clone())
                    )
        return snapshot

    parameters_before = [parameter.detach().cpu().clone() for parameter in trainable]
    first = optimizer_step()
    if first["replay_error"] > args.replay_tolerance:
        raise RuntimeError(
            f"Initial behavior replay mismatch {first['replay_error']:.6g} exceeds "
            f"{args.replay_tolerance:.6g}"
        )
    changed_after_first = sum(
        not torch.equal(before, parameter.detach().cpu())
        for before, parameter in zip(parameters_before, trainable)
    )
    del parameters_before
    if changed_after_first <= 0 or not trainer.optimizer.state:
        raise RuntimeError("Controlled optimizer step did not create updated state")

    trainer.update_index = 1
    trainer.policy_version = "policy-000001"
    metrics = UpdateMetrics(
        update=0,
        policy_version="policy-000000",
        games=1,
        turns=1,
        owned_tokens=len(sample.completion_token_ids),
        loss=first["loss"],
        policy_loss=first["loss"],
        kl_loss=0.0,
        mean_ratio_before_step=first["mean_ratio"],
        clip_fraction_before_step=first["clip_fraction"],
        replay_max_abs_error=first["replay_error"],
        grad_norm=first["grad_norm"],
        optimizer_steps=1,
    )
    checkpoint = trainer.save_checkpoint(metrics, validation_only=True)

    expected_second = optimizer_step()
    expected_parameters = [parameter.detach().cpu().clone() for parameter in trainable]
    expected_optimizer = optimizer_tensor_snapshot()
    trainer.load_checkpoint(checkpoint, allow_validation=True)
    resumed_second = optimizer_step()
    parameters_equal = all(
        torch.equal(expected, parameter.detach().cpu())
        for expected, parameter in zip(expected_parameters, trainable)
    )
    resumed_optimizer = optimizer_tensor_snapshot()
    optimizer_equal = len(expected_optimizer) == len(resumed_optimizer) and all(
        expected_key == resumed_key and torch.equal(expected_value, resumed_value)
        for (expected_key, expected_value), (resumed_key, resumed_value)
        in zip(expected_optimizer, resumed_optimizer)
    )
    scalar_continuation_equal = expected_second == resumed_second
    if not parameters_equal or not optimizer_equal or not scalar_continuation_equal:
        raise RuntimeError(
            "Optimizer continuation differs after reload: "
            f"parameters={parameters_equal}, optimizer={optimizer_equal}, "
            f"scalars={scalar_continuation_equal}"
        )
    files = [path for path in checkpoint.rglob("*") if path.is_file()]
    print(
        json.dumps(
            {
                "changed_parameter_tensors_after_first_step": changed_after_first,
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": sum(path.stat().st_size for path in files),
                "checkpoint_optimizer_steps": 1,
                "continuation_metrics_equal": scalar_continuation_equal,
                "continuation_optimizer_tensors_equal": optimizer_equal,
                "continuation_parameters_equal": parameters_equal,
                "first_step": first,
                "gradients_cleared": all(
                    parameter.grad is None for parameter in model.parameters()
                ),
                "optimizer_state_entries": len(trainer.optimizer.state),
                "optimizer_state_tensors_compared": len(expected_optimizer),
                "optimizer_steps_executed": 3,
                "resumed_second_step": resumed_second,
                "status": "ok",
                "synthetic_advantage": 1.0,
                "validation_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_diagnose_policy_replay(args: argparse.Namespace) -> int:
    """Compare recorded, KV-cached, and full-sequence probabilities."""

    import torch

    from self_play_grpo.policies.llm import (
        ConstrainedLLMPolicy,
        constrained_log_probs_cached,
        constrained_log_probs_full_sequence,
    )
    from self_play_grpo.rollouts.schema import read_matches_jsonl

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    config = load_config(args.config)
    matches = read_matches_jsonl(args.input)
    if not matches:
        raise ValueError("The input contains no match records")
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    rows: list[dict[str, object]] = []
    policy.model.eval()
    with torch.no_grad():
        for match in matches:
            for turn in match.turns:
                sample = turn.policy_sample
                if not sample.completion_token_ids:
                    continue
                full = constrained_log_probs_full_sequence(policy.model, sample).float().cpu()
                cached = constrained_log_probs_cached(policy.model, sample).float().cpu()
                behavior = torch.tensor(sample.behavior_log_probs, dtype=torch.float32)
                for token_index, token in enumerate(sample.completion_token_ids):
                    behavior_full_error = abs(float(full[token_index] - behavior[token_index]))
                    behavior_cached_error = abs(float(cached[token_index] - behavior[token_index]))
                    cached_full_error = abs(float(full[token_index] - cached[token_index]))
                    rows.append(
                        {
                            "allowed_tokens": len(sample.allowed_token_ids[token_index]),
                            "behavior_log_prob": float(behavior[token_index]),
                            "behavior_vs_cached_error": behavior_cached_error,
                            "behavior_vs_full_error": behavior_full_error,
                            "cached_log_prob": float(cached[token_index]),
                            "cached_vs_full_error": cached_full_error,
                            "full_log_prob": float(full[token_index]),
                            "full_ratio": float(torch.exp(full[token_index] - behavior[token_index])),
                            "game_id": match.game_id,
                            "joint_step": turn.joint_step,
                            "label": sample.chosen_label,
                            "prompt_tokens": len(sample.prompt_token_ids),
                            "sampling_config": dict(sample.sampling_config),
                            "seat": turn.seat,
                            "token_id": token,
                            "token_index": token_index,
                            "token_text": policy.tokenizer.decode(
                                [token], skip_special_tokens=False
                            ),
                        }
                    )
    if not rows:
        raise ValueError("The input contains no model-generated tokens")
    worst_full = sorted(
        rows,
        key=lambda row: float(row["behavior_vs_full_error"]),
        reverse=True,
    )[: args.top_k]
    worst_cached = sorted(
        rows,
        key=lambda row: float(row["behavior_vs_cached_error"]),
        reverse=True,
    )[: args.top_k]
    print(
        json.dumps(
            {
                "input": str(args.input),
                "max_behavior_vs_cached_error": max(
                    float(row["behavior_vs_cached_error"]) for row in rows
                ),
                "max_behavior_vs_full_error": max(
                    float(row["behavior_vs_full_error"]) for row in rows
                ),
                "max_cached_vs_full_error": max(
                    float(row["cached_vs_full_error"]) for row in rows
                ),
                "status": "diagnostic",
                "tokens_checked": len(rows),
                "worst_behavior_vs_cached": worst_cached,
                "worst_behavior_vs_full": worst_full,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="self-play-grpo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    distributed_runtime = subparsers.add_parser(
        "validate-distributed-runtime",
        help="validate a torchrun single-node HCCL rank/device assignment",
    )
    distributed_runtime.add_argument(
        "--expected-world-size",
        required=True,
        type=int,
    )
    distributed_runtime.add_argument("--timeout-seconds", type=int, default=120)
    distributed_runtime.set_defaults(handler=command_validate_distributed_runtime)

    distributed_optimizer_math = subparsers.add_parser(
        "validate-distributed-optimizer-math",
        help="average synthetic gradients and take one identical AdamW step",
    )
    distributed_optimizer_math.add_argument(
        "--expected-world-size",
        required=True,
        type=int,
    )
    distributed_optimizer_math.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    distributed_optimizer_math.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )
    distributed_optimizer_math.add_argument(
        "--parameter-count",
        type=int,
        default=16,
    )
    distributed_optimizer_math.add_argument("--timeout-seconds", type=int, default=120)
    distributed_optimizer_math.set_defaults(
        handler=command_validate_distributed_optimizer_math
    )

    distributed_update = subparsers.add_parser(
        "validate-distributed-policy-update",
        help="run one validation-only real LoRA update on a complete rollout manifest",
    )
    distributed_update.add_argument("--config", required=True, type=Path)
    distributed_update.add_argument("--input", required=True, type=Path)
    distributed_update.add_argument("--output", required=True, type=Path)
    distributed_update.add_argument(
        "--expected-world-size",
        required=True,
        type=int,
    )
    distributed_update.add_argument("--replay-tolerance", type=float, default=2e-4)
    distributed_update.add_argument("--timeout-seconds", type=int, default=1800)
    distributed_update.set_defaults(handler=command_validate_distributed_policy_update)

    distributed_resume = subparsers.add_parser(
        "validate-distributed-checkpoint-resume",
        help="compare exact uninterrupted and reloaded trainer continuation on saved D2/D3 inputs",
    )
    distributed_resume.add_argument("--config", required=True, type=Path)
    distributed_resume.add_argument("--source-rollout", required=True, type=Path)
    distributed_resume.add_argument("--source-checkpoint", required=True, type=Path)
    distributed_resume.add_argument("--output", required=True, type=Path)
    distributed_resume.add_argument("--expected-world-size", required=True, type=int)
    distributed_resume.add_argument("--joint-step", type=int, default=0)
    distributed_resume.add_argument("--match-offset", type=int, default=0)
    distributed_resume.add_argument("--seed", type=int, default=741)
    distributed_resume.add_argument("--timeout-seconds", type=int, default=1800)
    distributed_resume.add_argument("--allow-validation-source", action="store_true")
    distributed_resume.set_defaults(handler=command_validate_distributed_checkpoint_resume)

    distributed_pilot = subparsers.add_parser(
        "collect-distributed-policy-pilot",
        help="collect one rank-sharded frozen-policy gameplay batch over HCCL",
    )
    distributed_pilot.add_argument("--config", required=True, type=Path)
    distributed_pilot.add_argument("--output", required=True, type=Path)
    distributed_pilot.add_argument(
        "--expected-world-size",
        required=True,
        type=int,
    )
    distributed_pilot.add_argument("--games-per-rank", type=int, default=4)
    distributed_pilot.add_argument("--parallel-games", type=int)
    distributed_pilot.add_argument("--seed", type=int)
    distributed_pilot.add_argument("--replay-tolerance", type=float, default=2e-4)
    distributed_pilot.add_argument("--timeout-seconds", type=int, default=1800)
    distributed_pilot.set_defaults(handler=command_collect_distributed_policy_pilot)

    validate = subparsers.add_parser("validate-engine", help="run the deterministic M0 contract gates")
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(handler=command_validate_engine)

    tournament = subparsers.add_parser("bot-tournament", help="run a reproducible fixed-bot evaluation")
    tournament.add_argument("--config", required=True, type=Path)
    tournament.add_argument("--candidate", choices=("random", "shortest", "wall-aware"), default="shortest")
    tournament.add_argument(
        "--opponents",
        nargs=3,
        choices=("random", "shortest", "wall-aware"),
        default=("random", "shortest", "wall-aware"),
    )
    tournament.add_argument("--games-per-seat", type=int, default=2)
    tournament.add_argument("--seed", type=int, default=11)
    tournament.add_argument("--output", required=True, type=Path)
    tournament.set_defaults(handler=command_bot_tournament)

    model_assets = subparsers.add_parser(
        "validate-model-assets",
        help="validate the pinned local checkpoint and tokenizer without loading weights",
    )
    model_assets.add_argument("--config", required=True, type=Path)
    model_assets.set_defaults(handler=command_validate_model_assets)

    model_load = subparsers.add_parser(
        "validate-model-load",
        help="load the pinned model and LoRA adapter without a forward pass",
    )
    model_load.add_argument("--config", required=True, type=Path)
    model_load.set_defaults(handler=command_validate_model_load)

    policy_forward = subparsers.add_parser(
        "validate-policy-forward",
        help="sample one constrained action and replay its behavior probabilities",
    )
    policy_forward.add_argument("--config", required=True, type=Path)
    policy_forward.set_defaults(handler=command_validate_policy_forward)

    target_row_replay = subparsers.add_parser(
        "validate-padded-target-row-replay",
        help="compare exact batched replay with a padded single target row",
    )
    target_row_replay.add_argument("--config", required=True, type=Path)
    target_row_replay.add_argument("--input", required=True, type=Path)
    target_row_replay.add_argument("--joint-step", type=int, default=0)
    target_row_replay.add_argument("--replay-tolerance", type=float, default=2e-4)
    target_row_replay.set_defaults(
        handler=command_validate_padded_target_row_replay
    )

    batched_forward = subparsers.add_parser(
        "validate-policy-batched-forward",
        help="sample a padded multi-game batch and gate single-row probability replay",
    )
    batched_forward.add_argument("--config", required=True, type=Path)
    batched_forward.add_argument("--batch-size", type=int, default=2)
    batched_forward.add_argument("--replay-tolerance", type=float, default=2e-4)
    batched_forward.set_defaults(handler=command_validate_policy_batched_forward)

    collect_match = subparsers.add_parser(
        "collect-policy-match",
        help="collect and replay one complete model match without an optimizer step",
    )
    collect_match.add_argument("--config", required=True, type=Path)
    collect_match.add_argument("--output", required=True, type=Path)
    collect_match.add_argument("--seed", type=int)
    collect_match.add_argument("--replay-tolerance", type=float, default=2e-4)
    collect_match.set_defaults(handler=command_collect_policy_match)

    collect_pilot = subparsers.add_parser(
        "collect-policy-pilot",
        help="resume a frozen-policy multi-game pilot with atomic per-game commits",
    )
    collect_pilot.add_argument("--config", required=True, type=Path)
    collect_pilot.add_argument("--output", required=True, type=Path)
    collect_pilot.add_argument(
        "--games",
        required=True,
        type=int,
        help="total desired games, including games already present on resume",
    )
    collect_pilot.add_argument("--seed", type=int)
    collect_pilot.add_argument(
        "--max-new-games",
        type=int,
        help="bound work in this invocation while retaining the total target",
    )
    collect_pilot.add_argument(
        "--parallel-games",
        type=int,
        help=(
            "concurrent games in each model batch; defaults to "
            "rollout.parallel_games_per_rank"
        ),
    )
    collect_pilot.add_argument("--replay-tolerance", type=float, default=2e-4)
    collect_pilot.set_defaults(handler=command_collect_policy_pilot)

    analyze_pilot = subparsers.add_parser(
        "analyze-policy-pilot",
        help="summarize per-seat outcomes and movement without loading model or engine",
    )
    analyze_pilot.add_argument("--input", required=True, type=Path)
    analyze_pilot.add_argument("--top-k", type=int, default=10)
    analyze_pilot.set_defaults(handler=command_analyze_policy_pilot)

    validate_artifact = subparsers.add_parser(
        "validate-policy-artifact",
        help="replay a saved model match through the authoritative probability path",
    )
    validate_artifact.add_argument("--config", required=True, type=Path)
    validate_artifact.add_argument("--input", required=True, type=Path)
    validate_artifact.add_argument("--replay-tolerance", type=float, default=2e-4)
    validate_artifact.set_defaults(handler=command_validate_policy_artifact)

    backward = subparsers.add_parser(
        "validate-policy-backward", help="one saved-action backward probe; no optimizer step",
    )
    backward.add_argument("--config", required=True, type=Path)
    backward.add_argument("--input", required=True, type=Path)
    backward.add_argument("--joint-step", type=int, default=3)
    backward.add_argument("--replay-tolerance", type=float, default=2e-4)
    backward.set_defaults(handler=command_validate_policy_backward)

    batch_backward = subparsers.add_parser(
        "validate-policy-batch-backward",
        help="accumulate saved-action gradients with one graph resident at a time",
    )
    batch_backward.add_argument("--config", required=True, type=Path)
    batch_backward.add_argument("--input", required=True, type=Path)
    batch_backward.add_argument("--actions", type=int, default=4)
    batch_backward.add_argument("--replay-tolerance", type=float, default=2e-4)
    batch_backward.set_defaults(handler=command_validate_policy_batch_backward)

    history = subparsers.add_parser(
        "show-action-history",
        help="show recorded generated actions without loading the model or engine",
    )
    history.add_argument("--input", required=True, type=Path)
    history.add_argument("--seat", type=int, choices=range(4))
    history.add_argument("--start", type=int, default=0)
    history.add_argument("--limit", type=int)
    history.add_argument("--json", action="store_true")
    history.add_argument("--include-prompts", action="store_true")
    history.set_defaults(handler=command_show_action_history)

    checkpoint = subparsers.add_parser(
        "validate-checkpoint-roundtrip",
        help="atomically save and exactly restore an adapter-only checkpoint",
    )
    checkpoint.add_argument("--config", required=True, type=Path)
    checkpoint.add_argument("--input", required=True, type=Path)
    checkpoint.add_argument("--output", required=True, type=Path)
    checkpoint.add_argument("--joint-step", type=int, default=3)
    checkpoint.add_argument("--replay-tolerance", type=float, default=2e-4)
    checkpoint.set_defaults(handler=command_validate_checkpoint_roundtrip)

    optimizer_checkpoint = subparsers.add_parser(
        "validate-optimizer-checkpoint",
        help="require exact continuation from a checkpoint with non-empty Adam state",
    )
    optimizer_checkpoint.add_argument("--config", required=True, type=Path)
    optimizer_checkpoint.add_argument("--input", required=True, type=Path)
    optimizer_checkpoint.add_argument("--output", required=True, type=Path)
    optimizer_checkpoint.add_argument("--joint-step", type=int, default=3)
    optimizer_checkpoint.add_argument("--replay-tolerance", type=float, default=2e-4)
    optimizer_checkpoint.set_defaults(handler=command_validate_optimizer_checkpoint)

    diagnose_replay = subparsers.add_parser(
        "diagnose-policy-replay",
        help="compare cached and full-sequence replay for a saved match",
    )
    diagnose_replay.add_argument("--config", required=True, type=Path)
    diagnose_replay.add_argument("--input", required=True, type=Path)
    diagnose_replay.add_argument("--top-k", type=int, default=10)
    diagnose_replay.set_defaults(handler=command_diagnose_policy_replay)

    train = subparsers.add_parser("train", help="explicitly start sequential self-play training")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--artifact-dir", required=True, type=Path)
    train.add_argument("--updates", required=True, type=int)
    train.add_argument("--acknowledge-training", action="store_true")
    train.set_defaults(handler=command_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
