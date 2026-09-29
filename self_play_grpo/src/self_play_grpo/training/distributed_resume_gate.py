"""Bounded D4 distributed checkpoint continuation gate.

This module never runs at import time. It uses one recorded action per trainer
rank and synthetic credit only for the validation gate, never for production.
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from self_play_grpo.config import load_config
from self_play_grpo.distributed import (
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
    canonical_sha256,
    read_and_replay_pilot_match,
    read_pilot_manifest,
    validate_manifest_request,
)
from self_play_grpo.training.distributed import (
    average_trainable_gradients,
    global_source_max_abs_difference,
    trainer_match_indices,
    validate_distributed_training_manifest,
    verify_optimizer_replicas,
    write_rank_update_report,
)
from self_play_grpo.training.distributed_checkpoint import (
    DistributedCheckpointManifest,
    TrainerRankState,
    file_sha256,
    read_distributed_manifest,
    validate_resume_identity,
    verify_checkpoint_files,
    write_rank_rng_record,
)
from self_play_grpo.training.loop import SynchronousTrainer, UpdateMetrics
from self_play_grpo.training.loss import clipped_token_terms


def _collective_phase(torch: Any, dist: Any, runtime: Any, label: str, operation: Any) -> Any:
    """Let every surviving rank observe a local failure before proceeding."""

    error: Exception | None = None
    result: Any = None
    try:
        result = operation()
    except Exception as exc:
        error = exc
    passed = torch.tensor(
        [0.0 if error is not None else 1.0],
        dtype=torch.float32,
        device=torch.device(runtime.device),
    )
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if float(passed.cpu().item()) != 1.0:
        raise RuntimeError(
            f"D4 phase {label} failed"
            + (f" on rank {runtime.rank}: {error}" if error is not None else " on another rank")
        ) from error
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _runtime_identity(torch: Any) -> tuple[tuple[str, str], ...]:
    import peft
    import transformers

    try:
        import habana_frameworks.torch as habana_torch

        habana_version = str(getattr(habana_torch, "__version__", "not-reported"))
    except ImportError:
        habana_version = "not-importable"
    return tuple(sorted({
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "habana": habana_version,
    }.items()))


def _tokenizer_digest(config: Any) -> str:
    if config.model.local_path is None:
        raise ValueError("D4 requires the pinned local model assets")
    root = Path(config.model.local_path)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
    files = {name: file_sha256(root / name) for name in names if (root / name).is_file()}
    if "tokenizer.json" not in files or "tokenizer_config.json" not in files:
        raise ValueError("Pinned tokenizer files are missing from the local model")
    return canonical_sha256(files)


def _code_identity() -> str:
    package = Path(__file__).resolve().parents[1]
    names = (
        "cli.py",
        "distributed.py",
        "policies/llm.py",
        "training/distributed.py",
        "training/distributed_checkpoint.py",
        "training/distributed_resume_gate.py",
        "training/loop.py",
        "training/loss.py",
    )
    return "source-sha256:" + canonical_sha256({
        name: file_sha256(package / name) for name in names
    })


def _optimizer_schema(trainer: SynchronousTrainer) -> str:
    model = trainer.policy.model
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    by_id = {id(parameter): name for name, parameter in trainable}
    groups = []
    for group in trainer.optimizer.param_groups:
        groups.append({
            "parameters": [by_id[id(parameter)] for parameter in group["params"]],
            "options": _jsonable({
                key: value for key, value in group.items() if key != "params"
            }),
        })
    return canonical_sha256({
        "optimizer": type(trainer.optimizer).__qualname__,
        "groups": groups,
    })


def _adapter_schema(trainer: SynchronousTrainer) -> str:
    return canonical_sha256([
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in trainer.policy.model.named_parameters()
        if parameter.requires_grad
    ])


def _module_ids(world_size: int) -> tuple[str, ...]:
    visible = os.environ.get("HABANA_VISIBLE_MODULES")
    result = tuple(visible.split(",")[:world_size]) if visible is not None else tuple(
        str(rank) for rank in range(world_size)
    )
    if len(result) != world_size or len(set(result)) != world_size or not all(result):
        raise ValueError("D4 requires unique complete HPU module bindings")
    return result


def _manifest_template(
    *,
    args: Any,
    config: Any,
    rollout_manifest: Any,
    trainer: SynchronousTrainer,
    runtime: Any,
    torch: Any,
    dist: Any,
) -> DistributedCheckpointManifest:
    module_ids = _module_ids(runtime.world_size)
    ranks = []
    for rank in range(runtime.world_size):
        indices = trainer_match_indices(
            rank, runtime.world_size, len(rollout_manifest.matches)
        )
        entries = tuple(rollout_manifest.matches[index] for index in indices)
        ranks.append(TrainerRankState(
            rank=rank,
            local_rank=rank,
            module_id=module_ids[rank],
            device=f"hpu:{rank}",
            rng_path=f"distributed/rng/rank-{rank:03d}.pt",
            match_indices=indices,
            match_ids=tuple(entry.game_id for entry in entries),
            match_sha256=tuple(entry.sha256 for entry in entries),
        ))
    package = Path(__file__).resolve().parents[1]
    grammar_digest = canonical_sha256({
        "policy_source": file_sha256(package / "policies/llm.py"),
        "engine_revision": config.environment.engine_revision,
        "perspective": config.environment.action_perspective,
        "rollout": config.to_dict()["rollout"],
    })
    backend = str(dist.get_backend())
    return DistributedCheckpointManifest(
        run_kind="validation",
        run_id=args.output.name,
        policy_version=trainer.policy_version,
        update_index=trainer.update_index,
        config_sha256=canonical_sha256(config.to_dict()),
        model_id=config.model.id,
        model_revision=config.model.revision,
        adapter_schema_sha256=_adapter_schema(trainer),
        optimizer_schema_sha256=_optimizer_schema(trainer),
        tokenizer_sha256=_tokenizer_digest(config),
        grammar_sha256=grammar_digest,
        code_identity=_code_identity(),
        runtime_identity=_runtime_identity(torch),
        attention_backend=str(
            getattr(trainer.policy.model.config, "_attn_implementation", "unreported")
        ),
        dtype=config.model.dtype,
        backend=backend,
        trainer_world_size=runtime.world_size,
        source_rollout_manifest_sha256=file_sha256(args.source_rollout / "manifest.json"),
        ranks=tuple(ranks),
        files=(),
    )


def _expected_identity(template: DistributedCheckpointManifest) -> dict[str, Any]:
    return {
        "run_id": template.run_id,
        "policy_version": template.policy_version,
        "update_index": template.update_index,
        "config_sha256": template.config_sha256,
        "model_id": template.model_id,
        "model_revision": template.model_revision,
        "adapter_schema_sha256": template.adapter_schema_sha256,
        "optimizer_schema_sha256": template.optimizer_schema_sha256,
        "tokenizer_sha256": template.tokenizer_sha256,
        "grammar_sha256": template.grammar_sha256,
        "code_identity": template.code_identity,
        "runtime_identity": dict(template.runtime_identity),
        "attention_backend": template.attention_backend,
        "dtype": template.dtype,
        "backend": template.backend,
        "trainer_world_size": template.trainer_world_size,
        "source_rollout_manifest_sha256": template.source_rollout_manifest_sha256,
        "sharding_rule": template.sharding_rule,
        "rank_bindings": [
            {"rank": row.rank, "local_rank": row.local_rank,
             "module_id": row.module_id, "device": row.device}
            for row in template.ranks
        ],
        "match_shards": [
            {"match_indices": list(row.match_indices), "match_ids": list(row.match_ids),
             "match_sha256": list(row.match_sha256)}
            for row in template.ranks
        ],
    }


def _freeze(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _freeze(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_freeze(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze(item, torch) for item in value)
    return copy.deepcopy(value)


def _snapshot(trainer: SynchronousTrainer, metrics: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    return {
        "parameters": {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainer.policy.model.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": _freeze(trainer.optimizer.state_dict(), torch),
        "metrics": _freeze(dict(metrics), torch),
        "update_index": trainer.update_index,
        "policy_version": trainer.policy_version,
    }


def _assert_exact(left: Any, right: Any, torch: Any, path: str = "snapshot") -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right) or not torch.equal(left, right):
            raise AssertionError(f"{path} tensor differs")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise AssertionError(f"{path} type differs")
        if set(left) != set(right):
            raise AssertionError(f"{path} keys differ")
        for key in sorted(left, key=str):
            _assert_exact(left[key], right[key], torch, f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"{path} sequence differs")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_exact(a, b, torch, f"{path}[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise AssertionError(f"{path} scalar differs: {left!r} != {right!r}")


def _controlled_step(
    trainer: SynchronousTrainer,
    sample: Any,
    *,
    config: Any,
    runtime: Any,
    torch: Any,
    dist: Any,
) -> dict[str, float | int]:
    """One real optimizer step on one recorded action per rank."""

    model = trainer.policy.model
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.training.kl_beta != 0.0:
        raise ValueError("D4 controlled gate requires the validated no-KL baseline")
    trainer.optimizer.zero_grad(set_to_none=True)

    def local_backward() -> dict[str, float | int]:
        with checkpoint_qwen3_attention(model):
            current = constrained_log_probs_batched_shape(model, sample)
            old = torch.tensor(
                sample.behavior_log_probs, dtype=current.dtype, device=current.device
            )
            ownership = torch.tensor(
                sample.loss_mask, dtype=current.dtype, device=current.device
            )
            terms, ratios = clipped_token_terms(
                current, old, 1.0, ownership, config.training.clip_epsilon
            )
            loss = -terms.sum() / float(config.training.loss_normalizer_per_game)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("D4 controlled loss is non-finite")
            active = ownership > 0
            if not bool(active.any().item()):
                raise ValueError("D4 selected action owns no tokens")
            clip = (
                (ratios[active] < 1.0 - config.training.clip_epsilon)
                | (ratios[active] > 1.0 + config.training.clip_epsilon)
            ).float().mean()
            values = {
                "loss": float(loss.detach().float().cpu().item()),
                "mean_ratio": float(ratios[active].detach().float().mean().cpu().item()),
                "clip_fraction": float(clip.detach().cpu().item()),
                "owned_tokens": int(active.sum().cpu().item()),
                "replay_error": float(torch.max(torch.abs(current.detach() - old)).float().cpu().item()),
            }
            if not all(math.isfinite(float(values[key])) for key in ("loss", "mean_ratio", "clip_fraction", "replay_error")):
                raise FloatingPointError("D4 controlled metrics are non-finite")
            loss.backward()
            return values

    local = local_backward()
    average_trainable_gradients(model, dist, world_size=runtime.world_size)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, config.training.max_grad_norm, error_if_nonfinite=True
    )
    norm = float(grad_norm.detach().float().cpu().item())
    norm_probe = torch.tensor([norm], dtype=torch.float32, device=torch.device(runtime.device))
    norm_difference = global_source_max_abs_difference(torch, dist, norm_probe)
    if norm <= 0.0 or norm_difference != 0.0:
        raise RuntimeError(
            f"D4 gradient contract failed: norm={norm}, rank_difference={norm_difference}"
        )
    before_parameters = [parameter.detach().cpu().clone() for parameter in parameters]
    trainer.optimizer.step()
    if hasattr(torch, "hpu") and hasattr(torch.hpu, "synchronize"):
        torch.hpu.synchronize()
    changed_parameter_tensors = sum(
        not torch.equal(before, parameter.detach().cpu())
        for before, parameter in zip(before_parameters, parameters)
    )
    if changed_parameter_tensors == 0:
        raise RuntimeError("D4 optimizer step changed no trainable parameter tensors")
    replicas = verify_optimizer_replicas(
        torch, dist, parameters, trainer.optimizer, device=torch.device(runtime.device)
    )
    if any(float(value) != 0.0 for key, value in replicas.items() if key.startswith("global_max_abs_")):
        raise RuntimeError(f"D4 trainer replicas differ after optimizer step: {replicas}")
    trainer.optimizer.zero_grad(set_to_none=True)
    local["grad_norm"] = norm
    local["changed_parameter_tensors"] = changed_parameter_tensors
    local["optimizer_steps"] = 1
    trainer.update_index += 1
    trainer.policy_version = f"policy-{trainer.update_index:06d}"
    return local


def _draw_rng(torch: Any, device: str) -> dict[str, Any]:
    cpu = torch.rand(8)
    hpu = torch.rand(8, device=torch.device(device)).cpu()
    return {"cpu": cpu.clone(), "hpu": hpu.clone()}


def _prepare_source(args: Any, config: Any, rollout_manifest: Any) -> None:
    if not args.allow_validation_source:
        raise ValueError("D4 requires the explicit --allow-validation-source bridge")
    if args.output.exists():
        raise FileExistsError(f"D4 output must be new: {args.output}")
    if args.joint_step < 0 or args.match_offset < 0 or args.seed < 0:
        raise ValueError("D4 step, match offset, and seed must be nonnegative")
    validate_manifest_request(
        rollout_manifest,
        config=config.to_dict(),
        base_seed=rollout_manifest.base_seed,
        replay_tolerance=rollout_manifest.replay_tolerance,
    )
    state = json.loads((args.source_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("checkpoint_format") != 2 or state.get("validation_only") is not True:
        raise ValueError("D4 bridge requires the D3 validation-only format-2 checkpoint")
    if state.get("update_index") != 1 or state.get("policy_version") != "policy-000001":
        raise ValueError("D4 bridge requires the saved D3 policy-000001 checkpoint")
    if canonical_sha256(state.get("config")) != canonical_sha256(config.to_dict()):
        raise ValueError("D3 checkpoint configuration differs from D4 source rollout")
    summary_path = args.source_checkpoint.parent.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("behavior_adapter_sha256") != rollout_manifest.adapter_sha256:
        raise ValueError("D3 summary does not identify the D2 behavior adapter")
    if Path(summary.get("checkpoint", "")).resolve() != args.source_checkpoint.resolve():
        raise ValueError("D3 summary does not identify the requested source checkpoint")


def run_distributed_checkpoint_resume_gate(args: Any) -> int:
    """Execute the validation-only same-process D4 continuation comparison."""

    import torch

    config = load_config(args.config)
    rollout_manifest = read_pilot_manifest(args.source_rollout)
    _prepare_source(args, config, rollout_manifest)
    runtime = discover_torchrun_runtime(os.environ, expected_world_size=args.expected_world_size)
    validate_distributed_training_manifest(
        rollout_manifest, config=config.to_dict(), world_size=runtime.world_size
    )
    indices = trainer_match_indices(runtime.rank, runtime.world_size, len(rollout_manifest.matches))
    if args.match_offset >= len(indices):
        raise ValueError("--match-offset is outside the rank shard")
    match_index = indices[args.match_offset]
    match, digest = read_and_replay_pilot_match(
        args.source_rollout, rollout_manifest, match_index
    )
    if digest != rollout_manifest.matches[match_index].sha256:
        raise ValueError("D4 selected match digest differs")
    if args.joint_step >= len(match.turns):
        raise ValueError("--joint-step is outside the selected match")
    turn = match.turns[args.joint_step]
    sample = turn.policy_sample
    if not any(len(allowed) > 1 for allowed in sample.allowed_token_ids):
        raise ValueError("D4 selected action has no stochastic token")
    if not any(sample.loss_mask):
        raise ValueError("D4 selected action has no owned token")

    torch, dist = initialize_hccl_process_group(runtime, timeout_seconds=args.timeout_seconds)
    try:
        runtime_report = validate_initialized_hccl(runtime, torch, dist)
        _collective_phase(
            torch, dist, runtime, "create output",
            lambda: args.output.mkdir(parents=True) if runtime.rank == 0 else None,
        )
        policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
        trainer = SynchronousTrainer(config, policy, args.output)
        trainer.load_checkpoint(args.source_checkpoint, allow_validation=True)
        policy.model.train()
        parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
        initial_replicas = verify_optimizer_replicas(
            torch, dist, parameters, trainer.optimizer, device=torch.device(runtime.device)
        )
        if any(float(value) != 0.0 for key, value in initial_replicas.items() if key.startswith("global_max_abs_")):
            raise RuntimeError("D3 source checkpoint did not restore equal trainer replicas")

        # The D3 format-2 checkpoint has only rank-zero RNG. Establish new,
        # documented per-rank streams after loading; this is validation-only.
        torch.manual_seed(args.seed + runtime.rank)
        torch.hpu.manual_seed_all(args.seed + runtime.rank)
        first = _controlled_step(
            trainer, sample, config=config, runtime=runtime, torch=torch, dist=dist
        )
        template = _manifest_template(
            args=args, config=config, rollout_manifest=rollout_manifest,
            trainer=trainer, runtime=runtime, torch=torch, dist=dist,
        )
        template.ranks[runtime.rank].validate()
        staging = args.output / "rank_rng_staging"
        _collective_phase(
            torch, dist, runtime, "stage rank RNG",
            lambda: write_rank_rng_record(staging, template.ranks[runtime.rank]),
        )
        global_counts = torch.tensor(
            [float(first["owned_tokens"]), float(first["loss"]),
             float(first["mean_ratio"]) * float(first["owned_tokens"]),
             float(first["clip_fraction"]) * float(first["owned_tokens"])],
            dtype=torch.float32, device=torch.device(runtime.device),
        )
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        counts = global_counts.cpu().tolist()
        replay_error = torch.tensor(
            [float(first["replay_error"])], dtype=torch.float32,
            device=torch.device(runtime.device),
        )
        dist.all_reduce(replay_error, op=dist.ReduceOp.MAX)
        if counts[0] <= 0.0:
            raise RuntimeError("D4 controlled batch has no owned tokens")
        metrics = UpdateMetrics(
            update=1, policy_version="policy-000001",
            games=0, turns=runtime.world_size, owned_tokens=int(round(counts[0])),
            loss=float(counts[1] / runtime.world_size),
            policy_loss=float(counts[1] / runtime.world_size), kl_loss=0.0,
            mean_ratio_before_step=float(counts[2] / counts[0]),
            clip_fraction_before_step=float(counts[3] / counts[0]),
            replay_max_abs_error=float(replay_error.cpu().item()),
            grad_norm=float(first["grad_norm"]), optimizer_steps=1,
        )
        checkpoint = args.output / "checkpoints" / trainer.policy_version
        _collective_phase(
            torch, dist, runtime, "publish checkpoint",
            lambda: trainer.save_checkpoint(
                metrics, validation_only=True,
                distributed_manifest=template, rank_rng_staging=staging,
            ) if runtime.rank == 0 else None,
        )
        expected = _expected_identity(template)

        def verify_published() -> None:
            published = read_distributed_manifest(checkpoint)
            verify_checkpoint_files(checkpoint, published)
            validate_resume_identity(published, expected=expected, allow_validation=True)

        _collective_phase(torch, dist, runtime, "verify published checkpoint", verify_published)
        checkpoint_digest = file_sha256(checkpoint / "distributed/manifest.json")
        uninterrupted_draws = _draw_rng(torch, runtime.device)
        uninterrupted_metrics = _controlled_step(
            trainer, sample, config=config, runtime=runtime, torch=torch, dist=dist
        )
        uninterrupted = _snapshot(trainer, uninterrupted_metrics, torch)

        _collective_phase(
            torch, dist, runtime, "reload checkpoint",
            lambda: trainer.load_checkpoint(
                checkpoint, allow_validation=True,
                distributed_expected=expected, distributed_rank=runtime.rank,
            ),
        )
        resumed_draws = _draw_rng(torch, runtime.device)
        resumed_metrics = _controlled_step(
            trainer, sample, config=config, runtime=runtime, torch=torch, dist=dist
        )
        resumed = _snapshot(trainer, resumed_metrics, torch)
        def compare_continuation() -> None:
            _assert_exact(uninterrupted_draws, resumed_draws, torch, "rng_draws")
            _assert_exact(uninterrupted, resumed, torch, "continuation")

        _collective_phase(torch, dist, runtime, "compare continuation", compare_continuation)
        report = {
            "status": "ok",
            "rank": runtime.rank,
            "module_id": template.ranks[runtime.rank].module_id,
            "source_match_index": match_index,
            "source_game_id": match.game_id,
            "joint_step": args.joint_step,
            "chosen_label": sample.chosen_label,
            "checkpoint_manifest_sha256": checkpoint_digest,
            "first_step": first,
            "continuation": resumed_metrics,
            "cpu_rng_equal": True,
            "hpu_rng_equal": True,
            "parameters_equal": True,
            "optimizer_equal": True,
            "metrics_equal": True,
        }
        _collective_phase(
            torch, dist, runtime, "write rank report",
            lambda: write_rank_update_report(args.output, runtime.rank, report),
        )

        def write_summary() -> None:
            if runtime.rank != 0:
                return
            reports = [
                json.loads(
                    (args.output / "ranks" / f"rank-{rank:03d}.json").read_text(encoding="utf-8")
                )
                for rank in range(runtime.world_size)
            ]
            if any(row["status"] != "ok" for row in reports):
                raise RuntimeError("D4 rank reports are incomplete")
            summary = {
                "status": "ok",
                "validation_only": True,
                "run_id": template.run_id,
                "source_rollout": str(args.source_rollout),
                "source_checkpoint": str(args.source_checkpoint),
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_sha256": checkpoint_digest,
                "world_size": runtime.world_size,
                "runtime_contract": runtime_report,
                "rank_reports": reports,
            }
            target = args.output / "summary.json"
            temporary = args.output / ".summary.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(target)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

        _collective_phase(torch, dist, runtime, "write summary", write_summary)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
