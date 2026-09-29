"""Frozen-adapter four-rank rollout shard for a future D5 cycle.

Preparation and aggregation are CPU-only. ``collect_rank`` is an explicit HPU
operation and is never invoked at import time. Four independent workers write
complete, non-overlapping match shards; only the coordinator aggregates them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from self_play_grpo.config import ExperimentConfig, load_config
from self_play_grpo.rollouts.distributed import (
    DistributedRankReport,
    ordered_report_entries,
    rank_match_indices,
    read_rank_report,
    write_rank_report,
)
from self_play_grpo.rollouts.pilot import (
    PilotManifest,
    canonical_sha256,
    directory_sha256,
    make_match_entry,
    model_samples,
    pilot_game_id,
    read_and_replay_pilot_match,
    read_pilot_manifest,
    validate_registered_matches,
    write_pilot_manifest,
    write_pilot_match,
)
from self_play_grpo.training.coordinator import PolicyDescriptor


ROLL_OUT_RANKS = 4
GAMES_PER_RANK = 16
PARALLEL_GAMES = 2


def _require_profile(config: ExperimentConfig) -> None:
    if config.rollout.games_per_update != 64:
        raise ValueError("D5 rollout worker requires 64 games per update")
    if config.rollout.parallel_games_per_rank != PARALLEL_GAMES:
        raise ValueError("D5 rollout worker requires two active games per rank")
    if (config.training.reward_mode != "outcome" or config.training.group_size != 4
            or config.environment.players != 4):
        raise ValueError("D5 rollout worker requires four-seat outcome credit")
    if config.training.optimizer_epochs_per_batch != 1:
        raise ValueError("D5 rollout worker requires one optimizer epoch per batch")


def _require_production_policy(policy: PolicyDescriptor) -> None:
    if policy.run_kind != "production":
        raise ValueError("D5 rollout worker requires a production policy")


def _safe_adapter_source(source: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Adapter source must be a real directory")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Adapter source must not contain symlinks")


def prepare_frozen_batch(
    output: str | Path,
    *,
    config: ExperimentConfig,
    source_adapter: str | Path,
    policy: PolicyDescriptor,
    base_seed: int,
    replay_tolerance: float,
) -> PilotManifest:
    """Atomically publish an empty 64-game manifest and frozen adapter copy."""

    _require_profile(config)
    _require_production_policy(policy)
    if base_seed < 0 or not math.isfinite(replay_tolerance) or replay_tolerance <= 0:
        raise ValueError("D5 seed and replay tolerance must be valid")
    if policy.config_sha256 != canonical_sha256(config.to_dict()):
        raise ValueError("D5 policy configuration differs from the rollout config")
    if policy.model_revision != config.model.revision:
        raise ValueError("D5 policy model revision differs from the rollout config")
    source = Path(source_adapter)
    _safe_adapter_source(source)
    if directory_sha256(source) != policy.adapter_sha256:
        raise ValueError("Source adapter content differs from the policy descriptor")
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"D5 rollout output must be new: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}-preparing-") as name:
        staging = Path(name)
        shutil.copytree(source, staging / "policy_adapter", symlinks=False)
        if directory_sha256(staging / "policy_adapter") != policy.adapter_sha256:
            raise ValueError("Copied adapter content differs from source")
        (staging / "matches").mkdir()
        manifest = PilotManifest.create(
            config=config.to_dict(), adapter_sha256=policy.adapter_sha256,
            policy_version=policy.version, base_seed=base_seed,
            target_games=ROLL_OUT_RANKS * GAMES_PER_RANK,
            replay_tolerance=replay_tolerance,
        )
        (staging / "policy_descriptor.json").write_text(
            json.dumps(asdict(policy), indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        write_pilot_manifest(staging, manifest)
        staging.replace(target)
    return manifest


def _validate_prepared(root: Path, config: ExperimentConfig, policy: PolicyDescriptor) -> PilotManifest:
    _require_profile(config)
    _require_production_policy(policy)
    manifest = read_pilot_manifest(root)
    recorded_policy = PolicyDescriptor(**json.loads(
        (root / "policy_descriptor.json").read_text(encoding="utf-8")
    ))
    if recorded_policy != policy:
        raise ValueError("D5 prepared policy descriptor changed")
    if manifest.matches:
        raise ValueError("D5 rank collection requires an unpublished empty manifest")
    if (
        manifest.config_sha256 != canonical_sha256(config.to_dict())
        or manifest.config_sha256 != policy.config_sha256
        or manifest.policy_version != policy.version
        or manifest.adapter_sha256 != policy.adapter_sha256
        or manifest.target_games != 64
    ):
        raise ValueError("D5 prepared batch differs from the frozen policy")
    if directory_sha256(root / "policy_adapter") != policy.adapter_sha256:
        raise ValueError("D5 prepared adapter content changed")
    return manifest


def _worker_binding(rank: int) -> int:
    if rank not in range(ROLL_OUT_RANKS):
        raise ValueError("Rollout rank must be in [0, 4)")
    if os.environ.get("SP_GRPO_ROLE") != "rollout" or os.environ.get("SP_GRPO_RANK") != str(rank):
        raise RuntimeError("D5 rollout worker role/rank environment differs")
    raw_module = os.environ.get("SP_GRPO_MODULE_ID")
    if raw_module is None or os.environ.get("HLS_MODULE_ID") != raw_module:
        raise RuntimeError("D5 rollout physical module binding differs")
    visible = os.environ.get("HABANA_VISIBLE_MODULES")
    if visible is None or len(visible.split(",")) != ROLL_OUT_RANKS:
        raise RuntimeError("D5 rollout worker requires a four-module visibility map")
    try:
        modules = tuple(int(part) for part in visible.split(","))
    except ValueError as exc:
        raise RuntimeError("D5 rollout visibility map is malformed") from exc
    if len(set(modules)) != ROLL_OUT_RANKS or str(modules[rank]) != raw_module:
        raise RuntimeError("D5 rollout module mapping differs from rank")
    return modules[rank]


def _write_identity(root: Path, *, rank: int, module_id: int, parameter_sha256: str) -> Path:
    target = root / "ranks" / f"rank-{rank:03d}-identity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "role": "rollout", "rank": rank,
        "module_id": module_id, "parameter_sha256": parameter_sha256,
    }
    # An interrupted write leaves an invalid report; a retry must use a new
    # batch directory rather than silently overwrite identity evidence.
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def collect_rank(
    root: str | Path, *, config: ExperimentConfig,
    policy_descriptor: PolicyDescriptor, rank: int,
) -> DistributedRankReport:
    """Explicit per-HPU collection of exactly sixteen complete games."""

    from self_play_grpo.envs.quoridor import QuoridorEnv
    from self_play_grpo.policies.llm import ConstrainedLLMPolicy, max_behavior_replay_error
    from self_play_grpo.rollouts.collector import BatchedMatchCollector
    from self_play_grpo.rollouts.pilot import restore_initial_adapter
    from self_play_grpo.training.identity import trainable_parameter_sha256

    module_id = _worker_binding(rank)
    root_path = Path(root)
    manifest = _validate_prepared(root_path, config, policy_descriptor)
    report_path = root_path / "ranks" / f"rank-{rank:03d}.json"
    identity_path = root_path / "ranks" / f"rank-{rank:03d}-identity.json"
    if report_path.exists() or identity_path.exists():
        raise FileExistsError("D5 rollout rank already has published evidence")
    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    restore_initial_adapter(policy.model, root_path / "policy_adapter", manifest.adapter_sha256)
    policy.model.eval()
    parameter_sha256 = trainable_parameter_sha256(policy.model)
    collector = BatchedMatchCollector(
        collect_progress=True,
        proxy_temperature=config.process_extension.proxy_temperature,
    )
    indices = rank_match_indices(rank, ROLL_OUT_RANKS, GAMES_PER_RANK)
    entries = []
    for offset in range(0, GAMES_PER_RANK, PARALLEL_GAMES):
        batch_indices = indices[offset:offset + PARALLEL_GAMES]
        seeds = tuple(manifest.base_seed + index for index in batch_indices)
        game_ids = tuple(
            pilot_game_id(manifest.policy_version, index, seed)
            for index, seed in zip(batch_indices, seeds)
        )
        matches = collector.collect(
            tuple(QuoridorEnv(config.environment) for _ in batch_indices),
            policy, game_ids=game_ids, seeds=seeds,
            policy_version=manifest.policy_version,
        )
        if len(matches) != len(batch_indices):
            raise RuntimeError("D5 batched collector returned the wrong game count")
        for index, match in zip(batch_indices, matches):
            path = write_pilot_match(root_path, index, match)
            restored, _ = read_and_replay_pilot_match(root_path, manifest, index)
            if restored.to_json() != match.to_json():
                raise RuntimeError("D5 saved match did not round-trip exactly")
            error = max_behavior_replay_error(policy.model, model_samples(restored))
            if not math.isfinite(error) or error > manifest.replay_tolerance:
                raise RuntimeError(f"D5 behavior replay failed for game {index}: {error}")
            entries.append(make_match_entry(
                index=index, match=restored, path=path,
                max_abs_log_prob_error=error,
            ))
    report = DistributedRankReport(
        rank=rank, local_rank=rank, world_size=ROLL_OUT_RANKS,
        games_per_rank=GAMES_PER_RANK, policy_version=manifest.policy_version,
        adapter_sha256=manifest.adapter_sha256, matches=tuple(entries),
    )
    report.validate(manifest)
    _write_identity(root_path, rank=rank, module_id=module_id,
                    parameter_sha256=parameter_sha256)
    write_rank_report(root_path, report, manifest)
    return report


def aggregate_rank_shards(
    root: str | Path, *, config: ExperimentConfig,
    policy: PolicyDescriptor, expected_modules: tuple[int, int, int, int],
) -> PilotManifest:
    """CPU-only global publication after all four shard/identity reports exist."""

    if len(expected_modules) != 4 or len(set(expected_modules)) != 4:
        raise ValueError("D5 aggregation needs four distinct rollout modules")
    root_path = Path(root)
    manifest = _validate_prepared(root_path, config, policy)
    reports = []
    identities = []
    for rank in range(ROLL_OUT_RANKS):
        reports.append(read_rank_report(root_path, rank, manifest))
        path = root_path / "ranks" / f"rank-{rank:03d}-identity.json"
        identity = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(identity, dict)
            or set(identity) != {"schema_version", "role", "rank", "module_id", "parameter_sha256"}
            or identity["schema_version"] != 1
            or identity["role"] != "rollout"
            or identity["rank"] != rank
            or identity["module_id"] != expected_modules[rank]
            or not isinstance(identity["parameter_sha256"], str)
            or len(identity["parameter_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in identity["parameter_sha256"])
        ):
            raise ValueError(f"D5 rollout rank {rank} identity differs")
        identities.append(identity["parameter_sha256"])
    if len(set(identities)) != 1:
        raise ValueError("D5 rollout workers loaded different trainable tensor values")
    entries = ordered_report_entries(
        reports, world_size=ROLL_OUT_RANKS, games_per_rank=GAMES_PER_RANK,
    )
    for entry in entries:
        manifest.append(entry)
    validate_registered_matches(root_path, manifest)
    if manifest.summary()["illegal_action_substitutions"] != 0:
        raise ValueError("D5 rollout contains illegal-action substitutions")
    if manifest.summary()["max_abs_log_prob_error"] > manifest.replay_tolerance:
        raise ValueError("D5 rollout replay tolerance was exceeded")
    write_pilot_manifest(root_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D5 frozen-policy rollout shard")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--policy-descriptor", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data: dict[str, Any] = json.loads(args.policy_descriptor.read_text(encoding="utf-8"))
    policy = PolicyDescriptor(**data)
    report = collect_rank(args.root, config=config, policy_descriptor=policy, rank=args.rank)
    print(json.dumps({
        "status": "rank_complete", "rank": args.rank,
        "games": len(report.matches), "policy_version": report.policy_version,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
