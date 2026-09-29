"""Fail-closed trainer-to-rollout adapter refresh for the first D5 update.

No HPU is acquired on import. A rollout worker invokes ``refresh_rank`` only
after a production checkpoint and trainer probe have been published. Refresh
does not collect games; the next rollout phase must require all four reports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from self_play_grpo.config import load_config
from self_play_grpo.policies.llm import ConstrainedLLMPolicy, constrained_log_probs_batched_shape
from self_play_grpo.rollouts.pilot import (
    canonical_sha256, directory_sha256, file_sha256,
    read_and_replay_pilot_match, read_pilot_manifest, restore_initial_adapter,
)
from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.d5_preflight import verify_d5_launch_prerequisites
from self_play_grpo.training.distributed_checkpoint import (
    read_distributed_manifest, verify_checkpoint_files,
)
from self_play_grpo.training.identity import trainable_parameter_sha256
from self_play_grpo.training.rollout_worker import _worker_binding


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"D5 {name} is not a SHA-256 digest")
    return value


def _module_list(raw: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(part) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError("D5 module list is malformed") from exc
    if len(values) != 4 or len(set(values)) != 4 or any(value < 0 for value in values):
        raise ValueError("D5 module list must contain four distinct non-negative IDs")
    return values  # type: ignore[return-value]


def source_sample(root: Path, *, game_index: int = 0, joint_step: int = 0) -> tuple[Any, str]:
    """Read the exact recorded-shape action used as the refresh probe."""

    manifest = read_pilot_manifest(root)
    if game_index < 0 or game_index >= len(manifest.matches):
        raise ValueError("D5 refresh probe game index is outside the manifest")
    entry = manifest.matches[game_index]
    if entry.index != game_index:
        raise ValueError("D5 refresh probe manifest order differs")
    match, digest = read_and_replay_pilot_match(root, manifest, game_index)
    if digest != entry.sha256:
        raise ValueError("D5 refresh probe match digest differs")
    if joint_step < 0 or joint_step >= len(match.turns):
        raise ValueError("D5 refresh probe joint step is outside the match")
    return match.turns[joint_step].policy_sample, digest


def policy_probe(model: Any, root: Path) -> dict[str, Any]:
    """Evaluate one immutable source action under the *updated* policy."""

    import torch

    sample, digest = source_sample(root)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            values = constrained_log_probs_batched_shape(model, sample).detach().float().cpu().tolist()
    finally:
        model.train(was_training)
    if (len(values) != len(sample.completion_token_ids)
            or not values or any(not math.isfinite(float(value)) for value in values)):
        raise ValueError("D5 refresh probe returned invalid probabilities")
    return {
        "game_index": 0,
        "joint_step": 0,
        "match_sha256": digest,
        "completion_token_ids": list(sample.completion_token_ids),
        "log_probs": [float(value) for value in values],
    }


def validate_probe(probe: Any) -> dict[str, Any]:
    if not isinstance(probe, dict) or set(probe) != {
        "game_index", "joint_step", "match_sha256", "completion_token_ids", "log_probs",
    }:
        raise ValueError("D5 trainer refresh probe schema differs")
    if probe["game_index"] != 0 or probe["joint_step"] != 0:
        raise ValueError("D5 trainer refresh probe target differs")
    _digest(probe["match_sha256"], "probe match")
    tokens, values = probe["completion_token_ids"], probe["log_probs"]
    if (not isinstance(tokens, list) or not tokens
            or any(type(value) is not int or value < 0 for value in tokens)
            or not isinstance(values, list) or len(values) != len(tokens)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
        raise ValueError("D5 trainer refresh probe values are invalid")
    return probe


def compare_probe(expected: dict[str, Any], actual: dict[str, Any], tolerance: float) -> float:
    validate_probe(expected)
    validate_probe(actual)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("D5 refresh tolerance must be finite and positive")
    for key in ("game_index", "joint_step", "match_sha256", "completion_token_ids"):
        if expected[key] != actual[key]:
            raise ValueError(f"D5 refresh probe {key} differs")
    error = max(abs(left - right) for left, right in zip(expected["log_probs"], actual["log_probs"]))
    if error > tolerance:
        raise RuntimeError(f"D5 rollout refresh probability mismatch {error:.6g} exceeds {tolerance:.6g}")
    return error


def refresh_rank(args: Any) -> dict[str, Any]:
    """Load policy-000001 on one rollout HPU and publish one verified report."""

    config = load_config(args.config)
    rollout_modules = _module_list(args.rollout_modules)
    trainer_modules = _module_list(args.trainer_modules)
    layout = RoleLayout(rollout_modules, trainer_modules)
    verify_d5_launch_prerequisites(
        config_path=args.config, two_rank_d4_summary=args.d4_two_summary,
        four_rank_d4_summary=args.d4_four_summary, layout=layout,
        worker_role="rollout",
    )
    module_id = _worker_binding(args.rank)
    if module_id != rollout_modules[args.rank]:
        raise RuntimeError("D5 refresh module differs from requested layout")
    if args.output.resolve() in (args.rollout_root.resolve(), args.trainer_output.resolve()):
        raise ValueError("D5 refresh evidence needs its own output root")
    target = args.output / f"rank-{args.rank:03d}.json"
    if target.exists():
        raise FileExistsError("D5 refresh rank report already exists")

    rollout_manifest = read_pilot_manifest(args.rollout_root)
    source_digest = file_sha256(args.rollout_root / "manifest.json")
    if (len(rollout_manifest.matches) != 64
            or rollout_manifest.config_sha256 != canonical_sha256(config.to_dict())):
        raise ValueError("D5 refresh source is not the admitted 64-game batch")
    summary = json.loads((args.trainer_output / "trainer_summary.json").read_text(encoding="utf-8"))
    checkpoint = args.trainer_output / "trainer" / "checkpoints" / "policy-000001"
    if (summary.get("status") != "updated_checkpoint_committed_refresh_pending"
            or Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or summary.get("source_manifest_sha256") != source_digest):
        raise ValueError("D5 trainer summary does not bind this source/checkpoint")
    expected_adapter = _digest(summary.get("checkpoint_adapter_sha256"), "adapter")
    expected_parameters = _digest(summary.get("parameter_sha256"), "parameters")
    expected_checkpoint = _digest(summary.get("checkpoint_manifest_sha256"), "checkpoint")
    expected_probe = validate_probe(summary.get("policy_probe"))
    sample, match_digest = source_sample(args.rollout_root)
    if (expected_probe["match_sha256"] != match_digest
            or expected_probe["completion_token_ids"] != list(sample.completion_token_ids)):
        raise ValueError("D5 trainer probe is not bound to the rollout source action")
    if checkpoint.is_symlink() or file_sha256(checkpoint / "distributed" / "manifest.json") != expected_checkpoint:
        raise ValueError("D5 checkpoint identity differs from trainer summary")
    checkpoint_manifest = read_distributed_manifest(checkpoint)
    verify_checkpoint_files(checkpoint, checkpoint_manifest)
    if (checkpoint_manifest.run_kind != "production"
            or checkpoint_manifest.policy_version != "policy-000001"
            or checkpoint_manifest.update_index != 1
            or checkpoint_manifest.config_sha256 != canonical_sha256(config.to_dict())
            or checkpoint_manifest.model_revision != config.model.revision
            or checkpoint_manifest.source_rollout_manifest_sha256 != source_digest):
        raise ValueError("D5 checkpoint does not match the intended production update")
    if directory_sha256(checkpoint / "adapter") != expected_adapter:
        raise ValueError("D5 checkpoint adapter digest differs")

    policy = ConstrainedLLMPolicy.load(config.model, config.rollout)
    restore_initial_adapter(policy.model, checkpoint / "adapter", expected_adapter)
    policy.model.eval()
    parameter_digest = trainable_parameter_sha256(policy.model)
    if parameter_digest != expected_parameters:
        raise RuntimeError("D5 refreshed rollout adapter tensor values differ from trainer")
    actual_probe = policy_probe(policy.model, args.rollout_root)
    error = compare_probe(expected_probe, actual_probe, rollout_manifest.replay_tolerance)
    report = {
        "status": "refresh_verified", "rank": args.rank, "module_id": module_id,
        "policy_version": "policy-000001", "source_manifest_sha256": source_digest,
        "checkpoint_manifest_sha256": expected_checkpoint,
        "adapter_sha256": expected_adapter, "parameter_sha256": parameter_digest,
        "probe_max_abs_error": error,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify D5 policy-000001 on one rollout HPU")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--trainer-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--rollout-modules", required=True)
    parser.add_argument("--trainer-modules", required=True)
    parser.add_argument("--d4-two-summary", required=True, type=Path)
    parser.add_argument("--d4-four-summary", required=True, type=Path)
    print(json.dumps(refresh_rank(parser.parse_args(argv)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
