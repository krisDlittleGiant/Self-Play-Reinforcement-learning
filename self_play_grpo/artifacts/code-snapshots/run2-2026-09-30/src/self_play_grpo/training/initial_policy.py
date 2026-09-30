"""CPU-only provenance gate for the first production D5 policy.

Only a recorded frozen pilot adapter can seed policy-000000. This rejects a
D4 synthetic-update checkpoint as a silent production starting point.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from self_play_grpo.config import ExperimentConfig, load_config
from self_play_grpo.rollouts.pilot import (
    canonical_sha256, directory_sha256, file_sha256,
    read_and_replay_pilot_match, read_pilot_manifest,
)
from self_play_grpo.training.coordinator import PolicyDescriptor
from self_play_grpo.training.distributed_resume_gate import _tokenizer_digest


def _compatible_pilot_config(source: dict[str, Any], target: ExperimentConfig) -> None:
    expected = target.to_dict()
    if source.get("model") != expected["model"] or source.get("environment") != expected["environment"]:
        raise ValueError("Initial pilot model or environment differs from the D5 baseline")
    before = dict(source.get("rollout", {}))
    after = dict(expected["rollout"])
    for key in ("games_per_update", "parallel_games_per_rank"):
        before.pop(key, None)
        after.pop(key, None)
    if before != after:
        raise ValueError("Initial pilot decoding contract differs from D5")
    training = source.get("training")
    if not isinstance(training, dict) or training.get("reward_mode") != "outcome":
        raise ValueError("Initial pilot must use the outcome baseline")


def prepare_initial_policy(
    *, pilot_root: str | Path, config: ExperimentConfig, output: str | Path,
) -> tuple[PolicyDescriptor, dict[str, Any]]:
    """Copy an unchanged pilot adapter into a new initial-policy bundle."""

    pilot = Path(pilot_root)
    if pilot.is_symlink() or not pilot.is_dir():
        raise ValueError("Initial pilot root must be a real directory")
    manifest = read_pilot_manifest(pilot)
    if not manifest.policy_version.startswith("pilot:") or not manifest.matches:
        raise ValueError("Initial source needs at least one recorded pilot match")
    _compatible_pilot_config(manifest.config, config)
    source_adapter = pilot / "policy_adapter"
    if (source_adapter.is_symlink() or not source_adapter.is_dir()
            or any(candidate.is_symlink() for candidate in source_adapter.rglob("*"))):
        raise ValueError("Initial pilot adapter may not contain symlinks")
    adapter_digest = directory_sha256(source_adapter)
    if adapter_digest != manifest.adapter_sha256:
        raise ValueError("Initial pilot adapter digest differs from the manifest")
    match, match_digest = read_and_replay_pilot_match(pilot, manifest, 0)
    if match_digest != manifest.matches[0].sha256 or not match.turns:
        raise ValueError("Initial pilot match provenance differs")
    adapter_config = json.loads((source_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base = adapter_config.get("base_model_name_or_path")
    if (adapter_config.get("peft_type") != "LORA"
            or adapter_config.get("r") != config.model.lora_rank
            or adapter_config.get("lora_alpha") != config.model.lora_alpha
            or adapter_config.get("lora_dropout") != config.model.dropout
            or base not in {config.model.id, config.model.local_path}):
        raise ValueError("Initial pilot LoRA configuration differs from the pinned model")
    package = Path(__file__).resolve().parents[1]
    grammar_digest = canonical_sha256({
        "policy_source": file_sha256(package / "policies" / "llm.py"),
        "engine_revision": config.environment.engine_revision,
        "perspective": config.environment.action_perspective,
        "rollout": config.to_dict()["rollout"],
    })
    descriptor = PolicyDescriptor(
        version="policy-000000", update_index=0,
        adapter_sha256=adapter_digest,
        config_sha256=canonical_sha256(config.to_dict()),
        model_revision=config.model.revision,
        tokenizer_sha256=_tokenizer_digest(config),
        grammar_sha256=grammar_digest, run_kind="production",
    )
    provenance = {
        "status": "recorded_pilot_initial_policy",
        "source_pilot_manifest_sha256": file_sha256(pilot / "manifest.json"),
        "source_pilot_policy_version": manifest.policy_version,
        "source_first_match_sha256": match_digest,
        "adapter_sha256": adapter_digest,
        "target_config_sha256": descriptor.config_sha256,
    }
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Initial policy bundle output must be new")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}-preparing-") as name:
        staging = Path(name)
        shutil.copytree(source_adapter, staging / "policy_adapter")
        if directory_sha256(staging / "policy_adapter") != adapter_digest:
            raise ValueError("Initial policy adapter copy differs from source")
        (staging / "policy_descriptor.json").write_text(
            json.dumps(asdict(descriptor), indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        staging.replace(target)
    return descriptor, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare recorded pilot policy-000000 for D5")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    descriptor, provenance = prepare_initial_policy(
        pilot_root=args.pilot_root, config=load_config(args.config), output=args.output,
    )
    print(json.dumps({"descriptor": asdict(descriptor), "provenance": provenance}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
