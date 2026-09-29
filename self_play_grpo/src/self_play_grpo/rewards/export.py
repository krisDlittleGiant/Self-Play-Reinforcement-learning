"""Export a reward-training dataset from a verified, complete rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from self_play_grpo.rewards.dataset import outcome_examples, write_reward_examples
from self_play_grpo.rollouts.pilot import (
    file_sha256,
    read_pilot_manifest,
)
from self_play_grpo.rollouts.schema import read_matches_jsonl
from self_play_grpo.training.coordinator import PolicyDescriptor, verify_completed_batch


def export_verified_outcome_dataset(
    rollout_root: str | Path,
    output_jsonl: str | Path,
    *,
    config: Mapping[str, Any],
    policy: PolicyDescriptor,
    expected_games: int,
) -> dict[str, Any]:
    """Fail before writing unless all source matches and credits validate.

    The engine outcome is a supervised target, not a replacement for the
    authoritative reward used by the current GRPO trainer.
    """

    root = Path(rollout_root)
    output = Path(output_jsonl)
    if output.exists():
        raise FileExistsError(f"Reward dataset already exists: {output}")
    receipt = verify_completed_batch(
        root, config=config, policy=policy, expected_games=expected_games
    )
    manifest = read_pilot_manifest(root)
    examples = []
    for entry in manifest.matches:
        path = root / entry.path
        if file_sha256(path) != entry.sha256:
            raise ValueError(f"Reward source match changed: {entry.path}")
        matches = read_matches_jsonl(path)
        if len(matches) != 1 or matches[0].game_id != entry.game_id:
            raise ValueError(f"Reward source match identity changed: {entry.path}")
        examples.extend(
            outcome_examples(
                matches[0], source_manifest_sha256=receipt.manifest_sha256
            )
        )
    if file_sha256(root / "manifest.json") != receipt.manifest_sha256:
        raise ValueError("Reward source manifest changed during export")
    counts = write_reward_examples(output, examples)
    return {
        "schema_version": 1,
        "status": "complete",
        "source_manifest_sha256": receipt.manifest_sha256,
        "source_policy_version": policy.version,
        "source_adapter_sha256": policy.adapter_sha256,
        "games": len(receipt.game_ids),
        "examples": len(examples),
        "examples_by_split": counts,
        "dataset_sha256": file_sha256(output),
        "output": str(output),
    }
