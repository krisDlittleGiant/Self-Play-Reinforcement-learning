"""Outcome-supervised examples for an optional learned reward predictor.

Engine results remain the authoritative GRPO rewards.  These examples do not
authorize replacing those rewards with model predictions.  Splits are by
complete game, never by turn, to avoid train/evaluation leakage.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from self_play_grpo.rewards.outcome import outcome_advantages
from self_play_grpo.rollouts.schema import MatchRecord


REWARD_DATASET_SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")


def game_split(game_id: str) -> str:
    """Assign every turn in one game to the same stable 80/10/10 split."""

    if not game_id:
        raise ValueError("Reward dataset game_id cannot be empty")
    bucket = int.from_bytes(
        hashlib.sha256(game_id.encode("utf-8")).digest()[:8], "big"
    ) % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"


@dataclass(frozen=True)
class RewardExample:
    """Inputs and labels are intentionally separate to prevent target leakage."""

    game_id: str
    seed: int
    policy_version: str
    joint_step: int
    seat: int
    split: str
    observation: str
    chosen_action_label: str
    terminal_result: float
    outcome_advantage: float
    source_manifest_sha256: str
    schema_version: int = REWARD_DATASET_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != REWARD_DATASET_SCHEMA_VERSION:
            raise ValueError("Unsupported reward dataset schema")
        if self.split != game_split(self.game_id):
            raise ValueError("Reward example split differs from its game ID")
        if self.joint_step < 0 or self.seat not in range(4):
            raise ValueError("Reward example has invalid turn identity")
        if not self.policy_version or not self.observation or not self.chosen_action_label:
            raise ValueError("Reward example is missing its policy or input")
        if not 0.0 <= self.terminal_result <= 1.0:
            raise ValueError("Reward label is outside [0, 1]")
        if not math.isfinite(self.outcome_advantage):
            raise ValueError("Reward advantage must be finite")
        if len(self.source_manifest_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_manifest_sha256
        ):
            raise ValueError("Reward example lacks a source manifest digest")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "features": {
                "observation": self.observation,
                "chosen_action_label": self.chosen_action_label,
            },
            "labels": {
                "terminal_result": self.terminal_result,
                "outcome_advantage": self.outcome_advantage,
            },
            "provenance": {
                "game_id": self.game_id,
                "seed": self.seed,
                "policy_version": self.policy_version,
                "joint_step": self.joint_step,
                "seat": self.seat,
                "split": self.split,
                "source_manifest_sha256": self.source_manifest_sha256,
            },
        }


def outcome_examples(
    match: MatchRecord, *, source_manifest_sha256: str
) -> tuple[RewardExample, ...]:
    """Extract one labeled decision per turn from a complete, checked match."""

    match.validate()
    advantages = outcome_advantages(match.final_results)
    split = game_split(match.game_id)
    examples: list[RewardExample] = []
    for turn in match.turns:
        expected = advantages[turn.seat]
        if turn.credit.outcome_advantage is None or not math.isclose(
            turn.credit.outcome_advantage, expected, abs_tol=1e-9
        ):
            raise ValueError("Rollout outcome advantage differs from engine result")
        if turn.credit.training_advantage is None or not math.isclose(
            turn.credit.training_advantage, expected, abs_tol=1e-9
        ):
            raise ValueError("Rollout training advantage differs from engine result")
        example = RewardExample(
            game_id=match.game_id,
            seed=match.seed,
            policy_version=match.policy_version,
            joint_step=turn.joint_step,
            seat=turn.seat,
            split=split,
            observation=turn.observation,
            chosen_action_label=turn.policy_sample.chosen_label,
            terminal_result=match.final_results[turn.seat],
            outcome_advantage=expected,
            source_manifest_sha256=source_manifest_sha256,
        )
        example.validate()
        examples.append(example)
    return tuple(examples)


def write_reward_examples(path: str | Path, examples: Iterable[RewardExample]) -> dict[str, int]:
    """Write a new JSONL dataset; never overwrite an existing split."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Reward dataset already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = {split: 0 for split in SPLITS}
    games: dict[str, str] = {}
    temporary = target.with_name(f".{target.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"Incomplete reward dataset already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for example in examples:
                example.validate()
                prior_split = games.setdefault(example.game_id, example.split)
                if prior_split != example.split:
                    raise ValueError("One game appears in multiple reward splits")
                handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
                counts[example.split] += 1
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not games:
        target.unlink(missing_ok=True)
        raise ValueError("Reward dataset cannot be empty")
    return counts
