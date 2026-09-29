"""CPU-only per-batch leading indicators of policy behaviour.

These read persisted rollout, pilot, or evaluation match files and never load
a model or the game engine. They are cheap enough to compute for every
training batch and show behavioural drift long before a tournament can.

The mean negative log-probability of sampled actions is an unbiased estimate
of the constrained policy's action entropy averaged over visited states.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from self_play_grpo.rollouts.analysis import goal_progress_delta
from self_play_grpo.rollouts.schema import MatchRecord, read_matches_jsonl


_LABEL = re.compile(r"\b(?:MOVE|WALL)_[A-Z]\d+[HV]?\b")
_MENU_HEADERS = ("Legal actions:\n", "Legal pawn moves:\n")


def listed_labels(observation: str) -> list[str]:
    """Return menu labels in prompt order, ignoring the recent-move history."""

    starts = [observation.find(header) for header in _MENU_HEADERS]
    starts = [start for start in starts if start >= 0]
    if not starts:
        raise ValueError("Observation has no legal-action menu")
    return _LABEL.findall(observation[min(starts):])


def _fraction(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def batch_indicators(matches: Sequence[MatchRecord] | Iterable[MatchRecord]) -> dict[str, Any]:
    """Summarize model-sampled turns and complete-match outcomes."""

    records = list(matches)
    if not records:
        raise ValueError("At least one match is required")
    turns = moves = walls = forward = backward = lateral = 0
    first_listed = first_listed_move = 0
    position_histogram: Counter[str] = Counter()
    action_log_probs: list[float] = []
    distance_gain = 0
    opening: Counter[str] = Counter()
    wins = [0, 0, 0, 0]
    termination: Counter[str] = Counter()
    lengths: list[int] = []
    for match in records:
        termination[match.termination_reason] += 1
        lengths.append(len(match.turns))
        for seat, result in enumerate(match.final_results):
            if result == 1.0:
                wins[seat] += 1
        for turn in match.turns:
            sample = turn.policy_sample
            if not sample.completion_token_ids:
                continue  # A scripted opponent seat in an evaluation match.
            turns += 1
            label = sample.chosen_label
            action_log_probs.append(math.fsum(sample.behavior_log_probs))
            menu = listed_labels(turn.observation)
            position = menu.index(label)
            position_histogram[str(position) if position < 5 else "5+"] += 1
            first_listed += position == 0
            if turn.player_local_step == 0:
                opening[label] += 1
            if label.startswith("WALL_"):
                walls += 1
                continue
            moves += 1
            listed_moves = [item for item in menu if item.startswith("MOVE_")]
            first_listed_move += bool(listed_moves) and listed_moves[0] == label
            delta = goal_progress_delta(turn)
            forward += delta > 0
            backward += delta < 0
            lateral += delta == 0
            before = turn.credit.progress_before.get("distances")
            after = turn.credit.progress_after.get("distances")
            if before and after:
                distance_gain += int(before[turn.seat]) - int(after[turn.seat])
    return {
        "matches": len(records),
        "policy_versions": sorted({match.policy_version for match in records}),
        "model_turns": turns,
        "mean_game_length": sum(lengths) / len(lengths),
        "termination_counts": dict(sorted(termination.items())),
        "draw_rate": sum(value for key, value in termination.items() if key != "natural_win") / len(records),
        "wins_by_seat": wins,
        "wall_rate": _fraction(walls, turns),
        "first_listed_rate": _fraction(first_listed, turns),
        "first_listed_move_rate": _fraction(first_listed_move, moves),
        "forward_move_rate": _fraction(forward, moves),
        "lateral_move_rate": _fraction(lateral, moves),
        "backward_move_rate": _fraction(backward, moves),
        "mean_path_distance_gain_per_move": _fraction(distance_gain, moves),
        "entropy_estimate_nats": _fraction(-math.fsum(action_log_probs), len(action_log_probs)),
        "near_deterministic_rate": _fraction(sum(value > -0.01 for value in action_log_probs), len(action_log_probs)),
        "chosen_menu_position": dict(sorted(position_histogram.items())),
        "top_openings": [
            {"label": label, "count": count}
            for label, count in sorted(opening.items(), key=lambda item: (-item[1], item[0]))[:5]
        ],
    }


def read_match_directory(root: str | Path) -> list[MatchRecord]:
    """Read every match file of a rollout, pilot, or evaluation directory."""

    root = Path(root)
    directory = root / "matches" if (root / "matches").is_dir() else root
    paths = sorted(directory.glob("game-*.jsonl"))
    if not paths:
        raise ValueError(f"No game-*.jsonl match files under {root}")
    records: list[MatchRecord] = []
    for path in paths:
        records.extend(read_matches_jsonl(path))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="rollout, pilot, or match directories")
    parser.add_argument("--output", type=Path, help="write a JSON list of per-input indicators")
    args = parser.parse_args(argv)
    rows = [{"input": str(path), **batch_indicators(read_match_directory(path))} for path in args.inputs]
    keys = ("matches", "mean_game_length", "draw_rate", "wall_rate", "first_listed_rate",
            "first_listed_move_rate", "forward_move_rate", "lateral_move_rate",
            "mean_path_distance_gain_per_move", "entropy_estimate_nats")
    width = max(len(key) for key in keys)
    names = [Path(row["input"]).name or row["input"] for row in rows]
    print(" " * width + "  " + "  ".join(f"{name[-22:]:>22s}" for name in names))
    for key in keys:
        cells = [row[key] for row in rows]
        print(f"{key:<{width}s}  " + "  ".join(
            f"{'-' if value is None else (f'{value:.3f}' if isinstance(value, float) else value):>22}"
            for value in cells
        ))
    print(f"{'wins_by_seat':<{width}s}  " + "  ".join(f"{str(row['wins_by_seat']):>22s}" for row in rows))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
