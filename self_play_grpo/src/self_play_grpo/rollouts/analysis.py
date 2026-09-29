"""Pure, accelerator-free diagnostics over persisted policy matches."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from self_play_grpo.rollouts.schema import MatchRecord, TurnRecord


def goal_progress_delta(turn: TurnRecord) -> int:
    """Return signed coordinate progress toward the acting seat's goal edge."""

    seat = turn.seat
    before = turn.state_before.pawn_positions[seat]
    after = turn.state_after.pawn_positions[seat]
    if seat == 0:
        return before.y - after.y
    if seat == 1:
        return after.x - before.x
    if seat == 2:
        return after.y - before.y
    if seat == 3:
        return before.x - after.x
    raise ValueError(f"Unexpected canonical seat: {seat}")


def _ranked(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [
        {"label": label, "count": count}
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    ]


def analyze_policy_matches(
    matches: Sequence[MatchRecord] | Iterable[MatchRecord], *, top_k: int = 10
) -> dict[str, Any]:
    """Summarize outcomes and goal-directed movement for all four seats."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    records = list(matches)
    if not records:
        raise ValueError("At least one match is required")

    policy_versions = sorted({match.policy_version for match in records})
    termination_counts = Counter(match.termination_reason for match in records)
    wins = [0, 0, 0, 0]
    for match in records:
        for seat, result in enumerate(match.final_results):
            if result == 1.0:
                wins[seat] += 1

    seat_rows: list[dict[str, Any]] = []
    for seat in range(4):
        actions = moves = walls = forward = backward = lateral = net = 0
        labels: Counter[str] = Counter()
        first_labels: Counter[str] = Counter()
        for match in records:
            for turn in match.turns:
                if turn.seat != seat:
                    continue
                actions += 1
                label = turn.policy_sample.chosen_label
                labels[label] += 1
                if turn.player_local_step == 0:
                    first_labels[label] += 1
                selected = next(
                    action for action in turn.legal_actions if action.label == label
                )
                if selected.kind == "wall":
                    walls += 1
                    continue
                moves += 1
                progress = goal_progress_delta(turn)
                net += progress
                if progress > 0:
                    forward += 1
                elif progress < 0:
                    backward += 1
                else:
                    lateral += 1
        seat_rows.append(
            {
                "actions": actions,
                "first_actions": _ranked(first_labels, top_k),
                "goal_backward_moves": backward,
                "goal_forward_moves": forward,
                "lateral_moves": lateral,
                "moves": moves,
                "net_goal_progress": net,
                "seat": seat,
                "top_actions": _ranked(labels, top_k),
                "walls": walls,
                "wins": wins[seat],
            }
        )

    return {
        "decisive_games": sum(wins),
        "games": len(records),
        "policy_versions": policy_versions,
        "seat_metrics": seat_rows,
        "termination_counts": dict(sorted(termination_counts.items())),
        "wins_by_seat": wins,
    }
