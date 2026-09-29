"""Tournament design and uncertainty over independent complete matches."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from self_play_grpo.envs.quoridor import QuoridorEnv
from self_play_grpo.rollouts.collector import ActionPolicy, MatchCollector


@dataclass(frozen=True)
class EvaluationGame:
    game_id: str
    seed: int
    candidate_name: str
    candidate_seat: int
    opponent_names: tuple[str, str, str]
    candidate_result: float
    final_results: tuple[float, float, float, float]
    termination_reason: str
    joint_actions: int


@dataclass(frozen=True)
class TournamentSummary:
    games: int
    candidate_name: str
    mean_fractional_result: float
    win_rate: float
    draw_rate: float
    mean_game_length: float
    result_by_seat: dict[int, float]
    bootstrap_95_interval: tuple[float, float]


def bootstrap_match_mean(
    values: Sequence[float], *, seed: int = 0, samples: int = 10_000
) -> tuple[float, float]:
    """Bootstrap whole independent matches, never turns or player views."""

    if not values:
        raise ValueError("At least one match is required")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


class TournamentRunner:
    def __init__(
        self,
        env_factory: Callable[[], QuoridorEnv],
        *,
        collect_full_records: bool = False,
    ) -> None:
        self.env_factory = env_factory
        self.collect_full_records = collect_full_records
        self.collector = MatchCollector(collect_progress=collect_full_records)

    def run(
        self,
        candidate: ActionPolicy,
        opponents: Sequence[ActionPolicy],
        *,
        games_per_seat: int,
        seed: int,
    ) -> list[EvaluationGame]:
        if len(opponents) != 3:
            raise ValueError("A four-player candidate evaluation needs exactly three opponents")
        if games_per_seat <= 0:
            raise ValueError("games_per_seat must be positive")
        games: list[EvaluationGame] = []
        seed_rng = random.Random(seed)
        for candidate_seat in range(4):
            for repeat in range(games_per_seat):
                match_seed = seed_rng.randrange(0, 2**63)
                policies: list[ActionPolicy] = []
                opponent_index = 0
                for seat in range(4):
                    if seat == candidate_seat:
                        policies.append(candidate)
                    else:
                        policies.append(opponents[opponent_index])
                        opponent_index += 1
                game_id = (
                    f"eval-{candidate.name}-seat-{candidate_seat}-"
                    f"repeat-{repeat}-seed-{match_seed}"
                )
                match = self.collector.collect(
                    self.env_factory(),
                    policies,
                    game_id=game_id,
                    seed=match_seed,
                    policy_version=f"evaluation:{candidate.name}",
                )
                games.append(
                    EvaluationGame(
                        game_id=game_id,
                        seed=match_seed,
                        candidate_name=candidate.name,
                        candidate_seat=candidate_seat,
                        opponent_names=tuple(opponent.name for opponent in opponents),  # type: ignore[arg-type]
                        candidate_result=match.final_results[candidate_seat],
                        final_results=match.final_results,
                        termination_reason=match.termination_reason,
                        joint_actions=len(match.turns),
                    )
                )
        return games

    @staticmethod
    def summarize(games: Sequence[EvaluationGame], *, bootstrap_seed: int = 0) -> TournamentSummary:
        if not games:
            raise ValueError("No evaluation games were supplied")
        names = {game.candidate_name for game in games}
        if len(names) != 1:
            raise ValueError("A summary cannot mix candidate policies")
        results = [game.candidate_result for game in games]
        by_seat = {
            seat: float(
                np.mean([game.candidate_result for game in games if game.candidate_seat == seat])
            )
            for seat in range(4)
        }
        return TournamentSummary(
            games=len(games),
            candidate_name=next(iter(names)),
            mean_fractional_result=float(np.mean(results)),
            win_rate=float(np.mean([value == 1.0 for value in results])),
            draw_rate=float(np.mean([value == 0.25 for value in results])),
            mean_game_length=float(np.mean([game.joint_actions for game in games])),
            result_by_seat=by_seat,
            bootstrap_95_interval=bootstrap_match_mean(results, seed=bootstrap_seed),
        )


def write_evaluation_jsonl(
    path: str | Path,
    games: Sequence[EvaluationGame],
    summary: TournamentSummary,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "summary", **asdict(summary)}, sort_keys=True))
        handle.write("\n")
        for game in games:
            handle.write(json.dumps({"record_type": "game", **asdict(game)}, sort_keys=True))
            handle.write("\n")
    temporary.replace(target)

