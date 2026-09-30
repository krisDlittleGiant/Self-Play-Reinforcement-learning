"""Reproducible fixed policies for collection and external evaluation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from self_play_grpo.envs.quoridor import QuoridorEnv
from self_play_grpo.rewards.progress import path_distances
from self_play_grpo.rollouts.schema import PolicySample


def _bot_sample(env: QuoridorEnv, label: str, name: str) -> PolicySample:
    return PolicySample(
        chosen_label=label,
        prompt_text=env.observation(env.current_seat),
        completion_text=f"{label}\n",
        sampling_config={"policy_type": "bot", "policy_name": name},
    )


@dataclass(frozen=True)
class RandomPolicy:
    name: str = "random_legal"

    def select_action(
        self, env: QuoridorEnv, *, game_id: str, rng: random.Random
    ) -> PolicySample:
        del game_id
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("Random policy was asked to act in a terminal state")
        return _bot_sample(env, rng.choice(legal).label, self.name)


@dataclass(frozen=True)
class ShortestPathPolicy:
    """Choose a pawn move minimizing the mover's geometric path distance."""

    name: str = "shortest_path"

    def select_action(
        self, env: QuoridorEnv, *, game_id: str, rng: random.Random
    ) -> PolicySample:
        del game_id
        seat = env.current_seat
        candidates = [action for action in env.legal_actions() if action.kind == "move"]
        if not candidates:
            candidates = list(env.legal_actions())
        scored: list[tuple[int, str]] = []
        for action in candidates:
            future = env.clone()
            state = future.step(action)
            scored.append((path_distances(state)[seat], action.label))
        best = min(score for score, _ in scored)
        labels = sorted(label for score, label in scored if score == best)
        return _bot_sample(env, rng.choice(labels), self.name)


@dataclass(frozen=True)
class NoisyShortestPathPolicy:
    """Shortest-path racer that plays a uniformly random pawn move with probability epsilon.

    It sits between the random and shortest-path bots, so an evaluation
    ladder is not stuck at zero or full wins.
    """

    epsilon: float = 0.5
    name: str = "noisy_shortest_path"

    def select_action(
        self, env: QuoridorEnv, *, game_id: str, rng: random.Random
    ) -> PolicySample:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        moves = [action for action in env.legal_actions() if action.kind == "move"]
        if moves and rng.random() < self.epsilon:
            return _bot_sample(env, rng.choice(sorted(action.label for action in moves)), self.name)
        return replace_name(
            ShortestPathPolicy().select_action(env, game_id=game_id, rng=rng), self.name
        )


def replace_name(sample: PolicySample, name: str) -> PolicySample:
    return PolicySample(
        chosen_label=sample.chosen_label,
        prompt_text=sample.prompt_text,
        completion_text=sample.completion_text,
        sampling_config={"policy_type": "bot", "policy_name": name},
    )


@dataclass(frozen=True)
class WallAwarePolicy:
    """Frozen heuristic balancing own progress and opponents' obstruction."""

    opponent_weight: float = 0.25
    wall_cost: float = 0.05
    name: str = "wall_aware"

    def select_action(
        self, env: QuoridorEnv, *, game_id: str, rng: random.Random
    ) -> PolicySample:
        del game_id
        seat = env.current_seat
        before = path_distances(env.state_view())
        scored: list[tuple[float, str]] = []
        for action in env.legal_actions():
            future = env.clone()
            after = path_distances(future.step(action))
            own_progress = before[seat] - after[seat]
            obstruction = sum(
                after[other] - before[other] for other in range(4) if other != seat
            )
            cost = self.wall_cost if action.kind == "wall" else 0.0
            score = own_progress + self.opponent_weight * obstruction - cost
            scored.append((score, action.label))
        if not scored:
            raise RuntimeError("Wall-aware policy was asked to act in a terminal state")
        best = max(score for score, _ in scored)
        labels = sorted(
            label for score, label in scored if math.isclose(score, best, abs_tol=1e-12)
        )
        return _bot_sample(env, rng.choice(labels), self.name)

