"""Geometric progress proxy, potential shaping, and own-turn GAE."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np

from self_play_grpo.envs.observations import BoardState
from self_play_grpo.rewards.outcome import FOUR_PLAYER_OUTCOME_SCALE


def _is_goal(seat: int, x: int, y: int, board_size: int) -> bool:
    return (
        (seat == 0 and y == 0)
        or (seat == 1 and x == board_size - 1)
        or (seat == 2 and y == board_size - 1)
        or (seat == 3 and x == 0)
    )


def shortest_path_distance(board: BoardState, seat: int) -> int:
    """Wall-respecting BFS distance that deliberately ignores pawn occupancy."""

    if seat not in range(4):
        raise ValueError("seat must be in [0, 3]")
    start = board.pawn_positions[seat]
    queue: deque[tuple[int, int, int]] = deque([(start.x, start.y, 0)])
    visited = {(start.x, start.y)}
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    while queue:
        x, y, distance = queue.popleft()
        if _is_goal(seat, x, y, board.board_size):
            return distance
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= board.board_size or ny >= board.board_size:
                continue
            # Board squares occupy even/even cells in OpenSpiel's diameter
            # grid. The intervening odd coordinate contains a blocking wall.
            wall_cell = (2 * x + dx, 2 * y + dy)
            if wall_cell in board.wall_cells or (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny, distance + 1))
    raise RuntimeError(f"Seat {seat} has no goal path; the engine allowed an illegal wall state")


def path_distances(board: BoardState) -> tuple[int, int, int, int]:
    return tuple(shortest_path_distance(board, seat) for seat in range(4))  # type: ignore[return-value]


def proxy_scores(distances: Sequence[int | float], temperature: float = 2.0) -> tuple[float, ...]:
    if len(distances) != 4:
        raise ValueError("Exactly four distances are required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray([-float(value) / temperature for value in distances], dtype=np.float64)
    logits -= np.max(logits)
    weights = np.exp(logits)
    scores = weights / np.sum(weights)
    return tuple(float(value) for value in scores)


def potentials(distances: Sequence[int | float], temperature: float = 2.0) -> tuple[float, ...]:
    return tuple(score - 0.25 for score in proxy_scores(distances, temperature))


def progress_features(board: BoardState, temperature: float = 2.0) -> dict[str, list[float] | list[int]]:
    distances = path_distances(board)
    scores = proxy_scores(distances, temperature)
    return {
        "distances": list(distances),
        "proxy_scores": list(scores),
        "potentials": [value - 0.25 for value in scores],
        "walls_remaining": list(board.walls_remaining),
    }


def potential_shaped_returns(
    decision_potentials: Sequence[float],
    outcome: float,
    alpha: float = 1.0,
) -> tuple[float, ...]:
    """Complete returns over one player's own-decision sequence.

    The next state is the state before that player's next action. The terminal
    boundary potential is zero and the game result is paid only on the final
    transition, making each returned value equal to outcome-alpha*Phi(s_t).
    """

    if not decision_potentials:
        return ()
    rewards: list[float] = []
    for index, current in enumerate(decision_potentials):
        final = index == len(decision_potentials) - 1
        next_value = 0.0 if final else float(decision_potentials[index + 1])
        game_reward = float(outcome) if final else 0.0
        rewards.append(game_reward + alpha * (next_value - float(current)))
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + running
        returns[index] = running
    return tuple(returns)


def normalized_potential_advantages(
    decision_potentials: Sequence[float], outcome: float, alpha: float = 1.0
) -> tuple[float, ...]:
    returns = potential_shaped_returns(decision_potentials, outcome, alpha)
    return tuple((value - 0.25) / FOUR_PLAYER_OUTCOME_SCALE for value in returns)


def gae_advantages(
    decision_values: Sequence[float],
    outcome: float,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[float, ...]:
    """Compute GAE along one player's own decisions with a zero terminal value."""

    if gamma != 1.0:
        raise ValueError("This experiment specifies undiscounted gamma=1")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not decision_values:
        return ()
    deltas: list[float] = []
    for index, current in enumerate(decision_values):
        final = index == len(decision_values) - 1
        next_value = 0.0 if final else float(decision_values[index + 1])
        reward = float(outcome) if final else 0.0
        deltas.append(reward + gamma * next_value - float(current))
    advantages = [0.0] * len(deltas)
    running = 0.0
    for index in range(len(deltas) - 1, -1, -1):
        running = deltas[index] + gamma * gae_lambda * running
        advantages[index] = running
    return tuple(advantages)


def blend_process_advantages(
    outcome_advantage: float,
    gae: Iterable[float],
    eta: float = 0.25,
) -> tuple[float, ...]:
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    return tuple(
        (1.0 - eta) * float(outcome_advantage)
        + eta * float(value) / FOUR_PLAYER_OUTCOME_SCALE
        for value in gae
    )


def evaluator_features(board: BoardState) -> np.ndarray:
    """Encode public state for the optional small outcome evaluator."""

    size = board.board_size
    diameter = size * 2 - 1
    pawn_planes = np.zeros((4, size, size), dtype=np.float32)
    for seat, position in enumerate(board.pawn_positions):
        pawn_planes[seat, position.y, position.x] = 1.0
    walls = np.zeros((diameter, diameter), dtype=np.float32)
    for x, y in board.wall_cells:
        walls[y, x] = 1.0
    mover = np.zeros(4, dtype=np.float32)
    if board.current_seat is not None:
        mover[board.current_seat] = 1.0
    scalars = np.asarray(
        [
            *(value / max(1, size) for value in board.walls_remaining),
            *mover,
            board.actions_remaining / max(1, board.max_joint_actions),
        ],
        dtype=np.float32,
    )
    return np.concatenate((pawn_planes.reshape(-1), walls.reshape(-1), scalars))
