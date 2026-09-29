import pytest

from self_play_grpo.envs.observations import BoardState, Coordinate
from self_play_grpo.rewards.progress import (
    gae_advantages,
    path_distances,
    potential_shaped_returns,
    potentials,
    proxy_scores,
)


def initial_board(wall_cells=frozenset()) -> BoardState:
    return BoardState(
        board_size=5,
        pawn_positions=(
            Coordinate(2, 4),
            Coordinate(0, 2),
            Coordinate(2, 0),
            Coordinate(4, 2),
        ),
        walls_remaining=(2, 2, 2, 2),
        wall_cells=wall_cells,
        current_seat=0,
        joint_action_index=0,
        max_joint_actions=100,
    )


def test_empty_board_distances_follow_each_canonical_goal_orientation() -> None:
    assert path_distances(initial_board()) == (4, 4, 4, 4)


def test_wall_cells_block_board_graph_edges() -> None:
    # Horizontal wall directly north of seat 0's starting pawn.
    board = initial_board(frozenset({(4, 7), (5, 7), (6, 7)}))
    assert path_distances(board)[0] == 5


def test_proxy_example_matches_plan() -> None:
    before = proxy_scores((3, 6, 8, 9), temperature=2.0)
    after = proxy_scores((3, 4, 8, 9), temperature=2.0)
    assert before[1] == pytest.approx(0.165, abs=0.001)
    assert after[1] == pytest.approx(0.349, abs=0.001)
    assert sum(potentials((3, 6, 8, 9))) == pytest.approx(0.0)


def test_potential_complete_returns_telescope_including_terminal_correction() -> None:
    phis = (-0.10, 0.15, 0.05)
    returns = potential_shaped_returns(phis, outcome=1.0, alpha=0.5)
    assert returns == pytest.approx(tuple(1.0 - 0.5 * phi for phi in phis))


def test_move_away_move_back_process_rewards_cancel() -> None:
    phis = (0.10, -0.10, 0.10)
    returns = potential_shaped_returns(phis, outcome=0.0, alpha=1.0)
    assert returns[0] == pytest.approx(-phis[0])
    assert returns[2] == pytest.approx(-phis[2])


def test_gae_closes_last_pending_transition_at_zero_terminal_value() -> None:
    values = (0.3, 0.4)
    advantages = gae_advantages(values, outcome=1.0, gae_lambda=0.95)
    final_delta = 1.0 - values[-1]
    first_delta = values[1] - values[0]
    assert advantages == pytest.approx((first_delta + 0.95 * final_delta, final_delta))
