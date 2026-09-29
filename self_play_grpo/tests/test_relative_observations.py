import pytest

from self_play_grpo.envs.observations import (
    BoardState,
    Coordinate,
    LegalAction,
    player_relative_actions,
    player_relative_board,
    render_player_relative_observation,
    rotate_action_notation,
)


STARTS = (
    Coordinate(4, 8),
    Coordinate(0, 4),
    Coordinate(4, 0),
    Coordinate(8, 4),
)


def board(current_seat):
    return BoardState(
        board_size=9,
        pawn_positions=STARTS,
        walls_remaining=(5, 5, 5, 5),
        wall_cells=frozenset({(1, 0), (1, 1), (1, 2)}),
        current_seat=current_seat,
        joint_action_index=0,
        max_joint_actions=120,
    )


@pytest.mark.parametrize(
    ("seat", "absolute_forward"),
    ((0, "e8"), (1, "b5"), (2, "e2"), (3, "h5")),
)
def test_every_seat_has_the_same_relative_start_and_forward_label(
    seat, absolute_forward
):
    relative = player_relative_board(board(seat), seat)
    assert relative.current_seat == 0
    assert relative.pawn_positions[0] == Coordinate(4, 8)
    assert rotate_action_notation(absolute_forward, 9, seat) == "e8"


@pytest.mark.parametrize("notation", ("a1h", "a1v", "d4h", "h8v"))
def test_four_quarter_turns_restore_wall_notation(notation):
    rotated = notation
    for _ in range(4):
        rotated = rotate_action_notation(rotated, 9, 1)
    assert rotated == notation


def test_relative_actions_preserve_engine_ids_and_are_unique():
    absolute = (
        LegalAction(10, "MOVE_A4", "a4", "move", "move the pawn to a4"),
        LegalAction(11, "MOVE_B5", "b5", "move", "move the pawn to b5"),
        LegalAction(12, "WALL_A1H", "a1h", "wall", "wall"),
    )
    relative = player_relative_actions(absolute, seat=1, board_size=9)
    assert {action.engine_action for action in relative} == {10, 11, 12}
    assert {action.label for action in relative} == {
        "MOVE_D9",
        "MOVE_E8",
        "WALL_A8V",
    }


@pytest.mark.parametrize("seat", range(4))
def test_relative_rotation_is_a_bijection_over_complete_action_family(seat):
    moves = [
        f"{chr(ord('a') + x)}{y + 1}"
        for x in range(9)
        for y in range(9)
    ]
    walls = [
        f"{chr(ord('a') + x)}{y + 1}{orientation}"
        for x in range(8)
        for y in range(8)
        for orientation in ("h", "v")
    ]
    assert {rotate_action_notation(item, 9, seat) for item in moves} == set(moves)
    assert {rotate_action_notation(item, 9, seat) for item in walls} == set(walls)


def test_relative_prompt_hides_canonical_seat_and_uses_relative_goal():
    actions = (
        LegalAction(11, "MOVE_E8", "e8", "move", "move the pawn to e8"),
    )
    prompt = render_player_relative_observation(board(1), 1, actions, ())
    assert "You are player 0" in prompt
    assert "your goal is always the top edge" in prompt
    assert "Canonical seat" not in prompt
    assert "MOVE_E8" in prompt
