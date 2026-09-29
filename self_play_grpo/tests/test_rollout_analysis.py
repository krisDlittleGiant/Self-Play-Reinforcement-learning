from self_play_grpo.envs.observations import BoardState, Coordinate, LegalAction
from self_play_grpo.rollouts.analysis import analyze_policy_matches
from self_play_grpo.rollouts.schema import (
    MatchRecord,
    PolicySample,
    TurnCredit,
    TurnRecord,
)


def state(positions, current_seat, step):
    return BoardState(
        board_size=9,
        pawn_positions=tuple(Coordinate(*position) for position in positions),
        walls_remaining=(5, 5, 5, 5),
        wall_cells=frozenset(),
        current_seat=current_seat,
        joint_action_index=step,
        max_joint_actions=120,
    )


def turn(seat, before, after, label, kind, step):
    action = LegalAction(step, label, label, kind, label)
    return TurnRecord(
        game_id="g",
        joint_step=step,
        seat=seat,
        engine_player=(0, 2, 1, 3)[seat],
        player_local_step=0,
        policy_version="p",
        state_before=state(before, seat, step),
        state_after=state(after, None, step + 1),
        observation="observation",
        legal_actions=(action,),
        chosen_engine_action=step,
        chosen_notation=label,
        policy_sample=PolicySample(label, "prompt", f"{label}\n"),
        credit=TurnCredit(terminal_result=1.0 if seat == 2 else 0.0),
    )


def fixture_match():
    initial = [(4, 8), (0, 4), (4, 0), (8, 4)]
    turns = [
        turn(0, initial, [(4, 7), *initial[1:]], "MOVE_E8", "move", 0),
        turn(1, initial, initial, "MOVE_A4", "move", 1),
        turn(2, initial, [*initial[:2], (4, 1), initial[3]], "MOVE_E2", "move", 2),
        turn(3, initial, initial, "WALL_A1H", "wall", 3),
    ]
    return MatchRecord(
        game_id="g",
        policy_version="p",
        seed=11,
        environment_config={},
        engine_manifest={},
        seat_map=(0, 2, 1, 3),
        initial_state=state(initial, 0, 0),
        turns=turns,
        final_results=(0.0, 0.0, 1.0, 0.0),
        termination_reason="natural_win",
        final_environment="fixture",
    )


def test_policy_match_analysis_reports_goal_direction_by_seat():
    summary = analyze_policy_matches([fixture_match()], top_k=1)
    assert summary["games"] == 1
    assert summary["wins_by_seat"] == [0, 0, 1, 0]
    assert summary["policy_versions"] == ["p"]
    seats = summary["seat_metrics"]
    assert seats[0]["goal_forward_moves"] == 1
    assert seats[1]["lateral_moves"] == 1
    assert seats[2]["goal_forward_moves"] == 1
    assert seats[3]["walls"] == 1
    assert seats[1]["first_actions"] == [{"label": "MOVE_A4", "count": 1}]


def test_policy_match_analysis_rejects_empty_input():
    try:
        analyze_policy_matches([])
    except ValueError as exc:
        assert "At least one match" in str(exc)
    else:
        raise AssertionError("empty analysis input was accepted")
