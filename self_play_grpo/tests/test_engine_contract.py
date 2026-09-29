from pathlib import Path

import pytest

from self_play_grpo.cli import _validate_winning_fixture
from self_play_grpo.config import load_config
from self_play_grpo.envs.quoridor import QuoridorEnv
from self_play_grpo.policies.bots import ShortestPathPolicy
from self_play_grpo.rollouts.collector import MatchCollector
from self_play_grpo.rollouts.replay import replay_match

pyspiel = pytest.importorskip("pyspiel")


pytestmark = pytest.mark.engine
PROJECT = Path(__file__).resolve().parents[1]


def fixture_config():
    return load_config(PROJECT / "configs" / "quoridor_fixture.yaml")


def test_initial_engine_player_order_and_seat_map() -> None:
    env = QuoridorEnv(fixture_config().environment)
    assert env.seat_map.engine_players_by_seat == (0, 2, 1, 3)
    observed = []
    for _ in range(4):
        observed.append(env.seat_map.engine_player(env.current_seat))
        env.step(next(action for action in env.legal_actions() if action.kind == "move"))
    assert observed == [0, 2, 1, 3]


def test_every_legal_action_round_trips_through_label_and_engine_id() -> None:
    env = QuoridorEnv(fixture_config().environment)
    for action in env.legal_actions():
        assert env.action_for_label(action.label) == action
        clone = env.clone()
        clone.step(action)
        assert clone.joint_actions == 1


def test_complete_collected_match_replays_state_by_state() -> None:
    config = fixture_config()
    match = MatchCollector(collect_progress=False).collect(
        QuoridorEnv(config.environment),
        ShortestPathPolicy(),
        game_id="replay-contract",
        seed=11,
        policy_version="bot-replay-contract",
    )
    replayed = replay_match(match)
    assert replayed.serialize() == match.final_environment


def test_pinned_forced_pass_regression() -> None:
    """The pinned post-2.0.2 fix must encode a forced pass as relative ID 36."""

    game = pyspiel.load_game("quoridor", {"players": 4})
    state = game.new_initial_state()
    actions = (
        163, 215, 63, 87, 73, 199, 91, 141, 1, 267, 173, 185, 187, 195,
        15, 19, 2, 111, 51, 25, 29, 247, 109, 177, 99, 255, 221, 253,
        233, 137, 34, 81, 70, 133, 209, 57, 79, 159, 38, 11, 2, 70,
        241, 70, 70, 2, 38, 2, 167, 70, 123, 34, 227, 2, 34, 38,
        34, 2, 38, 34, 2, 2, 34, 38, 70, 38, 34, 34, 2, 38,
        34, 38, 70, 2, 34, 34, 38, 34, 38, 34, 34, 38, 38, 38,
        2, 34, 38, 34, 38, 38, 38, 38, 70, 70, 34, 34, 34, 34,
        34, 2, 34, 38, 70, 2, 2, 34, 34, 70, 2, 34, 70, 2,
        2, 70, 2, 70, 2, 2, 70, 2, 34, 38, 34, 70, 70, 68,
        70, 70, 2, 38, 70, 38, 0, 68,
    )
    for action in actions:
        state.apply_action(action)
    assert not state.is_terminal()
    assert state.current_player() == 1
    assert state.legal_actions() == [36]
    before = str(state)
    state.apply_action(36)
    assert str(state) == before
    assert state.current_player() == 3


@pytest.mark.parametrize("seat", range(4))
def test_terminal_reward_is_assigned_to_correct_canonical_seat(seat) -> None:
    fixture = _validate_winning_fixture(fixture_config(), seat)
    assert fixture["results"][seat] == 1.0
