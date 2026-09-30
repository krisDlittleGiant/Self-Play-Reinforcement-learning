"""Prompt-menu variants change presentation only and leave old runs byte-identical."""

from dataclasses import replace
from pathlib import Path

import pytest

from self_play_grpo.config import EnvironmentConfig, load_config
from self_play_grpo.envs.observations import (
    BoardState,
    Coordinate,
    LegalAction,
    MenuStyle,
    render_action_menu,
    render_player_relative_observation,
)
from self_play_grpo.rollouts.pilot import canonical_sha256


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "artifacts" / "d5-one-update-64games-qwen3-4b-seed-11"
# Recorded by rollout-000000 and rollout-000001 of the first production run.
RECORDED_64_GAME_CONFIG_SHA256 = "9b16e5eafca3f6e4d9b2bc4fba1fdb3ad51b9dbbc1d074791b2ce293755a5bd5"


def _action(label: str) -> LegalAction:
    notation = label.split("_", 1)[1].lower()
    kind = "wall" if notation.endswith(("h", "v")) else "move"
    description = (
        f"place a {'horizontal' if notation.endswith('h') else 'vertical'} wall starting at {notation[:-1]}"
        if kind == "wall" else f"move the pawn to {notation}"
    )
    return LegalAction(engine_action=0, label=label, notation=notation, kind=kind, description=description)


LEGAL = tuple(sorted(
    [_action(label) for label in ("MOVE_D9", "MOVE_E8", "MOVE_F9")]
    + [_action(f"WALL_{column}{row}{side}") for column in "AB" for row in "12" for side in "HV"],
    key=lambda action: action.label,
))


def test_existing_config_digest_is_unchanged_by_later_optional_fields():
    config = load_config(PROJECT / "configs" / "quoridor_outcome_64games.yaml")
    assert canonical_sha256(config.to_dict()) == RECORDED_64_GAME_CONFIG_SHA256
    assert "action_menu_order" not in config.to_dict()["environment"]
    assert "minibatches_per_update" not in config.to_dict()["training"]


def test_non_default_optional_fields_enter_the_digest():
    config = load_config(PROJECT / "configs" / "quoridor_outcome_64games.yaml")
    changed = replace(config, environment=replace(config.environment, action_menu_order="shuffled"))
    assert changed.to_dict()["environment"]["action_menu_order"] == "shuffled"
    assert canonical_sha256(changed.to_dict()) != RECORDED_64_GAME_CONFIG_SHA256
    changed = replace(config, training=replace(config.training, minibatches_per_update=4))
    assert changed.to_dict()["training"]["minibatches_per_update"] == 4


def test_invalid_menu_options_are_rejected():
    with pytest.raises(ValueError, match="action_menu_order"):
        EnvironmentConfig(action_menu_order="alphabetical").validate()
    with pytest.raises(ValueError, match="wall_menu"):
        EnvironmentConfig(wall_menu="none").validate()
    with pytest.raises(ValueError, match="player_relative"):
        EnvironmentConfig(move_descriptions="directional").validate()


def test_default_menu_is_the_original_sorted_described_list():
    expected = "Legal actions:\n" + "\n".join(f"- {a.label}: {a.description}" for a in LEGAL)
    assert render_action_menu(LEGAL, MenuStyle()) == expected


def test_shuffle_is_deterministic_per_key_keeps_moves_first_and_preserves_the_legal_set():
    style = MenuStyle(order="shuffled", shuffle_key="seed-1:step-0")
    first = render_action_menu(LEGAL, style)
    assert first == render_action_menu(LEGAL, style)
    labels = [line.split(":")[0][2:] for line in first.splitlines()[1:]]
    assert sorted(labels) == [action.label for action in LEGAL]
    assert all(label.startswith("MOVE") for label in labels[:3])
    orders = {
        tuple(render_action_menu(LEGAL, MenuStyle(order="shuffled", shuffle_key=f"k{i}")).splitlines()[1:4])
        for i in range(40)
    }
    assert len(orders) == 6
    with pytest.raises(ValueError, match="match-specific key"):
        render_action_menu(LEGAL, MenuStyle(order="shuffled"))


def test_compact_walls_list_every_wall_label_once_and_shrink_the_menu():
    compact = render_action_menu(LEGAL, MenuStyle(walls="compact"))
    walls = [action.label for action in LEGAL if action.kind == "wall"]
    wall_line = compact.splitlines()[-1]
    assert wall_line.split(" ") == walls
    shuffled = render_action_menu(LEGAL, MenuStyle(order="shuffled", walls="compact", shuffle_key="k"))
    assert shuffled.splitlines()[-1] == wall_line
    assert "Legal pawn moves:\n- MOVE_D9: move the pawn to d9" in compact
    assert len(compact) < len(render_action_menu(LEGAL, MenuStyle()))


def test_directional_descriptions_are_relative_to_the_upward_goal():
    menu = render_action_menu(
        LEGAL, MenuStyle(moves="directional"), mover_position=Coordinate(4, 8),
    )
    assert "- MOVE_E8: move the pawn forward to e8" in menu
    assert "- MOVE_D9: move the pawn sideways left to d9" in menu
    assert "- MOVE_F9: move the pawn sideways right to f9" in menu


def test_player_relative_default_prompt_matches_the_original_renderer_text():
    board = BoardState(
        board_size=9,
        pawn_positions=(Coordinate(4, 8), Coordinate(0, 4), Coordinate(4, 0), Coordinate(8, 4)),
        walls_remaining=(5, 5, 5, 5), wall_cells=frozenset(), current_seat=0,
        joint_action_index=0, max_joint_actions=120,
    )
    prompt = render_player_relative_observation(board, 0, LEGAL, ())
    menu = "\n".join(f"- {a.label}: {a.description}" for a in LEGAL)
    assert prompt.endswith(f"Legal actions:\n{menu}\n\nAction: ")


@pytest.mark.engine
def test_recorded_production_prompts_are_reproduced_byte_for_byte():
    pytest.importorskip("pyspiel")
    from self_play_grpo.envs.quoridor import QuoridorEnv
    from self_play_grpo.rollouts.schema import read_matches_jsonl

    matches_dir = RUN / "rollout-000001" / "matches"
    if not matches_dir.is_dir():
        pytest.skip("Recorded production run is not present")
    checked = 0
    for path in sorted(matches_dir.glob("game-*.jsonl"))[:4]:
        (match,) = read_matches_jsonl(path)
        env = QuoridorEnv(EnvironmentConfig(**match.environment_config))
        env.reset(seed=match.seed)
        for turn in match.turns:
            assert env.observation(turn.seat) == turn.observation
            env.step(env.action_for_label(turn.policy_sample.chosen_label))
            checked += 1
        assert env.serialize() == match.final_environment
    assert checked > 100


@pytest.mark.engine
def test_non_default_environment_round_trips_through_serialization():
    pytest.importorskip("pyspiel")
    from self_play_grpo.envs.quoridor import QuoridorEnv

    base = load_config(PROJECT / "configs" / "quoridor_outcome_64games.yaml").environment
    config = replace(base, action_menu_order="shuffled", wall_menu="compact", move_descriptions="directional")
    env = QuoridorEnv(config)
    env.reset(seed=123)
    opening = env.observation(env.current_seat)
    assert "Legal pawn moves:" in opening and "forward" in opening
    for _ in range(6):
        env.step(next(action for action in env.legal_actions() if action.kind == "move"))
    restored = QuoridorEnv.deserialize(env.serialize())
    assert restored.config == config
    assert restored.observation(restored.current_seat) == env.observation(env.current_seat)
    openings = set()
    for seed in range(120, 130):
        other = QuoridorEnv(config)
        other.reset(seed=seed)
        openings.add(other.observation(other.current_seat))
    assert len(openings) > 1


def _wall_history():
    return (
        {"joint_step": 0, "seat": 1, "engine_player": 2, "engine_action": 0, "label": "WALL_C3H",
         "notation": "c3h", "absolute_label": "WALL_C3H", "absolute_notation": "c3h"},
        {"joint_step": 1, "seat": 2, "engine_player": 1, "engine_action": 0, "label": "MOVE_E2",
         "notation": "e2", "absolute_label": "MOVE_E2", "absolute_notation": "e2"},
    )


def test_explicit_wall_listing_names_each_wall_in_the_actors_frame():
    from self_play_grpo.envs.observations import action_label, rotate_action_notation

    board = BoardState(
        board_size=9,
        pawn_positions=(Coordinate(4, 8), Coordinate(0, 4), Coordinate(4, 1), Coordinate(8, 4)),
        walls_remaining=(5, 4, 5, 5), wall_cells=frozenset(), current_seat=1,
        joint_action_index=2, max_joint_actions=120,
    )
    history = _wall_history()
    plain = render_player_relative_observation(board, 1, LEGAL, history)
    assert "Walls placed" not in plain
    listed = render_player_relative_observation(board, 1, LEGAL, history, MenuStyle(walls_placed="explicit"))
    relative = action_label(rotate_action_notation("c3h", 9, 1))
    assert f"Walls placed:\n{relative} (player 0)\n\n" in listed
    assert "MOVE_E2" not in listed.split("Walls placed:")[1].split("\n\n")[0]
    empty = replace(board, current_seat=1)
    assert "Walls placed:\n(none)\n\n" in render_player_relative_observation(
        empty, 1, LEGAL, history[1:], MenuStyle(walls_placed="explicit"))


def test_wall_listing_option_is_validated_and_omitted_at_default():
    config = load_config(PROJECT / "configs" / "quoridor_outcome_64games.yaml")
    assert "wall_listing" not in config.to_dict()["environment"]
    with pytest.raises(ValueError, match="wall_listing"):
        EnvironmentConfig(wall_listing="all").validate()
    run3 = load_config(PROJECT / "configs" / "quoridor_outcome_64games_run3.yaml")
    assert run3.environment.wall_listing == "explicit"
    assert run3.training.advantage_baseline == "seat_loo"
    assert run3.training.minibatches_per_update == 4
