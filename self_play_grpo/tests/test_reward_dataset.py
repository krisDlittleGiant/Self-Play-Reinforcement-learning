"""Model-free tests for outcome-supervised reward data provenance."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from self_play_grpo.rewards.dataset import (
    RewardExample,
    game_split,
    outcome_examples,
    write_reward_examples,
)


def example(game_id: str = "game-1", step: int = 0) -> RewardExample:
    return RewardExample(
        game_id=game_id,
        seed=11,
        policy_version="policy-000000",
        joint_step=step,
        seat=0,
        split=game_split(game_id),
        observation="Current board and legal actions",
        chosen_action_label="MOVE_A2",
        terminal_result=1.0,
        outcome_advantage=3 ** 0.5,
        source_manifest_sha256="a" * 64,
    )


def test_game_split_is_stable_and_turn_independent():
    assert game_split("game-1") == game_split("game-1")
    assert {game_split(f"game-{index}") for index in range(100)} == {
        "train", "validation", "test"
    }
    with pytest.raises(ValueError, match="cannot be empty"):
        game_split("")


def test_feature_and_label_boundaries_are_explicit():
    payload = example().to_dict()
    assert payload["features"] == {
        "observation": "Current board and legal actions",
        "chosen_action_label": "MOVE_A2",
    }
    assert payload["labels"]["terminal_result"] == 1.0
    assert "terminal_result" not in payload["features"]
    assert "game_id" not in payload["features"]
    with pytest.raises(ValueError, match="split"):
        replace(example(), split="wrong").validate()


def test_outcome_examples_check_labels_against_engine_result():
    credit = SimpleNamespace(
        outcome_advantage=3 ** 0.5,
        training_advantage=3 ** 0.5,
    )
    turns = [SimpleNamespace(
        joint_step=0,
        seat=0,
        observation="Board at step 0",
        policy_sample=SimpleNamespace(chosen_label="MOVE_A2"),
        credit=credit,
    )]
    match = SimpleNamespace(
        validate=lambda: None,
        game_id="game-1",
        seed=11,
        policy_version="policy-000000",
        final_results=(1.0, 0.0, 0.0, 0.0),
        turns=turns,
    )
    examples = outcome_examples(match, source_manifest_sha256="a" * 64)
    assert len(examples) == 1
    assert examples[0].terminal_result == 1.0
    assert examples[0].outcome_advantage == pytest.approx(3 ** 0.5)
    credit.training_advantage = -1.0
    with pytest.raises(ValueError, match="training advantage differs"):
        outcome_examples(match, source_manifest_sha256="a" * 64)


def test_write_is_new_only_and_keeps_complete_game_split(tmp_path):
    path = tmp_path / "reward-data.jsonl"
    counts = write_reward_examples(path, [example(step=0), example(step=1)])
    assert counts[game_split("game-1")] == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(FileExistsError, match="already exists"):
        write_reward_examples(path, [example()])
    with pytest.raises(ValueError, match="cannot be empty"):
        write_reward_examples(tmp_path / "empty.jsonl", [])
    assert not (tmp_path / "empty.jsonl").exists()
