from pathlib import Path
from dataclasses import replace

import pytest

from self_play_grpo.config import PINNED_ENGINE_REVISION, load_config


PROJECT = Path(__file__).resolve().parents[1]


def test_main_configuration_is_fully_pinned() -> None:
    config = load_config(PROJECT / "configs" / "quoridor_outcome.yaml")
    assert config.environment.engine_revision == PINNED_ENGINE_REVISION
    assert config.model.id == "Qwen/Qwen3-4B"
    assert config.model.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert config.model.local_path == "/scratch/svijay46/models/Qwen3-4B"
    assert config.environment.max_joint_actions == 120
    assert config.environment.action_perspective == "player_relative"
    assert config.training.loss_normalizer_per_game == 120
    assert config.training.weight_decay == 0.0
    assert config.rollout.parallel_games_per_rank == 2


def test_fixture_uses_same_game_with_explicit_smaller_parameters() -> None:
    config = load_config(PROJECT / "configs" / "quoridor_fixture.yaml")
    assert config.environment.players == 4
    assert config.environment.board_size == 5
    assert config.environment.wall_count == 2
    assert config.environment.max_joint_actions == 100
    assert config.environment.action_perspective == "player_relative"
    assert config.training.weight_decay == 0.0
    assert config.rollout.parallel_games_per_rank == 2


def test_parallel_games_per_rank_must_be_positive() -> None:
    config = load_config(PROJECT / "configs" / "quoridor_fixture.yaml")
    with pytest.raises(ValueError, match="parallel_games_per_rank must be positive"):
        replace(config.rollout, parallel_games_per_rank=0).validate()
