from pathlib import Path
import pytest
from self_play_grpo.config import load_config
from self_play_grpo.evaluation.fixed_checkpoint import build_suite, evaluate
from types import SimpleNamespace


CONFIG = Path(__file__).parents[1] / "configs/quoridor_outcome_64games.yaml"


def test_frozen_schedule_is_seat_balanced_reproducible_and_separate_from_training():
    config = load_config(CONFIG)
    suite = build_suite(config, games_per_seat=4, seed=20260929)
    assert suite == build_suite(config, games_per_seat=4, seed=20260929)
    assert len(suite["games"]) == 48
    assert len({game["seed"] for game in suite["games"]}) == 48
    for opponent in ("random", "shortest", "wall-aware"):
        for seat in range(4):
            assert sum(game["opponent"] == opponent and game["seat"] == seat for game in suite["games"]) == 4


def test_modified_suite_is_rejected_before_model_load(tmp_path):
    config = load_config(CONFIG)
    suite = build_suite(config, games_per_seat=1, seed=20)
    suite["games"][0]["seed"] += 1
    with pytest.raises(ValueError, match="Frozen evaluation suite differs"):
        evaluate(SimpleNamespace(run_root=tmp_path, output=tmp_path / "output", update=0), config, suite)
    assert not (tmp_path / "output").exists()
