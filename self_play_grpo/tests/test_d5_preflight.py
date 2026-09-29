"""CPU-only admission tests for the current 64-game D5 profile."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.training import d5_preflight as module
from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.d5_preflight import (
    supervise_guarded_workers, verify_d5_launch_prerequisites,
)


LAYOUT = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))
CONFIG = Path("self_play_grpo/configs/quoridor_outcome_64games.yaml")


def test_real_64_game_config_passes_with_mock_d4_evidence(monkeypatch):
    called = []

    def d4(*args, **kwargs):
        called.append((args, kwargs))
        return (
            SimpleNamespace(summary_path="d4-two/summary.json"),
            SimpleNamespace(summary_path="d4-four/summary.json"),
        )

    monkeypatch.setattr(module, "verify_d4_acceptance", d4)
    result = verify_d5_launch_prerequisites(
        config_path=CONFIG, two_rank_d4_summary="d4-two/summary.json",
        four_rank_d4_summary="d4-four/summary.json", layout=LAYOUT,
        environ={"SLURM_GPUS_ON_NODE": "8"},
    )
    assert result["games_per_rollout_rank"] == 16
    assert result["parallel_games_per_rollout_rank"] == 2
    assert called[0][1]["layout"] == LAYOUT


def test_one_hpu_allocation_fails_after_d4_read(monkeypatch):
    monkeypatch.setattr(module, "verify_d4_acceptance", lambda *args, **kwargs: (
        SimpleNamespace(summary_path="two"), SimpleNamespace(summary_path="four")
    ))
    with pytest.raises(ValueError, match="has 1 HPU"):
        verify_d5_launch_prerequisites(
            config_path=CONFIG, two_rank_d4_summary="two",
            four_rank_d4_summary="four", layout=LAYOUT,
            environ={"SLURM_GPUS_ON_NODE": "1"},
        )


def test_guarded_launch_does_not_create_output_when_d4_missing(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("D4 four-rank summary is missing")
    monkeypatch.setattr(module, "verify_d4_acceptance", missing)
    output = tmp_path / "not-created"
    with pytest.raises(FileNotFoundError, match="D4 four-rank"):
        supervise_guarded_workers(
            (), LAYOUT, output, config_path=CONFIG,
            two_rank_d4_summary="two", four_rank_d4_summary="four",
            trainer_master_port=29531, timeout_seconds=1,
            environ={"SLURM_GPUS_ON_NODE": "8"},
        )
    assert not output.exists()


def test_baseline_rejects_16_game_config(monkeypatch):
    monkeypatch.setattr(module, "verify_d4_acceptance", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="64 complete games"):
        verify_d5_launch_prerequisites(
            config_path="self_play_grpo/configs/quoridor_outcome.yaml",
            two_rank_d4_summary="two", four_rank_d4_summary="four",
            layout=LAYOUT, environ={"SLURM_GPUS_ON_NODE": "8"},
        )
