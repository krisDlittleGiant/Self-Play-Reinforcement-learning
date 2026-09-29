"""CPU-only gates for the opt-in resumed four-rank trainer worker."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.training import d6_trainer_worker as module


def test_d4_gate_fails_before_hpu_initialization(monkeypatch):
    monkeypatch.setattr(
        module, "verify_d5_launch_prerequisites",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("D4 evidence missing")),
    )
    monkeypatch.setattr(
        module, "initialize_hccl_process_group",
        lambda *args, **kwargs: pytest.fail("HPU initialized before D4 preflight"),
    )
    args = SimpleNamespace(
        config=Path("self_play_grpo/configs/quoridor_outcome_64games.yaml"),
        rollout_modules="0,1,2,3", trainer_modules="4,5,6,7",
        d4_two_summary=Path("missing-two"), d4_four_summary=Path("missing-four"),
    )
    with pytest.raises(FileNotFoundError, match="D4 evidence missing"):
        module.run_resumed_trainer_rank(args)


def test_incomplete_refresh_fails_before_hpu_initialization(monkeypatch):
    monkeypatch.setattr(module, "verify_d5_launch_prerequisites", lambda **kwargs: None)
    monkeypatch.setattr(module, "_binding", lambda *args: None)
    monkeypatch.setattr(module, "audit_recovery", lambda *args, **kwargs: SimpleNamespace(
        status="refresh_required", next_update=2,
    ))
    monkeypatch.setattr(
        module, "initialize_hccl_process_group",
        lambda *args, **kwargs: pytest.fail("HPU initialized before refresh preflight"),
    )
    args = SimpleNamespace(
        config=Path("self_play_grpo/configs/quoridor_outcome_64games.yaml"),
        rollout_modules="0,1,2,3", trainer_modules="4,5,6,7",
        d4_two_summary=Path("two"), d4_four_summary=Path("four"),
        rank=0, timeout_seconds=1, run_root=Path("run"),
        experiment_seed=11, run_id="d6-test",
    )
    with pytest.raises(RuntimeError, match="requires committed/refresh-complete"):
        module.run_resumed_trainer_rank(args)
