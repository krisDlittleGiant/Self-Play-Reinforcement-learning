"""CPU-only checks for parent-wide versus role-scoped D5 visibility."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.training import d5_preflight as module
from self_play_grpo.training.coordinator import RoleLayout


LAYOUT = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))
CONFIG = Path("self_play_grpo/configs/quoridor_outcome_64games.yaml")


def _accept_d4(monkeypatch):
    monkeypatch.setattr(module, "verify_d4_acceptance", lambda *args, **kwargs: (
        SimpleNamespace(summary_path="two"), SimpleNamespace(summary_path="four")
    ))


def _check(*, environ, worker_role=None):
    return module.verify_d5_launch_prerequisites(
        config_path=CONFIG, two_rank_d4_summary="two", four_rank_d4_summary="four",
        layout=LAYOUT, environ=environ, worker_role=worker_role,
    )


def test_parent_requires_all_eight_modules(monkeypatch):
    _accept_d4(monkeypatch)
    with pytest.raises(ValueError, match="outside HABANA_VISIBLE_MODULES"):
        _check(environ={"SLURM_GPUS_ON_NODE": "8", "HABANA_VISIBLE_MODULES": "0,1,2,3"})


@pytest.mark.parametrize("role,visible", [
    ("rollout", "0,1,2,3"), ("trainer", "4,5,6,7"),
])
def test_worker_accepts_only_exact_role_map(monkeypatch, role, visible):
    _accept_d4(monkeypatch)
    environ = {
        "SLURM_GPUS_ON_NODE": "8", "SP_GRPO_ROLE": role,
        "HABANA_VISIBLE_MODULES": visible,
    }
    assert _check(environ=environ, worker_role=role)["status"] == "preflight_passed_not_launched"
    with pytest.raises(ValueError, match="role-scoped visibility"):
        _check(environ=dict(environ, HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"), worker_role=role)
    with pytest.raises(ValueError, match="has 1 HPU"):
        _check(environ=dict(environ, SLURM_GPUS_ON_NODE="1"), worker_role=role)
