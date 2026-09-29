"""CPU-only launch validation for sequential four-worker D5 phases."""

import pytest

from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.phase_supervisor import supervise_role_phase
from self_play_grpo.training.process_supervisor import WorkerCommand


LAYOUT = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))


def workers(role):
    modules = LAYOUT.rollout_modules if role == "rollout" else LAYOUT.trainer_modules
    return tuple(WorkerCommand(role, rank, modules[rank], ("unused",)) for rank in range(4))


def test_incomplete_phase_rejected_without_output(tmp_path):
    target = tmp_path / "logs"
    with pytest.raises(ValueError, match="exactly four"):
        supervise_role_phase(
            workers("rollout")[:3], LAYOUT, target, role="rollout",
            trainer_master_port=29521, timeout_seconds=1,
            environ={"SLURM_GPUS_ON_NODE": "8"},
        )
    assert not target.exists()


def test_wrong_role_or_module_rejected_without_output(tmp_path):
    target = tmp_path / "logs"
    with pytest.raises(ValueError, match="exactly four"):
        supervise_role_phase(
            workers("trainer"), LAYOUT, target, role="rollout",
            trainer_master_port=29521, timeout_seconds=1,
            environ={"SLURM_GPUS_ON_NODE": "8"},
        )
    wrong = list(workers("rollout"))
    wrong[2] = WorkerCommand("rollout", 2, 6, ("unused",))
    with pytest.raises(ValueError, match="module differs"):
        supervise_role_phase(
            wrong, LAYOUT, target, role="rollout",
            trainer_master_port=29521, timeout_seconds=1,
            environ={"SLURM_GPUS_ON_NODE": "8"},
        )
    assert not target.exists()


def test_small_allocation_rejected_before_launch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No child may start in a one-HPU allocation")

    monkeypatch.setattr("self_play_grpo.training.phase_supervisor.subprocess.Popen", forbidden)
    target = tmp_path / "logs"
    with pytest.raises(ValueError, match="has 1 HPU"):
        supervise_role_phase(
            workers("trainer"), LAYOUT, target, role="trainer",
            trainer_master_port=29521, timeout_seconds=1,
            environ={"SLURM_GPUS_ON_NODE": "1"},
        )
    assert not target.exists()
