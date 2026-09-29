"""Small real CPU subprocesses verify teardown and inherited run locking."""
import sys
import pytest
from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.multi_run import run_lock
from self_play_grpo.training.multi_supervisor import supervise_role_phase
from self_play_grpo.training.process_supervisor import WorkerCommand, WorkerFailure


@pytest.mark.parametrize("fail", [False, True])
def test_four_worker_phase_and_failed_peer_teardown(tmp_path, fail):
    layout = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))
    code = "import os; print(os.environ['SP_GRPO_RANK'])"
    workers = tuple(WorkerCommand("trainer", rank, rank + 4, (sys.executable, "-c",
                    "raise SystemExit(7)" if fail and rank == 1 else code)) for rank in range(4))
    with run_lock(tmp_path) as fd:
        kwargs = dict(role="trainer", trainer_master_port=29642, timeout_seconds=5,
                      lock_fd=fd, environ={"SLURM_GPUS_ON_NODE": "8", "PATH": "/usr/bin:/bin"})
        if fail:
            with pytest.raises(WorkerFailure) as exc:
                supervise_role_phase(workers, layout, tmp_path / "logs", **kwargs)
            report = exc.value.report
        else:
            report = supervise_role_phase(workers, layout, tmp_path / "logs", **kwargs)
    assert len(report["workers"]) == 4
    assert all(row["returncode"] is not None for row in report["workers"])
    with run_lock(tmp_path):
        pass
