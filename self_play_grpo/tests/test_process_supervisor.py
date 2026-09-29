"""No-HPU tests for the bounded eight-worker process boundary."""

import json
import sys

import pytest

from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.process_supervisor import (
    WorkerFailure,
    build_worker_commands,
    supervise_workers,
    validate_worker_allocation,
    worker_environment,
)


LAYOUT = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))


def commands(code: str, *, failing_rank: int | None = None):
    rollout = [(sys.executable, "-c", code) for _ in range(4)]
    trainer = [
        (sys.executable, "-c", "import sys; sys.exit(7)" if rank == failing_rank else code)
        for rank in range(4)
    ]
    return build_worker_commands(LAYOUT, rollout_argv=rollout, trainer_argv=trainer)


def test_worker_binding_is_disjoint_and_rollout_has_no_trainer_rendezvous():
    workers = commands("print('ok')")
    base = {"RANK": "99", "MASTER_PORT": "1111", "SLURM_GPUS_ON_NODE": "8"}
    rollout = worker_environment(workers[1], LAYOUT, base, trainer_master_port=29531)
    trainer = worker_environment(workers[6], LAYOUT, base, trainer_master_port=29531)
    assert rollout["HABANA_VISIBLE_MODULES"] == "0,1,2,3"
    assert rollout["HLS_MODULE_ID"] == "1"
    assert "RANK" not in rollout and "MASTER_PORT" not in rollout
    assert trainer["HABANA_VISIBLE_MODULES"] == "4,5,6,7"
    assert trainer["RANK"] == "2" and trainer["LOCAL_RANK"] == "2"
    assert trainer["MASTER_PORT"] == "29531"


def test_known_one_hpu_allocation_is_rejected():
    with pytest.raises(ValueError, match="has 1 HPU"):
        validate_worker_allocation(LAYOUT, {"SLURM_GPUS_ON_NODE": "1"})
    with pytest.raises(ValueError, match="outside HABANA_VISIBLE_MODULES"):
        validate_worker_allocation(LAYOUT, {"HABANA_VISIBLE_MODULES": "0,1,2,3"})


def test_all_eight_fake_workers_exit_and_leave_logs(tmp_path):
    workers = commands("import os; print(os.environ['SP_GRPO_ROLE'], os.environ['SP_GRPO_RANK'])")
    report = supervise_workers(
        workers, LAYOUT, tmp_path / "workers", trainer_master_port=29531,
        timeout_seconds=5, environ={"SLURM_GPUS_ON_NODE": "8", "PATH": "/usr/bin:/bin"},
    )
    assert report["status"] == "workers_exited"
    assert len(report["workers"]) == 8
    assert all(item["returncode"] == 0 for item in report["workers"])
    assert (tmp_path / "workers" / "rollout-rank-000.log").read_text().strip() == "rollout 0"
    assert (tmp_path / "workers" / "trainer-rank-003.log").read_text().strip() == "trainer 3"
    assert json.loads((tmp_path / "workers" / "worker_summary.json").read_text())["status"] == "workers_exited"


def test_failed_worker_terminates_peers_and_preserves_report(tmp_path):
    workers = commands("import time; time.sleep(3)", failing_rank=2)
    with pytest.raises(WorkerFailure) as error:
        supervise_workers(
            workers, LAYOUT, tmp_path / "failure", trainer_master_port=29531,
            timeout_seconds=5, terminate_grace_seconds=0.25,
            environ={"SLURM_GPUS_ON_NODE": "8", "PATH": "/usr/bin:/bin"},
        )
    report = error.value.report
    assert report["status"] == "failed"
    assert len(report["workers"]) == 8
    assert all(item["returncode"] is not None for item in report["workers"])
    assert (tmp_path / "failure" / "worker_summary.json").is_file()


def test_incomplete_workers_rejected_before_process_creation(tmp_path):
    with pytest.raises(ValueError, match=r"every 4\+4 slot"):
        supervise_workers(
            commands("print('ok')")[:-1], LAYOUT, tmp_path / "not-created",
            trainer_master_port=29531, timeout_seconds=1,
            environ={"SLURM_GPUS_ON_NODE": "8"},
        )
    assert not (tmp_path / "not-created").exists()
