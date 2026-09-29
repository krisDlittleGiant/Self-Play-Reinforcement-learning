"""Bounded four-worker launch for one synchronous D5 role phase.

Rollout, trainer and refresh phases are separate. This module imports no HPU
runtime and launches nothing until ``supervise_role_phase`` is called.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.process_supervisor import (
    WorkerCommand, WorkerFailure, _group_exists, _stop_groups,
    validate_worker_allocation, worker_environment,
)


def supervise_role_phase(
    workers: Sequence[WorkerCommand], layout: RoleLayout, output: str | Path,
    *, role: str, trainer_master_port: int, timeout_seconds: float,
    terminate_grace_seconds: float = 2.0,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Require exactly four matching workers and a clean process-group exit."""

    if role not in {"rollout", "trainer"}:
        raise ValueError("D5 phase role must be rollout or trainer")
    if timeout_seconds <= 0 or terminate_grace_seconds <= 0:
        raise ValueError("D5 phase timeout and teardown grace must be positive")
    if len(workers) != 4 or {(worker.role, worker.rank) for worker in workers} != {
        (role, rank) for rank in range(4)
    }:
        raise ValueError("D5 role phase needs exactly four unique workers")
    modules = layout.rollout_modules if role == "rollout" else layout.trainer_modules
    for worker in workers:
        worker.validate()
        if worker.module_id != modules[worker.rank]:
            raise ValueError("D5 phase worker module differs from assigned layout")
    base = dict(os.environ if environ is None else environ)
    validate_worker_allocation(layout, base)
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"D5 phase output must be new: {path}")
    path.mkdir(parents=True)
    handles = []
    started: list[tuple[WorkerCommand, subprocess.Popen[bytes], Path]] = []
    failure: str | None = None
    deadline = time.monotonic() + timeout_seconds
    try:
        for worker in workers:
            log_path = path / f"{role}-rank-{worker.rank:03d}.log"
            handle = log_path.open("xb")
            handles.append(handle)
            child = subprocess.Popen(
                worker.argv,
                env=worker_environment(
                    worker, layout, base, trainer_master_port=trainer_master_port,
                ),
                stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
            started.append((worker, child, log_path))
        while True:
            exit_codes = [child.poll() for _, child, _ in started]
            if any(code is not None and code != 0 for code in exit_codes):
                failure = "D5 phase worker exited unsuccessfully"
                break
            if all(code == 0 for code in exit_codes):
                if any(_group_exists(child) for _, child, _ in started):
                    failure = "D5 phase worker descendants remain after exit"
                break
            if time.monotonic() >= deadline:
                failure = "D5 phase worker deadline expired"
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    except BaseException as exc:
        failure = f"D5 phase supervision interrupted: {type(exc).__name__}: {exc}"
    finally:
        if failure is not None:
            _stop_groups([child for _, child, _ in started], terminate_grace_seconds)
        for handle in handles:
            handle.close()
    report: dict[str, object] = {
        "status": "failed" if failure else "workers_exited",
        "role": role, "failure": failure,
        "workers": [
            {
                "rank": worker.rank, "module_id": worker.module_id,
                "pid": child.pid, "returncode": child.poll(),
                "log": log_path.name,
            }
            for worker, child, log_path in started
        ],
    }
    target = path / "phase_summary.json"
    temporary = path / ".phase_summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    if failure is not None or len(started) != 4:
        raise WorkerFailure(failure or "D5 phase launch was incomplete", report)
    return report
