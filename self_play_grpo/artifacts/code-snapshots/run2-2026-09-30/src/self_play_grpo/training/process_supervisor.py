"""Bounded single-node 4+4 worker supervision without implicit HPU launch.

This is a process boundary, not the D5 training algorithm.  The caller must
provide eight explicit worker commands after checking D4 acceptance and the
current resource allocation.  Workers are supervised as process groups so a
failed worker cannot leave its descendants running indefinitely.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from self_play_grpo.training.coordinator import RoleLayout


@dataclass(frozen=True)
class WorkerCommand:
    role: str
    rank: int
    module_id: int
    argv: tuple[str, ...]

    def validate(self) -> None:
        if self.role not in ("rollout", "trainer") or self.rank not in range(4):
            raise ValueError("Worker role and rank must be one of the D5 4+4 slots")
        if type(self.module_id) is not int or self.module_id < 0:
            raise ValueError("Worker module_id must be a non-negative integer")
        if not self.argv or any(not isinstance(arg, str) or not arg for arg in self.argv):
            raise ValueError("Worker argv must be a nonempty argument vector")


def build_worker_commands(
    layout: RoleLayout,
    *,
    rollout_argv: Sequence[Sequence[str]],
    trainer_argv: Sequence[Sequence[str]],
) -> tuple[WorkerCommand, ...]:
    """Bind eight caller-supplied commands to eight distinct physical modules."""

    if len(rollout_argv) != 4 or len(trainer_argv) != 4:
        raise ValueError("D5 requires exactly four rollout and four trainer commands")
    workers = tuple(
        WorkerCommand(role, rank, modules[rank], tuple(commands[rank]))
        for role, modules, commands in (
            ("rollout", layout.rollout_modules, rollout_argv),
            ("trainer", layout.trainer_modules, trainer_argv),
        )
        for rank in range(4)
    )
    for worker in workers:
        worker.validate()
    return workers


def validate_worker_allocation(
    layout: RoleLayout, environ: Mapping[str, str]
) -> None:
    """Reject a known smaller allocation; host-wide device files are not proof.

    An absent allocation count is intentionally inconclusive.  The live
    launcher must still make a real HPU acquire/collective check before model
    loading; this function does not grant or reserve devices.
    """

    count = environ.get("SLURM_GPUS_ON_NODE")
    if count is not None:
        try:
            allocated = int(count)
        except ValueError as exc:
            raise ValueError("SLURM_GPUS_ON_NODE is not an integer") from exc
        if allocated < 8:
            raise ValueError(
                f"Current process allocation has {allocated} HPU(s), but D5 needs eight"
            )
    visible = environ.get("HABANA_VISIBLE_MODULES")
    if visible is not None:
        try:
            allowed = {int(part.strip()) for part in visible.split(",")}
        except ValueError as exc:
            raise ValueError("HABANA_VISIBLE_MODULES is malformed") from exc
        requested = set(layout.rollout_modules + layout.trainer_modules)
        if not requested.issubset(allowed):
            raise ValueError("Requested D5 modules are outside HABANA_VISIBLE_MODULES")


def worker_environment(
    worker: WorkerCommand,
    layout: RoleLayout,
    base: Mapping[str, str],
    *,
    trainer_master_port: int,
) -> dict[str, str]:
    """Produce explicit rank/module binding for one already-authorized worker."""

    worker.validate()
    if not 1 <= trainer_master_port <= 65535:
        raise ValueError("Trainer rendezvous port is outside [1, 65535]")
    modules = layout.rollout_modules if worker.role == "rollout" else layout.trainer_modules
    if modules[worker.rank] != worker.module_id:
        raise ValueError("Worker module differs from its role/rank slot")
    result = dict(base)
    result["HABANA_VISIBLE_MODULES"] = ",".join(str(module) for module in modules)
    result["HLS_MODULE_ID"] = str(worker.module_id)
    result["SP_GRPO_ROLE"] = worker.role
    result["SP_GRPO_RANK"] = str(worker.rank)
    result["SP_GRPO_MODULE_ID"] = str(worker.module_id)
    if worker.role == "trainer":
        result.update({
            "RANK": str(worker.rank),
            "LOCAL_RANK": str(worker.rank),
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "4",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(trainer_master_port),
        })
    else:
        # Rollout workers are independent inference processes.  They must
        # never inherit the four-rank trainer rendezvous from the parent.
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            result.pop(name, None)
    return result


class WorkerFailure(RuntimeError):
    def __init__(self, message: str, report: dict[str, object]):
        super().__init__(message)
        self.report = report


def _group_exists(process: subprocess.Popen[bytes]) -> bool:
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False


def _stop_groups(processes: Sequence[subprocess.Popen[bytes]], grace_seconds: float) -> None:
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=max(0.1, grace_seconds))
        except subprocess.TimeoutExpired:
            # A process stuck in an uninterruptible kernel wait is reported as
            # still running; never claim a clean teardown in that case.
            pass


def supervise_workers(
    workers: Sequence[WorkerCommand],
    layout: RoleLayout,
    output: str | Path,
    *,
    trainer_master_port: int,
    timeout_seconds: float,
    terminate_grace_seconds: float = 2.0,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Opt-in launch; return zero-status evidence only after every child exits.

    This accepts commands supplied by a higher-level D5 driver.  It never
    synthesizes a production trainer command from the validation-only D3 CLI.
    """

    if timeout_seconds <= 0 or terminate_grace_seconds <= 0:
        raise ValueError("Worker timeout and termination grace must be positive")
    expected = {(role, rank) for role in ("rollout", "trainer") for rank in range(4)}
    if len(workers) != 8 or {(worker.role, worker.rank) for worker in workers} != expected:
        raise ValueError("D5 supervisor needs one unique worker for every 4+4 slot")
    for worker in workers:
        worker.validate()
        if (layout.rollout_modules if worker.role == "rollout" else layout.trainer_modules)[worker.rank] != worker.module_id:
            raise ValueError("Worker module binding differs from D5 layout")
    base = dict(os.environ if environ is None else environ)
    validate_worker_allocation(layout, base)
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Worker output must be new: {path}")
    path.mkdir(parents=True)
    processes: list[subprocess.Popen[bytes]] = []
    handles = []
    started: list[tuple[WorkerCommand, subprocess.Popen[bytes], Path]] = []
    deadline = time.monotonic() + timeout_seconds
    failure: str | None = None
    try:
        for worker in workers:
            log_path = path / f"{worker.role}-rank-{worker.rank:03d}.log"
            handle = log_path.open("xb")
            handles.append(handle)
            child = subprocess.Popen(
                worker.argv,
                env=worker_environment(
                    worker, layout, base, trainer_master_port=trainer_master_port
                ),
                stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
            processes.append(child)
            started.append((worker, child, log_path))
        while True:
            exited = [child.poll() for child in processes]
            if any(code is not None and code != 0 for code in exited):
                failure = "One or more D5 workers exited unsuccessfully"
                break
            if all(code == 0 for code in exited):
                if any(_group_exists(child) for child in processes):
                    failure = "D5 worker descendants remain after worker exit"
                break
            if time.monotonic() >= deadline:
                failure = "D5 worker deadline expired"
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    except BaseException as exc:
        failure = f"D5 worker supervision interrupted: {type(exc).__name__}: {exc}"
    finally:
        if failure is not None:
            _stop_groups(processes, terminate_grace_seconds)
        for handle in handles:
            handle.close()
    report: dict[str, object] = {
        "status": "failed" if failure else "workers_exited",
        "failure": failure,
        "workers": [
            {
                "role": worker.role,
                "rank": worker.rank,
                "module_id": worker.module_id,
                "pid": child.pid,
                "returncode": child.poll(),
                "log": log_path.name,
            }
            for worker, child, log_path in started
        ],
    }
    target = path / "worker_summary.json"
    temporary = path / ".worker_summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    if failure is not None or len(started) != 8:
        raise WorkerFailure(failure or "D5 worker launch was incomplete", report)
    return report
