"""Fail-closed admission for the current 64-game, four-plus-four D5 profile."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from self_play_grpo.config import load_config
from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.d4_acceptance import verify_d4_acceptance
from self_play_grpo.training.process_supervisor import (
    WorkerCommand,
    supervise_workers,
    validate_worker_allocation,
)


def verify_d5_launch_prerequisites(
    *,
    config_path: str | Path,
    two_rank_d4_summary: str | Path,
    four_rank_d4_summary: str | Path,
    layout: RoleLayout,
    environ: Mapping[str, str] | None = None,
    worker_role: str | None = None,
) -> dict[str, object]:
    """Require the exact current experiment and both real D4 evidence gates.

    This does not claim physical HPU acquisition. The workers still need to
    initialize their assigned devices and prove the trainer HCCL collective.
    """

    config = load_config(Path(config_path))
    if config.rollout.games_per_update != 64:
        raise ValueError("Current D5 profile requires 64 complete games per update")
    if config.rollout.parallel_games_per_rank != 2:
        raise ValueError("Current D5 profile requires two active games per rollout rank")
    if config.training.reward_mode != "outcome":
        raise ValueError("Current D5 baseline requires engine-outcome reward")
    if config.training.group_size != 4 or config.environment.players != 4:
        raise ValueError("Current D5 baseline requires complete four-seat games")
    if config.training.optimizer_epochs_per_batch != 1:
        raise ValueError("Current D5 baseline requires one optimizer epoch per batch")
    two, four = verify_d4_acceptance(
        two_rank_d4_summary, four_rank_d4_summary, layout=layout,
    )
    environment = dict(os.environ if environ is None else environ)
    if worker_role is not None:
        if worker_role not in {"rollout", "trainer"} or environment.get("SP_GRPO_ROLE") != worker_role:
            raise ValueError("D5 role-scoped preflight requires a matching worker role")
        modules = layout.rollout_modules if worker_role == "rollout" else layout.trainer_modules
        if environment.get("HABANA_VISIBLE_MODULES") != ",".join(str(module) for module in modules):
            raise ValueError("D5 role-scoped visibility differs from the assigned modules")
        # The parent already checks eight-module visibility. A worker is
        # intentionally restricted to only its own four-module role map.
        environment.pop("HABANA_VISIBLE_MODULES")
    validate_worker_allocation(layout, environment)
    return {
        "status": "preflight_passed_not_launched",
        "games_per_update": 64,
        "games_per_rollout_rank": 16,
        "parallel_games_per_rollout_rank": 2,
        "rollout_modules": list(layout.rollout_modules),
        "trainer_modules": list(layout.trainer_modules),
        "two_rank_d4_summary": two.summary_path,
        "four_rank_d4_summary": four.summary_path,
    }


def supervise_guarded_workers(
    workers: Sequence[WorkerCommand],
    layout: RoleLayout,
    output: str | Path,
    *,
    config_path: str | Path,
    two_rank_d4_summary: str | Path,
    four_rank_d4_summary: str | Path,
    trainer_master_port: int,
    timeout_seconds: float,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Only launch caller-supplied workers after D4 and allocation preflight.

    Exiting eight processes is not a successful GRPO cycle. A higher-level
    driver must still validate phase reports, batch, update, checkpoint and
    refresh using the coordinator and handoff contracts.
    """

    verify_d5_launch_prerequisites(
        config_path=config_path,
        two_rank_d4_summary=two_rank_d4_summary,
        four_rank_d4_summary=four_rank_d4_summary,
        layout=layout,
        environ=environ,
    )
    return supervise_workers(
        workers, layout, output,
        trainer_master_port=trainer_master_port,
        timeout_seconds=timeout_seconds,
        environ=environ,
    )
