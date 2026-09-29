"""Actual model-free coordinator transition from update 1 to update 2."""

from dataclasses import replace
from pathlib import Path

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256
from self_play_grpo.training import coordinator as coordinator_module
from self_play_grpo.training.coordinator import (
    BatchReceipt, CycleError, Phase, PolicyDescriptor, RoleLayout,
)
from self_play_grpo.training.initial_cycle import _reconstruct_coordinator


CONFIG = Path(__file__).parents[1] / "configs" / "quoridor_outcome_64games.yaml"


def _inputs():
    config = load_config(CONFIG)
    prior = PolicyDescriptor(
        version="policy-000001", update_index=1, adapter_sha256="a" * 64,
        config_sha256=canonical_sha256(config.to_dict()),
        model_revision=config.model.revision,
        tokenizer_sha256="b" * 64, grammar_sha256="c" * 64,
        run_kind="production",
    )
    next_policy = replace(prior, version="policy-000002", update_index=2, adapter_sha256="d" * 64)
    receipt = BatchReceipt(
        manifest_sha256="e" * 64, policy_version=prior.version,
        adapter_sha256=prior.adapter_sha256, config_sha256=prior.config_sha256,
        game_ids=tuple(f"game-{i}" for i in range(64)),
        match_indices_by_rank=tuple(tuple(range(16 * rank, 16 * (rank + 1))) for rank in range(4)),
        turns_by_rank=(16, 16, 16, 16), owned_tokens_by_rank=(16, 16, 16, 16),
        max_replay_error=0.0,
    )
    reports = tuple({"update": {
        "turns": 16, "owned_tokens": 16, "max_abs_log_prob_error": 0.0,
        "parameter_sha256": "f" * 64, "optimizer_sha256": "0" * 64,
        "optimizer_steps": 2, "gradient_sync_phases": 1,
        "changed_parameter_tensors": 252,
    }} for _ in range(4))
    return config, prior, next_policy, receipt, reports


def test_second_update_accepts_cumulative_optimizer_step(monkeypatch, tmp_path):
    config, prior, next_policy, receipt, reports = _inputs()
    monkeypatch.setattr(coordinator_module, "verify_completed_batch", lambda *args, **kwargs: receipt)
    coordinator = _reconstruct_coordinator(
        layout=RoleLayout((0, 1, 2, 3), (4, 5, 6, 7)),
        prior=prior, config=config, rollout_root=tmp_path,
        receipt=receipt, next_policy=next_policy, reports=reports,
    )
    assert coordinator.phase is Phase.UPDATED
    assert len(coordinator.updated) == 4


def test_per_call_optimizer_step_is_not_cumulative(monkeypatch, tmp_path):
    config, prior, next_policy, receipt, reports = _inputs()
    monkeypatch.setattr(coordinator_module, "verify_completed_batch", lambda *args, **kwargs: receipt)
    bad = tuple({"update": {**row["update"], "optimizer_steps": 1}} for row in reports)
    with pytest.raises(CycleError, match="metadata is incompatible"):
        _reconstruct_coordinator(
            layout=RoleLayout((0, 1, 2, 3), (4, 5, 6, 7)),
            prior=prior, config=config, rollout_root=tmp_path,
            receipt=receipt, next_policy=next_policy, reports=bad,
        )
