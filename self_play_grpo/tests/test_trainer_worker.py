"""CPU-only admission and reporting tests for the opt-in trainer process."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from self_play_grpo.training import trainer_worker as module


class _FourEqualRanks:
    ReduceOp = SimpleNamespace(SUM="sum", MAX="max")

    def all_reduce(self, tensor, op):
        if op == self.ReduceOp.SUM:
            tensor.mul_(4)


def test_module_list_and_physical_binding(monkeypatch):
    assert module._modules("4,5,6,7") == (4, 5, 6, 7)
    with pytest.raises(ValueError, match="distinct"):
        module._modules("4,5,5,7")
    for name, value in {
        "SP_GRPO_ROLE": "trainer", "SP_GRPO_RANK": "2",
        "SP_GRPO_MODULE_ID": "6", "HLS_MODULE_ID": "6",
        "HABANA_VISIBLE_MODULES": "4,5,6,7",
    }.items():
        monkeypatch.setenv(name, value)
    module._binding(2, (4, 5, 6, 7))
    monkeypatch.setenv("HLS_MODULE_ID", "5")
    with pytest.raises(RuntimeError, match="physical-module binding"):
        module._binding(2, (4, 5, 6, 7))


def test_global_metrics_use_all_ranks_not_rank_zero_only():
    result = SimpleNamespace(
        loss=-0.5, policy_loss=-0.5, kl_loss=0.0,
        mean_ratio=1.1, clip_fraction=0.25,
        owned_tokens=12, turns=25, max_abs_log_prob_error=1e-5,
        grad_norm=0.7,
    )
    trainer = SimpleNamespace(update_index=1, policy_version="policy-000001")
    metrics = module._global_metrics(
        result, trainer=trainer, torch=torch, dist=_FourEqualRanks(),
        device=torch.device("cpu"),
    )
    assert metrics.games == 64
    assert metrics.turns == 100
    assert metrics.owned_tokens == 48
    assert metrics.loss == pytest.approx(-0.5)
    assert metrics.mean_ratio_before_step == pytest.approx(1.1)
    assert metrics.clip_fraction_before_step == pytest.approx(0.25)


def test_d4_preflight_failure_occurs_before_hpu_initialization(monkeypatch):
    def reject(**kwargs):
        raise FileNotFoundError("D4 four-rank hardware evidence missing")

    def forbidden(*args, **kwargs):
        raise AssertionError("HPU initialization must not occur")

    monkeypatch.setattr(module, "verify_d5_launch_prerequisites", reject)
    monkeypatch.setattr(module, "initialize_hccl_process_group", forbidden)
    args = SimpleNamespace(
        config=Path("self_play_grpo/configs/quoridor_outcome_64games.yaml"),
        rollout_modules="0,1,2,3", trainer_modules="4,5,6,7",
        d4_two_summary=Path("missing-two"),
        d4_four_summary=Path("missing-four"),
    )
    with pytest.raises(FileNotFoundError, match="D4 four-rank"):
        module.run_initial_trainer_rank(args)
