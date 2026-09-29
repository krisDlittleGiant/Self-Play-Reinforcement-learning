"""CPU-only checks for the D5 trainer-to-rollout refresh gate."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from self_play_grpo.training import policy_refresh as module


def _probe(values=None):
    return {
        "game_index": 0, "joint_step": 0, "match_sha256": "a" * 64,
        "completion_token_ids": [12, 34],
        "log_probs": [-0.5, -0.1] if values is None else values,
    }


def test_probe_requires_exact_identity_and_finite_probabilities():
    expected = _probe()
    assert module.compare_probe(expected, _probe([-0.50001, -0.1]), 2e-4) == pytest.approx(1e-5)
    with pytest.raises(RuntimeError, match="probability mismatch"):
        module.compare_probe(expected, _probe([-0.6, -0.1]), 2e-4)
    corrupted = dict(expected, match_sha256="b" * 64)
    with pytest.raises(ValueError, match="match_sha256 differs"):
        module.compare_probe(expected, corrupted, 2e-4)
    with pytest.raises(ValueError, match="values are invalid"):
        module.validate_probe(_probe([float("nan"), -0.1]))


def test_policy_probe_restores_model_training_mode(monkeypatch):
    sample = SimpleNamespace(completion_token_ids=(12, 34))
    monkeypatch.setattr(module, "source_sample", lambda root: (sample, "a" * 64))
    observed = []

    def replay(model, actual_sample):
        observed.append((model.training, actual_sample))
        return torch.tensor([-0.5, -0.1])

    monkeypatch.setattr(module, "constrained_log_probs_batched_shape", replay)
    model = torch.nn.Linear(2, 2)
    model.train()
    result = module.policy_probe(model, Path("unused"))
    assert result["log_probs"] == pytest.approx([-0.5, -0.1])
    assert result["match_sha256"] == "a" * 64
    assert observed == [(False, sample)]
    assert model.training


def test_preflight_failure_before_model_load(monkeypatch, tmp_path):
    def reject(**kwargs):
        raise FileNotFoundError("D4 hardware acceptance missing")

    def forbidden(*args, **kwargs):
        raise AssertionError("Model load must not occur")

    monkeypatch.setattr(module, "verify_d5_launch_prerequisites", reject)
    monkeypatch.setattr(module.ConstrainedLLMPolicy, "load", forbidden)
    args = SimpleNamespace(
        config=Path("self_play_grpo/configs/quoridor_outcome_64games.yaml"),
        rollout_modules="0,1,2,3", trainer_modules="4,5,6,7",
        d4_two_summary=tmp_path / "two", d4_four_summary=tmp_path / "four",
    )
    with pytest.raises(FileNotFoundError, match="D4 hardware acceptance missing"):
        module.refresh_rank(args)
