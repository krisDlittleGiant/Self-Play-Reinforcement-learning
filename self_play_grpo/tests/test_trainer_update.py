"""CPU-only contracts for one four-rank synchronized optimizer step."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from self_play_grpo.config import load_config
from self_play_grpo.training import trainer_update as module
from self_play_grpo.training.coordinator import BatchReceipt, PolicyDescriptor
from self_play_grpo.training.loss import LossOutput
from self_play_grpo.training.trainer_handoff import TrainerShardAdmission


class _FourEqualRanks:
    """Emulate collectives for four identical tiny CPU replicas."""

    ReduceOp = SimpleNamespace(SUM="sum", MIN="min", MAX="max")

    def is_initialized(self):
        return True

    def get_world_size(self):
        return 4

    def get_rank(self):
        return 0

    def broadcast(self, tensor, src):
        assert src == 0

    def all_reduce(self, tensor, op):
        if op == self.ReduceOp.SUM:
            tensor.mul_(4)


def _fixture():
    config = load_config(Path("self_play_grpo/configs/quoridor_outcome_64games.yaml"))
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trainer = SimpleNamespace(
        config=config, policy=SimpleNamespace(model=model),
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        update_index=0, policy_version="policy-000000",
    )
    descriptor = PolicyDescriptor(
        version="policy-000000", update_index=0,
        adapter_sha256="a" * 64, config_sha256="b" * 64,
        model_revision=config.model.revision,
        tokenizer_sha256="c" * 64, grammar_sha256="d" * 64,
        run_kind="production",
    )
    indices = tuple(tuple(range(rank * 16, (rank + 1) * 16)) for rank in range(4))
    receipt = BatchReceipt(
        manifest_sha256="e" * 64, policy_version=descriptor.version,
        adapter_sha256=descriptor.adapter_sha256,
        config_sha256=descriptor.config_sha256,
        game_ids=tuple(f"game-{index}" for index in range(64)),
        match_indices_by_rank=indices,
        turns_by_rank=(16, 16, 16, 16),
        owned_tokens_by_rank=(4, 4, 4, 4),
        max_replay_error=0.0,
    )
    admission = TrainerShardAdmission(
        rank=0, match_indices=indices[0],
        matches=tuple(SimpleNamespace(turns=[object()]) for _ in range(16)),
        receipt=receipt,
        replay_tolerance=2e-4,
    )
    runtime = SimpleNamespace(rank=0, world_size=4, device="cpu")
    return trainer, admission, descriptor, runtime


def test_one_synchronized_step_changes_params_and_populates_optimizer(monkeypatch):
    trainer, admission, descriptor, runtime = _fixture()
    monkeypatch.setattr(module, "checkpoint_qwen3_attention", lambda model: nullcontext())

    def backward(model, matches, **kwargs):
        assert len(matches) == 16
        assert kwargs["log_prob_function"] is module.constrained_log_probs_batched_shape
        value = model.weight.square().sum()
        value.backward()
        return LossOutput(
            loss=value.detach(), policy_loss=value.detach(),
            kl_loss=torch.tensor(0.0), mean_ratio=1.0,
            clip_fraction=0.0, owned_tokens=4, games=16,
            max_abs_log_prob_error=0.0,
        )

    monkeypatch.setattr(module, "backward_training_loss", backward)
    result = module.synchronized_trainer_update(
        trainer, admission, descriptor, runtime=runtime,
        torch=torch, dist=_FourEqualRanks(),
    )
    assert result.optimizer_steps == 1
    assert result.gradient_sync_phases == 1
    assert result.changed_parameter_tensors == 1
    assert result.parameter_sha256 != result.initial_parameter_sha256
    assert result.next_policy_version == "policy-000001"
    assert trainer.update_index == 1
    assert trainer.optimizer.state
    assert all(parameter.grad is None for parameter in trainer.policy.model.parameters())


def test_backward_failure_does_not_step_optimizer(monkeypatch):
    trainer, admission, descriptor, runtime = _fixture()
    monkeypatch.setattr(module, "checkpoint_qwen3_attention", lambda model: nullcontext())

    def fail(*args, **kwargs):
        raise RuntimeError("replay mismatch")

    monkeypatch.setattr(module, "backward_training_loss", fail)
    before = trainer.policy.model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="streamed backward failed"):
        module.synchronized_trainer_update(
            trainer, admission, descriptor, runtime=runtime,
            torch=torch, dist=_FourEqualRanks(),
        )
    assert torch.equal(before, trainer.policy.model.weight)
    assert trainer.update_index == 0
    assert not trainer.optimizer.state


def test_stale_trainer_state_rejected_before_collectives():
    trainer, admission, descriptor, runtime = _fixture()
    trainer.policy_version = "policy-000001"
    with pytest.raises(ValueError, match="trainer state differs"):
        module.synchronized_trainer_update(
            trainer, admission, descriptor, runtime=runtime,
            torch=torch, dist=_FourEqualRanks(),
        )


def _with_minibatches(trainer, count):
    from dataclasses import replace

    trainer.config = replace(
        trainer.config, training=replace(trainer.config.training, minibatches_per_update=count),
    )


def test_minibatches_take_one_synchronized_step_each_and_replay_only_before_steps(monkeypatch):
    trainer, admission, descriptor, runtime = _fixture()
    _with_minibatches(trainer, 4)
    monkeypatch.setattr(module, "checkpoint_qwen3_attention", lambda model: nullcontext())
    calls = []

    def backward(model, matches, **kwargs):
        calls.append((len(matches), kwargs["replay_tolerance"]))
        value = model.weight.square().sum()
        value.backward()
        return LossOutput(
            loss=value.detach(), policy_loss=value.detach(),
            kl_loss=torch.tensor(0.0), mean_ratio=1.0 + 0.1 * (len(calls) - 1),
            clip_fraction=0.0 if len(calls) == 1 else 0.25, owned_tokens=1, games=len(matches),
            max_abs_log_prob_error=0.0 if len(calls) == 1 else 0.05,
        )

    monkeypatch.setattr(module, "backward_training_loss", backward)
    result = module.synchronized_trainer_update(
        trainer, admission, descriptor, runtime=runtime,
        torch=torch, dist=_FourEqualRanks(),
    )
    assert calls == [(4, 2e-4), (4, None), (4, None), (4, None)]
    assert result.gradient_sync_phases == 4
    assert result.optimizer_steps == 1
    assert trainer.update_index == 1
    (state,) = trainer.optimizer.state.values()
    assert float(state["step"]) == 4.0
    assert result.max_abs_log_prob_error == 0.0
    assert result.owned_tokens == 4
    assert [row["max_abs_log_ratio"] for row in result.minibatches] == [0.0, 0.05, 0.05, 0.05]
    assert result.mean_ratio == pytest.approx(1.15)
    assert result.clip_fraction == pytest.approx(0.1875)


def test_single_minibatch_result_has_no_minibatch_rows(monkeypatch):
    trainer, admission, descriptor, runtime = _fixture()
    monkeypatch.setattr(module, "checkpoint_qwen3_attention", lambda model: nullcontext())

    def backward(model, matches, **kwargs):
        assert kwargs["replay_tolerance"] == 2e-4
        value = model.weight.square().sum()
        value.backward()
        return LossOutput(
            loss=value.detach(), policy_loss=value.detach(), kl_loss=torch.tensor(0.0),
            mean_ratio=1.0, clip_fraction=0.0, owned_tokens=4, games=16, max_abs_log_prob_error=0.0,
        )

    monkeypatch.setattr(module, "backward_training_loss", backward)
    result = module.synchronized_trainer_update(
        trainer, admission, descriptor, runtime=runtime, torch=torch, dist=_FourEqualRanks(),
    )
    assert result.minibatches == ()
    assert result.gradient_sync_phases == 1


def test_minibatch_token_shortfall_fails_before_the_final_step(monkeypatch):
    trainer, admission, descriptor, runtime = _fixture()
    _with_minibatches(trainer, 2)
    monkeypatch.setattr(module, "checkpoint_qwen3_attention", lambda model: nullcontext())

    def backward(model, matches, **kwargs):
        value = model.weight.square().sum()
        value.backward()
        return LossOutput(
            loss=value.detach(), policy_loss=value.detach(), kl_loss=torch.tensor(0.0),
            mean_ratio=1.0, clip_fraction=0.0, owned_tokens=1, games=len(matches),
            max_abs_log_prob_error=0.0,
        )

    monkeypatch.setattr(module, "backward_training_loss", backward)
    with pytest.raises(RuntimeError, match="streamed backward 2/2 failed"):
        module.synchronized_trainer_update(
            trainer, admission, descriptor, runtime=runtime, torch=torch, dist=_FourEqualRanks(),
        )
    (state,) = trainer.optimizer.state.values()
    assert float(state["step"]) == 1.0
    assert trainer.update_index == 0


def test_uneven_minibatches_are_rejected_before_any_collective():
    trainer, admission, descriptor, runtime = _fixture()
    _with_minibatches(trainer, 3)
    with pytest.raises(ValueError, match="cannot form 3 equal minibatches"):
        module.synchronized_trainer_update(
            trainer, admission, descriptor, runtime=runtime, torch=torch, dist=_FourEqualRanks(),
        )
    assert not trainer.optimizer.state


def test_expected_sync_phases_reads_objects_and_canonical_dicts():
    from self_play_grpo.config import expected_gradient_sync_phases

    trainer, *_ = _fixture()
    assert expected_gradient_sync_phases(trainer.config) == 1
    assert expected_gradient_sync_phases(trainer.config.to_dict()) == 1
    _with_minibatches(trainer, 4)
    assert expected_gradient_sync_phases(trainer.config) == 4
    assert expected_gradient_sync_phases(trainer.config.to_dict()) == 4
