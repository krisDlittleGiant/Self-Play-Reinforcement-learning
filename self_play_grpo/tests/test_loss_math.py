from types import SimpleNamespace
import weakref

import pytest

from self_play_grpo.policies.llm import _checkpoint_attention_registry
from self_play_grpo.training import loss as loss_module
from self_play_grpo.training.loss import (
    backward_training_loss,
    clipped_token_terms,
    compute_training_loss,
)

pytestmark = pytest.mark.torch


@pytest.fixture
def torch():
    try:
        import torch as torch_module
    except (ImportError, RuntimeError, AssertionError) as exc:
        pytest.skip(f"compatible Torch runtime unavailable: {exc}")
    return torch_module


@pytest.mark.parametrize("advantage,expected_sign", [(1.0, -1), (-1.0, 1)])
def test_gradient_direction_at_behavior_policy(torch, advantage, expected_sign) -> None:
    new = torch.tensor([0.0], requires_grad=True)
    old = torch.tensor([0.0])
    mask = torch.tensor([1.0])
    terms, ratios = clipped_token_terms(new, old, advantage, mask, 0.2)
    loss = -terms.sum()
    loss.backward()
    assert ratios.item() == pytest.approx(1.0)
    assert int(torch.sign(new.grad).item()) == expected_sign


def test_only_owned_tokens_contribute(torch) -> None:
    new = torch.tensor([0.0, 0.0], requires_grad=True)
    old = torch.tensor([0.0, 0.0])
    terms, _ = clipped_token_terms(new, old, 1.0, torch.tensor([1.0, 0.0]), 0.2)
    (-terms.sum()).backward()
    assert new.grad[0] != 0
    assert new.grad[1] == 0


def test_streaming_backward_matches_summed_objective(torch, monkeypatch) -> None:
    class ScalarPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_prob = torch.nn.Parameter(torch.tensor(0.0))

    sample = SimpleNamespace(
        completion_token_ids=(1,),
        behavior_log_probs=(0.0,),
        loss_mask=(1,),
    )
    turns = [
        SimpleNamespace(
            policy_sample=sample,
            credit=SimpleNamespace(training_advantage=advantage),
        )
        for advantage in (1.0, -0.25)
    ]

    class Match:
        policy_version = "p0"

        def __init__(self) -> None:
            self.turns = turns

        def validate(self) -> None:
            return None

    monkeypatch.setattr(
        loss_module,
        "constrained_log_probs",
        lambda model, unused_sample: model.log_prob.reshape(1),
    )
    summed_model = ScalarPolicy()
    summed = compute_training_loss(
        summed_model, [Match()], clip_epsilon=0.2, loss_normalizer_per_game=10
    )
    summed.loss.backward()

    streamed_model = ScalarPolicy()
    streamed = backward_training_loss(
        streamed_model, [Match()], clip_epsilon=0.2, loss_normalizer_per_game=10
    )
    assert streamed.loss.item() == pytest.approx(summed.loss.item())
    assert streamed_model.log_prob.grad.item() == pytest.approx(
        summed_model.log_prob.grad.item()
    )
    assert streamed.max_abs_log_prob_error == pytest.approx(0.0)


def test_streaming_backward_releases_previous_action_graph_before_next_forward(
    torch,
    monkeypatch,
) -> None:
    class ScalarPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_prob = torch.nn.Parameter(torch.tensor(0.0))

    sample = SimpleNamespace(
        completion_token_ids=(1,),
        behavior_log_probs=(0.0,),
        loss_mask=(1,),
    )
    turns = [
        SimpleNamespace(
            policy_sample=sample,
            credit=SimpleNamespace(training_advantage=1.0),
        )
        for _ in range(3)
    ]

    class Match:
        policy_version = "p0"

        def __init__(self) -> None:
            self.turns = turns

        def validate(self) -> None:
            return None

    previous_outputs: list[weakref.ReferenceType] = []

    def replay(model, unused_sample):
        if previous_outputs:
            assert previous_outputs[-1]() is None
        output = (model.log_prob + 0.0).reshape(1)
        previous_outputs.append(weakref.ref(output))
        return output

    monkeypatch.setattr(loss_module, "constrained_log_probs", replay)
    model = ScalarPolicy()
    result = backward_training_loss(
        model,
        [Match()],
        clip_epsilon=0.2,
        loss_normalizer_per_game=10,
    )
    assert result.owned_tokens == 3
    assert result.loss.device.type == "cpu"
    assert previous_outputs[-1]() is None


def test_attention_checkpoint_preserves_output_and_gradients(torch) -> None:
    class Registry:
        def get_interface(self, implementation, default):
            assert implementation == "toy"
            return default

    registry = Registry()

    def attention(module, query, key, value, attention_mask, **kwargs):
        del module, attention_mask, kwargs
        return (query * key + value.square()).sum(), None

    expected_inputs = [
        torch.tensor([1.0, 2.0], requires_grad=True),
        torch.tensor([3.0, 4.0], requires_grad=True),
        torch.tensor([5.0, 6.0], requires_grad=True),
    ]
    expected, _ = attention(None, *expected_inputs, None)
    expected.backward()

    actual_inputs = [tensor.detach().clone().requires_grad_() for tensor in expected_inputs]
    with _checkpoint_attention_registry(registry):
        replay = registry.get_interface("toy", attention)
        actual, weights = replay(None, *actual_inputs, None)
        actual.backward()

    assert weights is None
    assert "get_interface" not in vars(registry)
    torch.testing.assert_close(actual, expected)
    for observed, baseline in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(observed.grad, baseline.grad)


def test_nonfinite_replay_rejected_before_backward(torch) -> None:
    sample = SimpleNamespace(
        completion_token_ids=(1,),
        behavior_log_probs=(0.0,),
        loss_mask=(1,),
    )

    class Match:
        policy_version = "p0"

        def __init__(self) -> None:
            self.turns = [
                SimpleNamespace(
                    policy_sample=sample,
                    credit=SimpleNamespace(training_advantage=1.0),
                )
            ]

        def validate(self) -> None:
            return None

    model = torch.nn.Linear(1, 1, bias=False)

    def nonfinite_replay(model, unused_sample):
        return model.weight.reshape(1) * float("nan")

    with pytest.raises(RuntimeError, match="Non-finite behavior replay error"):
        backward_training_loss(
            model,
            [Match()],
            clip_epsilon=0.2,
            loss_normalizer_per_game=10,
            log_prob_function=nonfinite_replay,
            replay_tolerance=2e-4,
        )
    assert model.weight.grad is None
