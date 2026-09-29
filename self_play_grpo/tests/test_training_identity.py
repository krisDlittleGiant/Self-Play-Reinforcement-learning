"""CPU-only tests for D5 model/optimizer content identities."""

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

from self_play_grpo.training.identity import (
    optimizer_state_sha256,
    trainable_parameter_sha256,
)


def test_trainable_identity_is_stable_across_mapping_order_and_device():
    first = torch.nn.Module()
    first.register_parameter("alpha", torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16)))
    first.register_parameter("beta", torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float32)))
    second = torch.nn.Module()
    second.register_parameter("beta", torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float32)))
    second.register_parameter("alpha", torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16)))
    assert trainable_parameter_sha256(first) == trainable_parameter_sha256(second)
    with torch.no_grad():
        second.alpha[0] = 4.0
    assert trainable_parameter_sha256(first) != trainable_parameter_sha256(second)


def test_trainable_identity_includes_dtype_name_and_trainability():
    first = torch.nn.Linear(2, 1, bias=False)
    same_values_other_dtype = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        first.weight.fill_(1.0)
        same_values_other_dtype.weight.fill_(1.0)
    assert trainable_parameter_sha256(first) != trainable_parameter_sha256(same_values_other_dtype)
    first.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable"):
        trainable_parameter_sha256(first)


def test_optimizer_identity_includes_moments_and_groups():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    initial = optimizer_state_sha256(optimizer)
    parameter.sum().backward()
    optimizer.step()
    after_step = optimizer_state_sha256(optimizer)
    assert initial != after_step
    equivalent_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    equivalent = torch.optim.AdamW([equivalent_parameter], lr=0.001)
    equivalent.load_state_dict(optimizer.state_dict())
    assert optimizer_state_sha256(equivalent) == after_step
    equivalent.state[equivalent_parameter]["exp_avg"][0] += 0.25
    assert optimizer_state_sha256(equivalent) != after_step
    equivalent.load_state_dict(optimizer.state_dict())
    equivalent.param_groups[0]["lr"] = 0.002
    assert optimizer_state_sha256(equivalent) != after_step


def test_identity_rejects_unsupported_optimizer_metadata():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.param_groups[0]["unexpected"] = object()
    with pytest.raises(TypeError, match="Unsupported fingerprint value"):
        optimizer_state_sha256(optimizer)
