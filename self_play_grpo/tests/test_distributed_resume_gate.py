"""Model-free and CPU-only contracts for the D4 continuation gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from self_play_grpo.cli import build_parser
from self_play_grpo.training.distributed_resume_gate import (
    _assert_exact,
    _collective_phase,
    _expected_identity,
    _freeze,
    _module_ids,
    _prepare_source,
)


def test_cli_requires_explicit_validation_source_and_paths() -> None:
    args = build_parser().parse_args([
        "validate-distributed-checkpoint-resume",
        "--config", "config.yaml",
        "--source-rollout", "rollout",
        "--source-checkpoint", "checkpoint",
        "--output", "new-output",
        "--expected-world-size", "2",
        "--allow-validation-source",
    ])
    assert args.source_rollout == Path("rollout")
    assert args.source_checkpoint == Path("checkpoint")
    assert args.expected_world_size == 2
    assert args.allow_validation_source is True
    assert args.joint_step == 0
    assert callable(args.handler)


def test_validation_only_bridge_fails_closed_without_flag(tmp_path: Path) -> None:
    args = SimpleNamespace(
        allow_validation_source=False, output=tmp_path / "output",
        joint_step=0, match_offset=0, seed=741,
    )
    with pytest.raises(ValueError, match="allow-validation-source"):
        _prepare_source(args, None, None)


def test_rank_module_binding_requires_unique_complete_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3,5,7,1")
    assert _module_ids(4) == ("3", "5", "7", "1")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3,3")
    with pytest.raises(ValueError, match="unique complete"):
        _module_ids(2)
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")
    with pytest.raises(ValueError, match="unique complete"):
        _module_ids(2)


def test_snapshot_comparator_checks_named_tensor_and_scalar_values() -> None:
    left = {
        "parameters": {"lora_A": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([3.0])}}},
        "metrics": {"loss": 0.5},
    }
    right = _freeze(left, torch)
    _assert_exact(left, right, torch)
    right["optimizer"]["state"][0]["exp_avg"][0] = 4.0
    with pytest.raises(AssertionError, match="optimizer.state.0.exp_avg"):
        _assert_exact(left, right, torch)
    right = _freeze(left, torch)
    right["metrics"]["loss"] = 0.6
    with pytest.raises(AssertionError, match="metrics.loss"):
        _assert_exact(left, right, torch)


def test_snapshot_freeze_is_independent_copy() -> None:
    original = {"tensor": torch.tensor([1.0]), "nested": [2]}
    frozen = _freeze(original, torch)
    original["tensor"][0] = 9.0
    original["nested"][0] = 7
    assert frozen["tensor"].item() == 1.0
    assert frozen["nested"] == [2]


def test_expected_identity_contains_rank_bindings_and_complete_shards() -> None:
    rank = SimpleNamespace(
        rank=0, local_rank=0, module_id="2", device="hpu:0",
        match_indices=(0, 1), match_ids=("g0", "g1"),
        match_sha256=("a" * 64, "b" * 64),
    )
    template = SimpleNamespace(
        run_id="gate", policy_version="policy-000002", update_index=2,
        config_sha256="1" * 64, model_id="toy", model_revision="rev",
        adapter_schema_sha256="2" * 64, optimizer_schema_sha256="3" * 64,
        tokenizer_sha256="4" * 64, grammar_sha256="5" * 64,
        code_identity="code", runtime_identity=(("torch", "test"),),
        attention_backend="sdpa", dtype="bfloat16", backend="nccl",
        trainer_world_size=1, source_rollout_manifest_sha256="6" * 64,
        sharding_rule="equal_contiguous_complete_matches_v1", ranks=(rank,),
    )
    expected = _expected_identity(template)
    assert expected["rank_bindings"] == [
        {"rank": 0, "local_rank": 0, "module_id": "2", "device": "hpu:0"}
    ]
    assert expected["match_shards"][0]["match_indices"] == [0, 1]
    assert expected["match_shards"][0]["match_ids"] == ["g0", "g1"]


def test_collective_phase_reports_local_failure_without_hpu() -> None:
    class FakeDist:
        ReduceOp = SimpleNamespace(MIN="min")

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            assert op == "min"
            assert tensor.device.type == "cpu"

    runtime = SimpleNamespace(rank=0, device="cpu")
    assert _collective_phase(torch, FakeDist(), runtime, "ok", lambda: 7) == 7

    def fail() -> None:
        raise ValueError("fixture error")

    with pytest.raises(RuntimeError, match="fixture error"):
        _collective_phase(torch, FakeDist(), runtime, "bad", fail)


def test_preflight_accepts_json_equivalent_config_but_rejects_real_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from self_play_grpo.training import distributed_resume_gate as gate

    monkeypatch.setattr(gate, "validate_manifest_request", lambda *a, **k: None)
    source = tmp_path / "checkpoints" / "policy-000001"
    source.mkdir(parents=True)
    state = {
        "checkpoint_format": 2, "validation_only": True,
        "update_index": 1, "policy_version": "policy-000001",
        "config": {"evaluation": {"training_seeds": [11, 22, 33]}},
    }
    (source / "trainer_state.json").write_text(json.dumps(state))
    (tmp_path / "summary.json").write_text(json.dumps({
        "behavior_adapter_sha256": "a" * 64,
        "checkpoint": str(source),
    }))
    config = SimpleNamespace(
        to_dict=lambda: {"evaluation": {"training_seeds": (11, 22, 33)}}
    )
    manifest = SimpleNamespace(
        adapter_sha256="a" * 64, base_seed=11, replay_tolerance=2e-4
    )
    args = SimpleNamespace(
        allow_validation_source=True, output=tmp_path / "new-output",
        joint_step=0, match_offset=0, seed=741, source_checkpoint=source,
    )
    _prepare_source(args, config, manifest)
    state["config"]["evaluation"]["training_seeds"] = [99]
    (source / "trainer_state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="configuration differs"):
        _prepare_source(args, config, manifest)
