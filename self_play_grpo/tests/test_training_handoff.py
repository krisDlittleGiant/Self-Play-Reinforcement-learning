"""CPU-only checkpoint/refresh handoff tests, with fake checkpoint reader."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256
from self_play_grpo.training import ledger as ledger_module
from self_play_grpo.training.coordinator import (
    BatchReceipt, CycleCoordinator, CycleError, Phase, PolicyDescriptor, RoleLayout,
)
from self_play_grpo.training.handoff import (
    acknowledge_refresh_with_identity, publish_cycle_checkpoint,
)
from self_play_grpo.training.ledger import read_commits


def fixture(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoints" / "policy-000001"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"updated-weights")
    (adapter / "adapter_config.json").write_text('{"rank": 8}')
    manifest_path = checkpoint / "distributed" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({"source": "fixture"}))
    config = {"rollout": {"games_per_update": 4}}
    old = PolicyDescriptor(
        version="policy-000000", update_index=0,
        adapter_sha256="a" * 64, config_sha256=canonical_sha256(config),
        model_revision="revision", tokenizer_sha256="b" * 64,
        grammar_sha256="c" * 64, run_kind="production",
    )
    new = replace(old, version="policy-000001", update_index=1,
                  adapter_sha256=directory_sha256(adapter))
    coordinator = CycleCoordinator(
        RoleLayout((0, 1, 2, 3), (4, 5, 6, 7)), old, config, 4,
    )
    coordinator.batch = BatchReceipt(
        manifest_sha256="d" * 64, policy_version=old.version,
        adapter_sha256=old.adapter_sha256, config_sha256=old.config_sha256,
        game_ids=("g0", "g1", "g2", "g3"),
        match_indices_by_rank=((0,), (1,), (2,), (3,)),
        turns_by_rank=(1, 1, 1, 1), owned_tokens_by_rank=(4, 4, 4, 4),
        max_replay_error=0.0,
    )
    coordinator.next_policy = new
    coordinator.updated = {rank: ("1" * 64, "2" * 64, new.adapter_sha256, 1) for rank in range(4)}
    coordinator.phase = Phase.UPDATED

    def reader(path):
        return SimpleNamespace(
            run_kind="production", run_id="run-1", update_index=1,
            policy_version="policy-000001",
            source_rollout_manifest_sha256="d" * 64,
        )

    monkeypatch.setattr(ledger_module, "read_distributed_manifest", reader)
    monkeypatch.setattr(ledger_module, "verify_checkpoint_files", lambda *args: None)
    return coordinator, new, adapter


def test_checkpoint_is_committed_before_refresh(tmp_path, monkeypatch):
    coordinator, updated, _ = fixture(tmp_path, monkeypatch)
    record = publish_cycle_checkpoint(
        coordinator, run_root=tmp_path,
        checkpoint_path="checkpoints/policy-000001", run_id="run-1",
    )
    assert coordinator.phase is Phase.REFRESHING
    assert read_commits(tmp_path) == (record,)
    for rank in range(3):
        acknowledge_refresh_with_identity(
            coordinator, rank, policy=updated,
            loaded_parameter_sha256="1" * 64,
            probe_error=0.0, probe_tolerance=2e-4,
        )
    assert coordinator.phase is Phase.REFRESHING
    acknowledge_refresh_with_identity(
        coordinator, 3, policy=updated,
        loaded_parameter_sha256="1" * 64,
        probe_error=0.0, probe_tolerance=2e-4,
    )
    assert coordinator.phase is Phase.READY
    assert coordinator.policy == updated


def test_refresh_rejects_wrong_loaded_weights(tmp_path, monkeypatch):
    coordinator, updated, _ = fixture(tmp_path, monkeypatch)
    publish_cycle_checkpoint(
        coordinator, run_root=tmp_path,
        checkpoint_path="checkpoints/policy-000001", run_id="run-1",
    )
    with pytest.raises(CycleError, match="different trainable tensor"):
        acknowledge_refresh_with_identity(
            coordinator, 0, policy=updated,
            loaded_parameter_sha256="f" * 64,
            probe_error=0.0, probe_tolerance=2e-4,
        )
    assert coordinator.phase is Phase.FAILED


def test_checkpoint_rejects_wrong_adapter_and_does_not_commit(tmp_path, monkeypatch):
    coordinator, _, adapter = fixture(tmp_path, monkeypatch)
    (adapter / "adapter_model.safetensors").write_bytes(b"stale-weights")
    with pytest.raises(CycleError, match="adapter differs"):
        publish_cycle_checkpoint(
            coordinator, run_root=tmp_path,
            checkpoint_path="checkpoints/policy-000001", run_id="run-1",
        )
    assert coordinator.phase is Phase.FAILED
    assert read_commits(tmp_path) == ()


def test_validation_policy_rejected_before_ledger_write(tmp_path, monkeypatch):
    coordinator, _, _ = fixture(tmp_path, monkeypatch)
    coordinator.policy = replace(coordinator.policy, run_kind="validation")
    with pytest.raises(CycleError, match="Validation-only"):
        publish_cycle_checkpoint(
            coordinator, run_root=tmp_path,
            checkpoint_path="checkpoints/policy-000001", run_id="run-1",
        )
    assert read_commits(tmp_path) == ()
