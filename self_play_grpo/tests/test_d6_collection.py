"""CPU-only contracts for preparing a post-commit collection."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256
from self_play_grpo.training import d6_collection as module
from self_play_grpo.training.coordinator import PolicyDescriptor
from self_play_grpo.training.d6_recovery import RecoveryPlan


CONFIG = Path(__file__).parents[1] / "configs" / "quoridor_outcome_64games.yaml"


@pytest.fixture
def prepared_head(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    descriptor = PolicyDescriptor(
        version="policy-000000", update_index=0,
        adapter_sha256="a" * 64,
        config_sha256=canonical_sha256(config.to_dict()),
        model_revision=config.model.revision,
        tokenizer_sha256="b" * 64, grammar_sha256="c" * 64,
        run_kind="production",
    )
    prior = tmp_path / "rollout-000000"
    prior.mkdir()
    (prior / "policy_descriptor.json").write_text(json.dumps(asdict(descriptor)), encoding="utf-8")
    checkpoint = tmp_path / "trainer-000001" / "trainer" / "checkpoints" / "policy-000001"
    (checkpoint / "adapter").mkdir(parents=True)
    plan = RecoveryPlan(
        status="ready_for_next_collection", run_id="d6-test", committed_update=1,
        policy_version="policy-000001",
        checkpoint=str(checkpoint.relative_to(tmp_path)),
        checkpoint_manifest_sha256="d" * 64,
        source_manifest_sha256="e" * 64,
        next_update=2, next_collection_index=1,
        next_policy_version="policy-000001", next_base_seed=123456,
        seed_derivation="self_play_grpo/production_collection/v1",
        consumed_batches=("e" * 64,),
    )
    monkeypatch.setattr(module, "read_distributed_manifest", lambda _: SimpleNamespace(
        run_kind="production", run_id=plan.run_id,
        policy_version=plan.policy_version, update_index=plan.committed_update,
        tokenizer_sha256=descriptor.tokenizer_sha256,
        grammar_sha256=descriptor.grammar_sha256,
        config_sha256=descriptor.config_sha256,
    ))
    monkeypatch.setattr(module, "directory_sha256", lambda _: "f" * 64)
    return tmp_path, config, plan


def test_descriptor_binds_latest_checkpoint_to_next_collection(prepared_head):
    root, config, plan = prepared_head
    descriptor = module.descriptor_from_commit(root, plan, config=config)
    assert descriptor.version == "policy-000001"
    assert descriptor.update_index == 1
    assert descriptor.adapter_sha256 == "f" * 64
    assert descriptor.run_kind == "production"


def test_incomplete_refresh_cannot_prepare(prepared_head):
    root, config, plan = prepared_head
    with pytest.raises(RuntimeError, match="forbidden"):
        module.descriptor_from_commit(root, replace(plan, status="refresh_required"), config=config)


def test_preparation_uses_new_seed_and_checkpoint_adapter(prepared_head, monkeypatch):
    root, config, plan = prepared_head
    monkeypatch.setattr(module, "audit_recovery", lambda *args, **kwargs: plan)
    calls = []
    monkeypatch.setattr(module, "prepare_frozen_batch", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = module.prepare_next_collection(
        root, config_path=CONFIG, experiment_seed=11,
        expected_run_id=plan.run_id, replay_tolerance=2e-4,
    )
    assert result["status"] == "prepared_not_collected"
    assert result["collection_index"] == 1
    assert result["next_update"] == 2
    assert calls[0][0][0] == root / "rollout-000001"
    assert calls[0][1]["source_adapter"] == root / plan.checkpoint / "adapter"
    assert calls[0][1]["base_seed"] == plan.next_base_seed
    assert calls[0][1]["policy"].version == "policy-000001"
