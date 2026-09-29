"""CPU-only contracts for the production cold-process trainer restore wrapper."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.training import d6_trainer_resume as module
from self_play_grpo.training.d6_recovery import RecoveryPlan


CONFIG = Path(__file__).parents[1] / "configs" / "quoridor_outcome_64games.yaml"
RUNTIME = {
    "python": "3.12", "torch": "2.7", "transformers": "5.12",
    "peft": "0.20", "habana": "1.22",
}


@pytest.fixture
def restore_case(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    checkpoint = tmp_path / "trainer-000001" / "trainer" / "checkpoints" / "policy-000001"
    checkpoint.mkdir(parents=True)
    refresh = tmp_path / "refresh-000001"
    refresh.mkdir()
    (refresh / "refresh_summary.json").write_text(json.dumps({
        "status": "refresh_reports_verified",
        "policy_version": "policy-000001",
        "checkpoint_manifest_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "parameter_sha256": "c" * 64,
        "optimizer_sha256": "d" * 64,
    }), encoding="utf-8")
    plan = RecoveryPlan(
        status="ready_for_next_collection", run_id="d6-test", committed_update=1,
        policy_version="policy-000001", checkpoint=str(checkpoint.relative_to(tmp_path)),
        checkpoint_manifest_sha256="a" * 64, source_manifest_sha256="b" * 64,
        next_update=2, next_collection_index=1, next_policy_version="policy-000001",
        next_base_seed=1234, seed_derivation="self_play_grpo/production_collection/v1",
        consumed_batches=("b" * 64,),
    )
    ledger_head = SimpleNamespace(
        run_id=plan.run_id, update_index=plan.committed_update,
        policy_version=plan.policy_version, checkpoint_path=plan.checkpoint,
        checkpoint_manifest_sha256=plan.checkpoint_manifest_sha256,
        source_manifest_sha256=plan.source_manifest_sha256,
    )
    monkeypatch.setattr(module, "read_commits", lambda _: (ledger_head,))
    calls = []

    class Trainer:
        update_index = 0
        policy_version = "policy-000000"
        evaluator = None
        policy = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa")))
        optimizer = object()

        def __init__(self):
            self.config = config

        def load_checkpoint(self, path, **kwargs):
            calls.append((path, kwargs))
            self.update_index = 1
            self.policy_version = "policy-000001"

    monkeypatch.setattr(module, "read_distributed_manifest", lambda _: object())
    monkeypatch.setattr(module, "verify_checkpoint_files", lambda *args: None)
    monkeypatch.setattr(module, "_expected_identity", lambda _: {"match_shards": []})
    monkeypatch.setattr(module, "_adapter_schema", lambda _: "e" * 64)
    monkeypatch.setattr(module, "_optimizer_schema", lambda _: "f" * 64)
    monkeypatch.setattr(module, "production_code_identity", lambda: "source-sha256:test")
    monkeypatch.setattr(module, "validate_resume_identity", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(module, "trainable_parameter_sha256", lambda _: "c" * 64)
    monkeypatch.setattr(module, "optimizer_state_sha256", lambda _: "d" * 64)
    return tmp_path, plan, Trainer(), calls


def test_restores_only_after_identity_gate_and_without_step(restore_case):
    root, plan, trainer, calls = restore_case
    result = module.restore_first_production_update(
        trainer, root=root, plan=plan, rank=2,
        trainer_module_ids=(4, 5, 6, 7), runtime_identity=RUNTIME,
    )
    assert result["status"] == "restored_no_optimizer_step"
    assert result["policy_version"] == "policy-000001"
    assert calls[0]["expected"]["rank_bindings"][2]["module_id"] == "6"
    assert calls[0]["expected"]["optimizer_schema_sha256"] == "f" * 64
    assert calls[1][1]["distributed_rank"] == 2
    assert calls[1][1]["allow_validation"] is False


def test_missing_refresh_or_wrong_rank_rejects_before_state_mutation(restore_case):
    root, plan, trainer, calls = restore_case
    with pytest.raises(ValueError, match="rank must"):
        module.restore_first_production_update(
            trainer, root=root, plan=plan, rank=4,
            trainer_module_ids=(4, 5, 6, 7), runtime_identity=RUNTIME,
        )
    assert calls == []
    (root / "refresh-000001" / "refresh_summary.json").unlink()
    with pytest.raises(ValueError, match="Missing or symlinked"):
        module.restore_first_production_update(
            trainer, root=root, plan=plan, rank=0,
            trainer_module_ids=(4, 5, 6, 7), runtime_identity=RUNTIME,
        )
    assert trainer.update_index == 0
    assert not any(isinstance(row, tuple) for row in calls)


def test_postrestore_fingerprint_mismatch_fails_closed(restore_case, monkeypatch):
    root, plan, trainer, _ = restore_case
    monkeypatch.setattr(module, "optimizer_state_sha256", lambda _: "0" * 64)
    with pytest.raises(RuntimeError, match="optimizer differ"):
        module.restore_first_production_update(
            trainer, root=root, plan=plan, rank=0,
            trainer_module_ids=(4, 5, 6, 7), runtime_identity=RUNTIME,
        )
