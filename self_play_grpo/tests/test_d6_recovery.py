"""CPU-only restart-boundary tests; no model or HPU is initialized."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256
from self_play_grpo.training import d6_recovery as module
from self_play_grpo.training.ledger import CommitRecord


CONFIG = Path(__file__).parents[1] / "configs" / "quoridor_outcome_64games.yaml"


@pytest.fixture
def recovery_root(tmp_path, monkeypatch):
    run_id = "d6-test"
    checkpoint_relative = "trainer-000001/trainer/checkpoints/policy-000001"
    checkpoint = tmp_path / checkpoint_relative
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "rollout-000000" / "manifest.json"
    source.parent.mkdir()
    source.write_text('{"batch": 1}', encoding="utf-8")
    record = CommitRecord(
        run_id=run_id, update_index=1, policy_version="policy-000001",
        source_manifest_sha256=file_sha256(source),
        checkpoint_path=checkpoint_relative,
        checkpoint_manifest_sha256="a" * 64,
        previous_record_sha256=None,
    )
    config = load_config(CONFIG)
    monkeypatch.setattr(module, "read_commits", lambda _: (record,))
    monkeypatch.setattr(module, "read_distributed_manifest", lambda _: SimpleNamespace(
        config_sha256=canonical_sha256(config.to_dict()),
        model_id=config.model.id, model_revision=config.model.revision,
        code_identity="source-sha256:test", trainer_world_size=4,
    ))
    monkeypatch.setattr(module, "production_code_identity", lambda: "source-sha256:test")
    return tmp_path, record, directory_sha256(adapter)


def _refresh(root, record, adapter_digest):
    directory = root / "refresh-000001"
    directory.mkdir()
    summary = {
        "status": "refresh_reports_verified",
        "policy_version": record.policy_version,
        "checkpoint_manifest_sha256": record.checkpoint_manifest_sha256,
        "source_manifest_sha256": record.source_manifest_sha256,
        "adapter_sha256": adapter_digest,
        "refresh_ranks": 4,
    }
    (directory / "refresh_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for rank in range(4):
        report = {
            "status": "refresh_verified", "rank": rank,
            "policy_version": record.policy_version,
            "checkpoint_manifest_sha256": record.checkpoint_manifest_sha256,
            "source_manifest_sha256": record.source_manifest_sha256,
            "adapter_sha256": adapter_digest,
        }
        (directory / f"rank-{rank:03d}.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


def test_committed_update_without_refresh_cannot_collect(recovery_root):
    root, record, _ = recovery_root
    plan = module.audit_recovery(root, config_path=CONFIG, experiment_seed=11,
                                 expected_run_id=record.run_id)
    assert plan.status == "refresh_required"
    assert plan.committed_update == 1
    assert plan.next_update == 2
    assert plan.next_collection_index == 1
    assert plan.consumed_batches == (record.source_manifest_sha256,)


def test_complete_refresh_allows_next_collection_with_stable_seed(recovery_root):
    root, record, adapter_digest = recovery_root
    _refresh(root, record, adapter_digest)
    first = module.audit_recovery(root, config_path=CONFIG, experiment_seed=11)
    second = module.audit_recovery(root, config_path=CONFIG, experiment_seed=11)
    assert first == second
    assert first.status == "ready_for_next_collection"
    assert first.next_base_seed != module._next_seed(11, 2, record.run_id)


def test_refresh_tamper_and_missing_rank_fail_closed(recovery_root):
    root, record, adapter_digest = recovery_root
    directory = _refresh(root, record, adapter_digest)
    (directory / "rank-003.json").unlink()
    with pytest.raises(ValueError, match="Missing or symlinked"):
        module.audit_recovery(root, config_path=CONFIG, experiment_seed=11)
    _refresh_report = {
        "status": "refresh_verified", "rank": 3,
        "policy_version": record.policy_version,
        "checkpoint_manifest_sha256": "b" * 64,
        "source_manifest_sha256": record.source_manifest_sha256,
        "adapter_sha256": adapter_digest,
    }
    (directory / "rank-003.json").write_text(json.dumps(_refresh_report), encoding="utf-8")
    with pytest.raises(ValueError, match="rank 3 differs"):
        module.audit_recovery(root, config_path=CONFIG, experiment_seed=11)


def test_wrong_run_or_missing_committed_batch_rejected(recovery_root):
    root, record, _ = recovery_root
    with pytest.raises(ValueError, match="run ID differs"):
        module.audit_recovery(root, config_path=CONFIG, experiment_seed=11,
                              expected_run_id="other")
    (root / "rollout-000000" / "manifest.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="batch is absent or differs"):
        module.audit_recovery(root, config_path=CONFIG, experiment_seed=11)


def test_invalid_seed_rejected(recovery_root):
    root, _, _ = recovery_root
    with pytest.raises(ValueError, match="non-negative"):
        module.audit_recovery(root, config_path=CONFIG, experiment_seed=-1)
