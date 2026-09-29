"""CPU-only versioned recovery identity for the resumed update-2 checkpoint."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256
from self_play_grpo.training import d6_recovery as module


CONFIG = Path(__file__).parents[1] / "configs" / "quoridor_outcome_64games.yaml"


def test_second_commit_uses_new_code_identity_and_next_index(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    source = tmp_path / "rollout-000001" / "manifest.json"
    source.parent.mkdir()
    source.write_text('{"batch":2}', encoding="utf-8")
    adapter = tmp_path / "trainer-000002" / "trainer" / "checkpoints" / "policy-000002" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    head = SimpleNamespace(
        run_id="d6-test", update_index=2, policy_version="policy-000002",
        checkpoint_path=str(adapter.parent.relative_to(tmp_path)),
        checkpoint_manifest_sha256="a" * 64,
        source_manifest_sha256=file_sha256(source),
    )
    first = SimpleNamespace(source_manifest_sha256="b" * 64)
    monkeypatch.setattr(module, "read_commits", lambda _: (first, head))
    monkeypatch.setattr(module, "d6_code_identity", lambda: "source-sha256:d6-test")
    monkeypatch.setattr(module, "read_distributed_manifest", lambda _: SimpleNamespace(
        config_sha256=canonical_sha256(config.to_dict()),
        model_id=config.model.id, model_revision=config.model.revision,
        code_identity="source-sha256:d6-test", trainer_world_size=4,
    ))
    refresh = tmp_path / "refresh-000002"
    refresh.mkdir()
    (refresh / "refresh_summary.json").write_text(json.dumps({
        "status": "refresh_reports_verified", "policy_version": head.policy_version,
        "checkpoint_manifest_sha256": head.checkpoint_manifest_sha256,
        "source_manifest_sha256": head.source_manifest_sha256,
        "adapter_sha256": directory_sha256(adapter), "refresh_ranks": 4,
    }), encoding="utf-8")
    for rank in range(4):
        (refresh / f"rank-{rank:03d}.json").write_text(json.dumps({
            "status": "refresh_verified", "rank": rank,
            "policy_version": head.policy_version,
            "checkpoint_manifest_sha256": head.checkpoint_manifest_sha256,
            "source_manifest_sha256": head.source_manifest_sha256,
            "adapter_sha256": directory_sha256(adapter),
        }), encoding="utf-8")
    plan = module.audit_recovery(tmp_path, config_path=CONFIG, experiment_seed=11)
    assert plan.status == "ready_for_next_collection"
    assert plan.committed_update == 2
    assert plan.next_update == 3
    assert plan.next_collection_index == 2
    assert plan.consumed_batches == ("b" * 64, head.source_manifest_sha256)

    monkeypatch.setattr(module, "d6_code_identity", lambda: "source-sha256:different")
    with pytest.raises(ValueError, match="code or topology"):
        module.audit_recovery(tmp_path, config_path=CONFIG, experiment_seed=11)
