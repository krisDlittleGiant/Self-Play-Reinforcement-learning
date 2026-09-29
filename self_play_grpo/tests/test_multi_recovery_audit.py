import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256
from self_play_grpo.training import multi_recovery as m


CONFIG = Path(__file__).parents[1] / "configs/quoridor_outcome_64games.yaml"


def test_new_generation_audit_and_missing_refresh_blocks_collection(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    initial = tmp_path / "rollout-000000"
    initial.mkdir()
    (initial / "manifest.json").write_text(json.dumps({"base_seed": 11}))
    source = tmp_path / "rollout-000003/manifest.json"
    source.parent.mkdir()
    source.write_text("{}")
    checkpoint = tmp_path / "trainer-000004/trainer/checkpoints/policy-000004"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "weights").write_text("weights")
    head = NS(run_id="test", update_index=4, policy_version="policy-000004",
              checkpoint_path=str(checkpoint.relative_to(tmp_path)), checkpoint_manifest_sha256="a" * 64,
              source_manifest_sha256=file_sha256(source))
    monkeypatch.setattr(m, "read_commits", lambda _: (head,))
    monkeypatch.setattr(m, "checkpoint_code_identity", lambda update: f"source-{update}")
    monkeypatch.setattr(m, "read_distributed_manifest", lambda _: NS(
        config_sha256=canonical_sha256(config.to_dict()), model_id=config.model.id,
        model_revision=config.model.revision, code_identity="source-4", trainer_world_size=4))
    monkeypatch.setattr("self_play_grpo.rollouts.pilot.read_pilot_manifest", lambda _: NS(replay_tolerance=2e-4))
    plan = m.audit_recovery(tmp_path, config_path=CONFIG, experiment_seed=11)
    assert plan.status == "refresh_required" and plan.next_update == 5
    refresh = tmp_path / "refresh-000004"
    refresh.mkdir()
    evidence = dict(policy_version=head.policy_version, checkpoint_manifest_sha256="a" * 64,
                    source_manifest_sha256=head.source_manifest_sha256, adapter_sha256=directory_sha256(adapter),
                    parameter_sha256="b" * 64)
    (refresh / "refresh_summary.json").write_text(json.dumps(dict(evidence, status="refresh_reports_verified",
                                                                 refresh_ranks=4, max_abs_probe_error=0.0)))
    for rank in range(4):
        (refresh / f"rank-{rank:03d}.json").write_text(json.dumps(dict(evidence, status="refresh_verified",
                                                                     rank=rank, probe_max_abs_error=0.0)))
    assert m.audit_recovery(tmp_path, config_path=CONFIG, experiment_seed=11).status == "ready_for_next_collection"
    with pytest.raises(ValueError, match="seed differs"):
        m.audit_recovery(tmp_path, config_path=CONFIG, experiment_seed=22)
    path = refresh / "rank-003.json"
    row = json.loads(path.read_text())
    row["probe_max_abs_error"] = 0.1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="probability"):
        m.audit_recovery(tmp_path, config_path=CONFIG, experiment_seed=11)
