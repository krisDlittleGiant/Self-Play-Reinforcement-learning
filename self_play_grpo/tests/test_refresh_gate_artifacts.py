"""CPU-only integration fixture for the four-rank refresh evidence reader."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256
from self_play_grpo.training import refresh_gate as module
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout


SOURCE = "a" * 64
CHECKPOINT = "b" * 64
ADAPTER = "c" * 64
PARAMETERS = "d" * 64
OPTIMIZER = "e" * 64
MATCH = "f" * 64
LAYOUT = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))


def fixture(tmp_path: Path, monkeypatch):
    config = load_config(Path("self_play_grpo/configs/quoridor_outcome_64games.yaml"))
    source = tmp_path / "rollout"
    trained = tmp_path / "trainer-output"
    refreshed = tmp_path / "refresh-output"
    checkpoint = trained / "trainer" / "checkpoints" / "policy-000001"
    source.mkdir()
    (trained / "ranks").mkdir(parents=True)
    refreshed.mkdir()
    checkpoint.mkdir(parents=True)
    prior = PolicyDescriptor(
        version="policy-000000", update_index=0, adapter_sha256="1" * 64,
        config_sha256=canonical_sha256(config.to_dict()),
        model_revision=config.model.revision, tokenizer_sha256="2" * 64,
        grammar_sha256="3" * 64, run_kind="production",
    )
    (source / "policy_descriptor.json").write_text(json.dumps(asdict(prior)))
    summary = {
        "status": "updated_checkpoint_committed_refresh_pending",
        "checkpoint": str(checkpoint),
        "source_manifest_sha256": SOURCE,
        "checkpoint_manifest_sha256": CHECKPOINT,
        "checkpoint_adapter_sha256": ADAPTER,
        "parameter_sha256": PARAMETERS,
        "policy_probe": {
            "game_index": 0, "joint_step": 0, "match_sha256": MATCH,
            "completion_token_ids": [10], "log_probs": [-0.3],
        },
    }
    (trained / "trainer_summary.json").write_text(json.dumps(summary))
    for rank in range(4):
        trainer_report = {
            "status": "updated_checkpoint_committed_refresh_pending",
            "rank": rank, "module_id": str(LAYOUT.trainer_modules[rank]),
            "source_manifest_sha256": SOURCE,
            "checkpoint_manifest_sha256": CHECKPOINT,
            "checkpoint_adapter_sha256": ADAPTER,
            "update": {
                "parameter_sha256": PARAMETERS, "optimizer_sha256": OPTIMIZER,
                "next_policy_version": "policy-000001",
                "consumed_manifest_sha256": SOURCE,
                "optimizer_steps": 1, "gradient_sync_phases": 1,
            },
        }
        (trained / "ranks" / f"rank-{rank:03d}.json").write_text(json.dumps(trainer_report))
        refresh_report = {
            "status": "refresh_verified", "rank": rank, "module_id": LAYOUT.rollout_modules[rank],
            "policy_version": "policy-000001", "source_manifest_sha256": SOURCE,
            "checkpoint_manifest_sha256": CHECKPOINT,
            "adapter_sha256": ADAPTER, "parameter_sha256": PARAMETERS,
            "probe_max_abs_error": 0.0,
        }
        (refreshed / f"rank-{rank:03d}.json").write_text(json.dumps(refresh_report))
    rollout_manifest = SimpleNamespace(
        matches=(None,) * 64, config_sha256=prior.config_sha256,
        policy_version=prior.version, adapter_sha256=prior.adapter_sha256,
        replay_tolerance=2e-4,
    )
    checkpoint_manifest = SimpleNamespace(
        run_kind="production", policy_version="policy-000001", update_index=1,
        source_rollout_manifest_sha256=SOURCE, config_sha256=prior.config_sha256,
        model_id=config.model.id, model_revision=prior.model_revision,
        tokenizer_sha256=prior.tokenizer_sha256, grammar_sha256=prior.grammar_sha256,
        code_identity="source-sha256:fixture", trainer_world_size=4,
    )
    monkeypatch.setattr(module, "read_pilot_manifest", lambda path: rollout_manifest)
    monkeypatch.setattr(module, "read_distributed_manifest", lambda path: checkpoint_manifest)
    monkeypatch.setattr(module, "verify_checkpoint_files", lambda *args: None)
    monkeypatch.setattr(module, "directory_sha256", lambda path: ADAPTER)
    monkeypatch.setattr(module, "production_code_identity", lambda: "source-sha256:fixture")

    def hash_file(path):
        if Path(path) == source / "manifest.json":
            return SOURCE
        if Path(path) == checkpoint / "distributed" / "manifest.json":
            return CHECKPOINT
        raise AssertionError(f"Unexpected hash path: {path}")

    monkeypatch.setattr(module, "file_sha256", hash_file)
    return source, trained, refreshed, config


def test_complete_refresh_artifacts_validate(tmp_path, monkeypatch):
    source, trained, refreshed, config = fixture(tmp_path, monkeypatch)
    policy, evidence = module.aggregate_refresh_evidence(
        rollout_root=source, trainer_output=trained, refresh_output=refreshed,
        config=config, layout=LAYOUT,
    )
    assert policy.version == "policy-000001"
    assert policy.adapter_sha256 == ADAPTER
    assert evidence["refresh_ranks"] == 4
    assert evidence["max_abs_probe_error"] == 0.0


def test_mismatched_trainer_report_rejected(tmp_path, monkeypatch):
    source, trained, refreshed, config = fixture(tmp_path, monkeypatch)
    path = trained / "ranks" / "rank-002.json"
    row = json.loads(path.read_text())
    row["update"]["parameter_sha256"] = "0" * 64
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="trainer rank 2 report differs"):
        module.aggregate_refresh_evidence(
            rollout_root=source, trainer_output=trained, refresh_output=refreshed,
            config=config, layout=LAYOUT,
        )
