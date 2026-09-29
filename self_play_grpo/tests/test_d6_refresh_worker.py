"""CPU-only aggregation of the four update-2 policy refresh reports."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.training import d6_refresh_worker as module
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout


CONFIG = Path(__file__).parents[1] / "configs" / "quoridor_outcome_64games.yaml"


@pytest.fixture
def refresh_case(tmp_path, monkeypatch):
    source = tmp_path / "rollout-000001"
    trained = tmp_path / "trainer-000002"
    refreshed = tmp_path / "refresh-000002"
    (trained / "ranks").mkdir(parents=True)
    refreshed.mkdir()
    prior = PolicyDescriptor(
        version="policy-000001", update_index=1, adapter_sha256="a" * 64,
        config_sha256="b" * 64, model_revision="rev",
        tokenizer_sha256="c" * 64, grammar_sha256="d" * 64,
        run_kind="production",
    )
    summary = {
        "checkpoint_manifest_sha256": "e" * 64,
        "checkpoint_adapter_sha256": "f" * 64,
        "parameter_sha256": "0" * 64,
    }
    monkeypatch.setattr(module, "_source_and_checkpoint", lambda *args, **kwargs: (
        prior, summary, trained / "trainer" / "checkpoints" / "policy-000002", "1" * 64,
    ))
    monkeypatch.setattr(module, "read_pilot_manifest", lambda *args: SimpleNamespace(replay_tolerance=2e-4))
    for rank in range(4):
        (trained / "ranks" / f"rank-{rank:03d}.json").write_text(json.dumps({
            "status": "updated_checkpoint_committed_refresh_pending",
            "rank": rank, "module_id": str(4 + rank),
            "source_manifest_sha256": "1" * 64,
            "checkpoint_manifest_sha256": "e" * 64,
            "checkpoint_adapter_sha256": "f" * 64,
            "source_policy_version": prior.version,
            "update": {
                "source_policy_version": prior.version,
                "next_policy_version": "policy-000002",
                "parameter_sha256": "0" * 64,
                "consumed_manifest_sha256": "1" * 64,
                "optimizer_sha256": "2" * 64,
                "optimizer_steps": 2, "gradient_sync_phases": 1,
            },
        }), encoding="utf-8")
        (refreshed / f"rank-{rank:03d}.json").write_text(json.dumps({
            "status": "refresh_verified", "rank": rank, "module_id": rank,
            "policy_version": "policy-000002",
            "source_manifest_sha256": "1" * 64,
            "checkpoint_manifest_sha256": "e" * 64,
            "adapter_sha256": "f" * 64,
            "parameter_sha256": "0" * 64,
            "probe_max_abs_error": 0.0,
        }), encoding="utf-8")
    return source, trained, refreshed


def test_four_refresh_reports_advance_policy_to_two(refresh_case):
    source, trained, refreshed = refresh_case
    policy, evidence = module.aggregate_refresh(
        rollout_root=source, trainer_output=trained,
        refresh_output=refreshed, config=load_config(CONFIG),
        layout=RoleLayout((0, 1, 2, 3), (4, 5, 6, 7)), run_id="d6-test",
    )
    assert policy.version == "policy-000002"
    assert policy.update_index == 2
    assert evidence["optimizer_sha256"] == "2" * 64
    assert evidence["refresh_ranks"] == 4


def test_stale_refresh_version_fails(refresh_case):
    source, trained, refreshed = refresh_case
    target = refreshed / "rank-003.json"
    row = json.loads(target.read_text(encoding="utf-8"))
    row["policy_version"] = "policy-000001"
    target.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="rank 3 identity differs"):
        module.aggregate_refresh(
            rollout_root=source, trainer_output=trained,
            refresh_output=refreshed, config=load_config(CONFIG),
            layout=RoleLayout((0, 1, 2, 3), (4, 5, 6, 7)), run_id="d6-test",
        )
