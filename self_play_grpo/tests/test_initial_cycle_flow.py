"""Model-free first-cycle sequencing with fake workers and artifacts."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from self_play_grpo.training import initial_cycle as module
from self_play_grpo.training.coordinator import Phase, PolicyDescriptor


def test_one_cycle_commits_checkpoint_before_rollout_refresh(tmp_path, monkeypatch):
    events = []
    prior = PolicyDescriptor(
        version="policy-000000", update_index=0, adapter_sha256="a" * 64,
        config_sha256="b" * 64, model_revision="rev",
        tokenizer_sha256="c" * 64, grammar_sha256="d" * 64,
        run_kind="production",
    )
    updated = replace(prior, version="policy-000001", update_index=1, adapter_sha256="e" * 64)
    receipt = SimpleNamespace(manifest_sha256="f" * 64)
    record = SimpleNamespace(
        checkpoint_manifest_sha256="1" * 64,
        checkpoint_path="trainer-000001/trainer/checkpoints/policy-000001",
    )
    coordinator = SimpleNamespace(
        phase=Phase.UPDATED, policy=prior,
        snapshot=lambda: {"phase": "ready", "policy_version": updated.version},
    )
    monkeypatch.setattr(module, "verify_d5_launch_prerequisites", lambda **kwargs: events.append("preflight"))
    monkeypatch.setattr(module, "prepare_initial_policy", lambda **kwargs: events.append("bootstrap"))
    monkeypatch.setattr(module, "_initial_policy", lambda *args: prior)
    monkeypatch.setattr(module, "prepare_frozen_batch", lambda *args, **kwargs: events.append("prepare_batch"))
    monkeypatch.setattr(module, "aggregate_rank_shards", lambda *args, **kwargs: events.append("aggregate_batch"))
    monkeypatch.setattr(module, "verify_completed_batch", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(module, "_trainer_evidence", lambda *args, **kwargs: (
        updated, {"metrics": {"loss": 0.1}}, ({},) * 4,
    ))
    monkeypatch.setattr(module, "_reconstruct_coordinator", lambda **kwargs: coordinator)

    def phase(workers, layout, output, *, role, **kwargs):
        if "logs-rollout" in str(output):
            events.append("rollout")
        elif "logs-trainer" in str(output):
            events.append("trainer")
        else:
            events.append("refresh")
            assert coordinator.phase is Phase.REFRESHING
            refresh = tmp_path / "run" / "refresh-000001"
            refresh.mkdir()
            for rank in range(4):
                (refresh / f"rank-{rank:03d}.json").write_text(json.dumps({
                    "parameter_sha256": "2" * 64,
                    "probe_max_abs_error": 0.0,
                }))

    monkeypatch.setattr(module, "supervise_role_phase", phase)

    def commit(*args, **kwargs):
        events.append("ledger_commit")
        coordinator.phase = Phase.REFRESHING
        return record

    monkeypatch.setattr(module, "publish_cycle_checkpoint", commit)
    monkeypatch.setattr(module, "aggregate_refresh_evidence", lambda **kwargs: (
        updated, {"max_abs_probe_error": 0.0},
    ))

    def acknowledge(coordinator, rank, **kwargs):
        events.append(f"ack_{rank}")
        if rank == 3:
            coordinator.phase = Phase.READY
            coordinator.policy = updated

    monkeypatch.setattr(module, "acknowledge_refresh_with_identity", acknowledge)
    monkeypatch.setattr(module, "publish_refresh_evidence", lambda *args: events.append("publish_refresh"))
    monkeypatch.setattr(module, "read_commits", lambda root: (record,))
    output = tmp_path / "run"
    args = SimpleNamespace(
        config=Path("self_play_grpo/configs/quoridor_outcome_64games.yaml"),
        rollout_modules="0,1,2,3", trainer_modules="4,5,6,7",
        d4_two_summary=tmp_path / "two", d4_four_summary=tmp_path / "four",
        seed=11, replay_tolerance=2e-4,
        rollout_timeout_seconds=1, trainer_timeout_seconds=1,
        refresh_timeout_seconds=1, trainer_master_port=29531,
        run_id="fake-cycle", pilot_root=tmp_path / "pilot", output=output,
    )
    result = module.run_initial_cycle(args)
    assert result["status"] == "one_update_complete"
    assert events == [
        "preflight", "bootstrap", "prepare_batch", "rollout", "aggregate_batch",
        "trainer", "ledger_commit", "refresh", "ack_0", "ack_1",
        "ack_2", "ack_3", "publish_refresh",
    ]
    assert json.loads((output / "cycle_summary.json").read_text())["status"] == "one_update_complete"
