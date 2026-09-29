"""A published complete batch may be reused; a partial one is not overwritten."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.training import d6_cycle as module
from self_play_grpo.training.coordinator import PolicyDescriptor
from self_play_grpo.training.d6_recovery import RecoveryPlan


def test_complete_unconsumed_batch_skips_recollection(tmp_path, monkeypatch):
    root = tmp_path / "run"
    rollout = root / "rollout-000001"
    rollout.mkdir(parents=True)
    (rollout / "policy_adapter").mkdir()
    prior = PolicyDescriptor(
        version="policy-000001", update_index=1, adapter_sha256="a" * 64,
        config_sha256="b" * 64, model_revision="rev",
        tokenizer_sha256="c" * 64, grammar_sha256="d" * 64,
        run_kind="production",
    )
    (rollout / "policy_descriptor.json").write_text(json.dumps(prior.__dict__), encoding="utf-8")
    plan = RecoveryPlan(
        status="ready_for_next_collection", run_id="d6-test", committed_update=1,
        policy_version=prior.version, checkpoint="checkpoint",
        checkpoint_manifest_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        next_update=2, next_collection_index=1,
        next_policy_version=prior.version, next_base_seed=1234,
        seed_derivation="self_play_grpo/production_collection/v1",
        consumed_batches=("f" * 64,),
    )
    args = SimpleNamespace(
        config=Path("self_play_grpo/configs/quoridor_outcome_64games.yaml"),
        run_root=root, run_id=plan.run_id,
        rollout_modules="0,1,2,3", trainer_modules="4,5,6,7",
        d4_two_summary=tmp_path / "two", d4_four_summary=tmp_path / "four",
        seed=11, replay_tolerance=2e-4, trainer_master_port=29531,
        rollout_timeout_seconds=1, trainer_timeout_seconds=1,
        refresh_timeout_seconds=1,
    )
    monkeypatch.setattr(module, "verify_d5_launch_prerequisites", lambda **kwargs: None)
    monkeypatch.setattr(module, "audit_recovery", lambda *args, **kwargs: plan)
    monkeypatch.setattr(module, "descriptor_from_commit", lambda *args, **kwargs: prior)
    monkeypatch.setattr(module, "directory_sha256", lambda _: prior.adapter_sha256)
    monkeypatch.setattr(module, "read_pilot_manifest", lambda path: (
        SimpleNamespace(base_seed=11) if Path(path).name == "rollout-000000"
        else SimpleNamespace(base_seed=1234, replay_tolerance=2e-4, matches=tuple(range(64)))
    ))
    monkeypatch.setattr(module, "verify_completed_batch", lambda *args, **kwargs: SimpleNamespace(
        manifest_sha256="1" * 64,
    ))
    monkeypatch.setattr(module, "prepare_next_collection", lambda *args, **kwargs: pytest.fail("reprepared"))
    monkeypatch.setattr(module, "aggregate_rank_shards", lambda *args, **kwargs: pytest.fail("reaggregated"))
    events = []

    def phase(workers, layout, output, *, role, **kwargs):
        if role == "rollout":
            pytest.fail("a complete rollout was recollected")
        events.append("trainer")
        raise RuntimeError("stop before HPU worker launch")

    monkeypatch.setattr(module, "supervise_role_phase", phase)
    monkeypatch.setattr(module, "read_commits", lambda *args: (SimpleNamespace(),))
    with pytest.raises(RuntimeError, match="stop before"):
        module.run_second_cycle(args)
    assert events == ["trainer"]
