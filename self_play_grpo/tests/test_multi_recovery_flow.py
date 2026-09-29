"""Recovery contracts with fake workers; no accelerator is acquired."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256
from self_play_grpo.training import multi_run as m
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout
from self_play_grpo.training.d6_recovery import RecoveryPlan
from self_play_grpo.training.multi_evidence import publish_json

CONFIG = Path(__file__).parents[1] / "configs/quoridor_outcome_64games.yaml"
LAYOUT = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))


def setup_case(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    config = load_config(CONFIG)
    adapter = root / "trainer-000002/trainer/checkpoints/policy-000002/adapter"
    adapter.mkdir(parents=True)
    (adapter / "weights").write_bytes(b"adapter")
    prior = PolicyDescriptor("policy-000002", 2, directory_sha256(adapter),
                             canonical_sha256(config.to_dict()), config.model.revision,
                             "a" * 64, "b" * 64, "production")
    plan = RecoveryPlan("ready_for_next_collection", "test", 2, prior.version,
                        str(adapter.parent.relative_to(root)), "c" * 64, "d" * 64,
                        3, 2, prior.version, 1234, "v1", ("d" * 64,))
    args = NS(config=CONFIG, run_root=root, run_id="test", seed=11, replay_tolerance=2e-4,
              audit=False, until_update=3, rollout_modules="0,1,2,3", trainer_modules="4,5,6,7",
              trainer_master_port=29642, rollout_timeout_seconds=10, trainer_timeout_seconds=10,
              refresh_timeout_seconds=10, d4_two_summary=tmp_path / "two", d4_four_summary=tmp_path / "four")
    return root, config, prior, plan, args


def test_stop_at_three_then_resume_to_five_without_duplicate_update(tmp_path, monkeypatch):
    root, config, prior, current, args = setup_case(tmp_path)
    events = []
    monkeypatch.setattr(m, "verify_d5_launch_prerequisites", lambda **kw: None)
    monkeypatch.setattr(m, "read_distributed_manifest", lambda _: NS(ranks=[NS(module_id=str(i)) for i in range(4, 8)]))
    monkeypatch.setattr(m, "audit_recovery", lambda *a, **kw: current)
    def collect(*a):
        assert current.status == "ready_for_next_collection"
        events.append(("collect", current.next_update))
        return root / f"rollout-{current.next_collection_index:06d}", prior, object()
    def update(*a):
        nonlocal current
        index = current.next_update
        events.append(("train", index))
        current = replace(current, status="refresh_required", committed_update=index,
                          next_update=index + 1, next_collection_index=index, policy_version=f"policy-{index:06d}")
    def refresh(*a):
        nonlocal current
        events.append(("refresh", current.committed_update))
        current = replace(current, status="ready_for_next_collection")
    monkeypatch.setattr(m, "collect_next", collect)
    monkeypatch.setattr(m, "update_next", update)
    monkeypatch.setattr(m, "finish_refresh", refresh)
    assert m.run(args)["committed_update"] == 3
    args.until_update = 5
    assert m.run(args)["committed_update"] == 5
    assert m.run(args)["committed_update"] == 5
    assert events == [(stage, index) for index in (3, 4, 5) for stage in ("collect", "train", "refresh")]


@pytest.mark.parametrize("interruption", ["collection", "training", "committed"])
def test_retry_after_phase_interruption_does_not_duplicate_committed_step(tmp_path, monkeypatch, interruption):
    root, config, prior, current, args = setup_case(tmp_path)
    monkeypatch.setattr(m, "verify_d5_launch_prerequisites", lambda **kw: None)
    monkeypatch.setattr(m, "read_distributed_manifest", lambda _: NS(ranks=[NS(module_id=str(i)) for i in range(4, 8)]))
    monkeypatch.setattr(m, "audit_recovery", lambda *a, **kw: current)
    failed = False
    optimizer_commits = []
    def maybe_fail(stage):
        nonlocal failed
        if stage == interruption and not failed:
            failed = True
            raise RuntimeError("injected interruption")
    def collect(*a):
        maybe_fail("collection")
        return root / "rollout-000002", prior, object()
    def update(*a):
        nonlocal current
        maybe_fail("training")
        optimizer_commits.append(3)
        current = replace(current, status="refresh_required", committed_update=3, next_update=4)
        maybe_fail("committed")
    def refresh(*a):
        nonlocal current
        current = replace(current, status="ready_for_next_collection")
    monkeypatch.setattr(m, "collect_next", collect)
    monkeypatch.setattr(m, "update_next", update)
    monkeypatch.setattr(m, "finish_refresh", refresh)
    with pytest.raises(RuntimeError, match="injected"):
        m.run(args)
    assert m.run(args)["committed_update"] == 3
    assert optimizer_commits == [3]


def test_complete_batch_is_reused_and_partial_batch_archived(tmp_path, monkeypatch):
    root, config, prior, plan, args = setup_case(tmp_path)
    monkeypatch.setattr(m, "descriptor_from_commit", lambda *a, **kw: prior)
    output = root / "rollout-000002"
    m.prepare_frozen_batch(output, config=config, source_adapter=root / plan.checkpoint / "adapter",
                           policy=prior, base_seed=plan.next_base_seed, replay_tolerance=2e-4)
    (output / "matches/partial.jsonl").write_text("partial evidence")
    calls = []
    monkeypatch.setattr(m, "launch", lambda *a: calls.append("collect"))
    monkeypatch.setattr(m, "aggregate_rank_shards", lambda *a, **kw: None)
    receipt = NS(manifest_sha256="e" * 64)
    monkeypatch.setattr(m, "verify_completed_batch", lambda *a, **kw: receipt)
    assert m.collect_next(args, config, LAYOUT, plan, 0)[2] == receipt
    assert calls == ["collect"]
    assert len(list((root / "recovery_archive").glob("*/matches/partial.jsonl"))) == 1
    manifest = m.read_pilot_manifest(output)
    monkeypatch.setattr(m, "read_pilot_manifest", lambda _: NS(
        base_seed=manifest.base_seed, replay_tolerance=2e-4, policy_version=prior.version,
        adapter_sha256=prior.adapter_sha256, config_sha256=prior.config_sha256, matches=[None] * 64))
    assert m.collect_next(args, config, LAYOUT, plan, 0)[2] == receipt
    assert calls == ["collect"]
    with pytest.raises(ValueError, match="already consumed"):
        m.collect_next(args, config, LAYOUT, replace(plan, consumed_batches=("e" * 64,)), 0)


def test_existing_checkpoint_is_never_retrained_even_if_invalid(tmp_path, monkeypatch):
    root, config, prior, plan, args = setup_case(tmp_path)
    checkpoint = root / "trainer-000003/trainer/checkpoints/policy-000003"
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(m, "launch", lambda *a: pytest.fail("Would repeat optimizer step"))
    monkeypatch.setattr(m, "finalize_checkpoint", lambda **kw: (_ for _ in ()).throw(ValueError("corrupt checkpoint")))
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        m.update_next(args, config, LAYOUT, plan, 0, root / "rollout-000002", prior, NS(manifest_sha256="e" * 64))
    assert checkpoint.exists()


def test_worker_inherits_lock_after_coordinator_closes_it(tmp_path):
    with m.run_lock(tmp_path) as fd:
        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                                 stdin=subprocess.PIPE, pass_fds=(fd,))
    try:
        with pytest.raises(RuntimeError, match="live coordinator or worker"):
            with m.run_lock(tmp_path):
                pass
    finally:
        child.communicate(timeout=5)
    with m.run_lock(tmp_path):
        pass


def test_publish_json_is_idempotent_but_rejects_conflicts(tmp_path):
    target = tmp_path / "evidence.json"
    publish_json(target, {"step": 3})
    publish_json(target, {"step": 3})
    with pytest.raises(ValueError, match="differs"):
        publish_json(target, {"step": 4})
    assert json.loads(target.read_text()) == {"step": 3}


def test_stale_ledger_staging_is_archived_without_touching_commit(tmp_path):
    commits = tmp_path / "commits"
    commits.mkdir()
    (commits / "update-000001.json").write_text("committed")
    (commits / ".update-000002.json.123.pending").write_text("unpublished")
    m.clean_pending_ledger(tmp_path)
    assert (commits / "update-000001.json").read_text() == "committed"
    assert len(list((tmp_path / "recovery_archive").iterdir())) == 1
