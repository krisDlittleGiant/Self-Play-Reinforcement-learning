"""Accelerator-free contracts for the D5 eight-worker lifecycle."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.rollouts.pilot import (
    PilotManifest,
    PilotMatchEntry,
    canonical_sha256,
    pilot_game_id,
    pilot_match_relative_path,
)
from self_play_grpo.training import coordinator as module
from self_play_grpo.training.coordinator import (
    BatchReceipt,
    CycleCoordinator,
    CycleError,
    Phase,
    PolicyDescriptor,
    RoleLayout,
)


def fixture_cycle(games: int = 8) -> tuple[CycleCoordinator, BatchReceipt]:
    config = {"rollout": {"games_per_update": games}}
    policy = PolicyDescriptor(
        version="policy-000000",
        update_index=0,
        adapter_sha256="a" * 64,
        config_sha256=canonical_sha256(config),
        model_revision="pinned-model-revision",
        tokenizer_sha256="b" * 64,
        grammar_sha256="c" * 64,
        run_kind="production",
    )
    coordinator = CycleCoordinator(
        layout=RoleLayout((0, 1, 2, 3), (4, 5, 6, 7)),
        policy=policy,
        config=config,
        expected_games=games,
    )
    shards = tuple(tuple(range(rank * (games // 4), (rank + 1) * (games // 4))) for rank in range(4))
    receipt = BatchReceipt(
        manifest_sha256="d" * 64,
        policy_version=policy.version,
        adapter_sha256=policy.adapter_sha256,
        config_sha256=policy.config_sha256,
        game_ids=tuple(f"game-{index}" for index in range(games)),
        match_indices_by_rank=shards,
        turns_by_rank=(8, 9, 10, 11),
        owned_tokens_by_rank=(24, 27, 30, 33),
        max_replay_error=0.0,
    )
    return coordinator, receipt


def ready_and_commit(monkeypatch: pytest.MonkeyPatch) -> tuple[CycleCoordinator, BatchReceipt]:
    coordinator, receipt = fixture_cycle()
    for role in ("rollout", "trainer"):
        for rank in range(4):
            coordinator.acknowledge_ready(role, rank, coordinator.policy)
    coordinator.begin_collection()
    monkeypatch.setattr(module, "verify_completed_batch", lambda *args, **kwargs: receipt)
    assert coordinator.commit_batch(Path("unused")) == receipt
    return coordinator, receipt


def verify_all(coordinator: CycleCoordinator, receipt: BatchReceipt) -> None:
    for rank in range(4):
        coordinator.acknowledge_verified(
            rank,
            manifest_sha256=receipt.manifest_sha256,
            match_indices=receipt.match_indices_by_rank[rank],
            turns=receipt.turns_by_rank[rank],
            owned_tokens=receipt.owned_tokens_by_rank[rank],
            max_replay_error=0.0,
        )


def next_policy(coordinator: CycleCoordinator) -> PolicyDescriptor:
    return replace(
        coordinator.policy,
        version="policy-000001",
        update_index=1,
        adapter_sha256="e" * 64,
    )


def test_full_cycle_requires_four_refresh_acknowledgements(monkeypatch):
    coordinator, receipt = ready_and_commit(monkeypatch)
    verify_all(coordinator, receipt)
    assert coordinator.phase is Phase.VERIFIED
    updated = next_policy(coordinator)
    for rank in range(4):
        coordinator.acknowledge_update(
            rank,
            manifest_sha256=receipt.manifest_sha256,
            next_policy=updated,
            parameter_sha256="1" * 64,
            optimizer_sha256="2" * 64,
            optimizer_steps=1,
            gradient_sync_phases=1,
            changed_tensors=252,
        )
    assert coordinator.phase is Phase.UPDATED
    coordinator.commit_checkpoint(
        manifest_sha256=receipt.manifest_sha256,
        checkpoint_sha256="3" * 64,
        policy=updated,
    )
    for rank in range(3):
        coordinator.acknowledge_refreshed(
            rank, policy=updated, probe_error=0.0, probe_tolerance=2e-4,
        )
    assert coordinator.phase is Phase.REFRESHING
    assert coordinator.snapshot()["refreshed_rollouts"] == 3
    coordinator.acknowledge_refreshed(
        3, policy=updated, probe_error=0.0, probe_tolerance=2e-4,
    )
    assert coordinator.phase is Phase.READY
    assert coordinator.policy == updated
    assert coordinator.snapshot()["batch_sha256"] is None


def test_collection_cannot_start_with_missing_worker():
    coordinator, _ = fixture_cycle()
    for role in ("rollout", "trainer"):
        for rank in range(4):
            if (role, rank) != ("trainer", 3):
                coordinator.acknowledge_ready(role, rank, coordinator.policy)
    with pytest.raises(CycleError, match="all eight"):
        coordinator.begin_collection()
    assert coordinator.phase is Phase.FAILED


def test_stale_ready_acknowledgement_fails_closed():
    coordinator, _ = fixture_cycle()
    stale = replace(coordinator.policy, version="old")
    with pytest.raises(CycleError, match="stale"):
        coordinator.acknowledge_ready("rollout", 0, stale)
    assert coordinator.phase is Phase.FAILED


def test_duplicate_games_or_overlapping_modules_rejected():
    _, receipt = fixture_cycle()
    with pytest.raises(ValueError, match="duplicate game IDs"):
        replace(receipt, game_ids=(receipt.game_ids[0],) + receipt.game_ids[1:-1] + (receipt.game_ids[0],))
    with pytest.raises(ValueError, match="eight distinct"):
        RoleLayout((0, 1, 2, 3), (3, 4, 5, 6))


def test_bad_trainer_coverage_rejected_before_update(monkeypatch):
    coordinator, receipt = ready_and_commit(monkeypatch)
    with pytest.raises(CycleError, match="differs"):
        coordinator.acknowledge_verified(
            0,
            manifest_sha256=receipt.manifest_sha256,
            match_indices=(99,),
            turns=receipt.turns_by_rank[0],
            owned_tokens=receipt.owned_tokens_by_rank[0],
            max_replay_error=0.0,
        )
    assert coordinator.phase is Phase.FAILED


def test_inconsistent_trainer_replica_rejected(monkeypatch):
    coordinator, receipt = ready_and_commit(monkeypatch)
    verify_all(coordinator, receipt)
    updated = next_policy(coordinator)
    for rank in range(4):
        with pytest.raises(CycleError, match="replicas differ") if rank == 3 else _nullcontext():
            coordinator.acknowledge_update(
                rank,
                manifest_sha256=receipt.manifest_sha256,
                next_policy=updated,
                parameter_sha256=("9" if rank == 3 else "1") * 64,
                optimizer_sha256="2" * 64,
                optimizer_steps=1,
                gradient_sync_phases=1,
                changed_tensors=252,
            )
    assert coordinator.phase is Phase.FAILED


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def test_bad_manifest_handoff_fails_closed(monkeypatch):
    coordinator, _ = fixture_cycle()
    for role in ("rollout", "trainer"):
        for rank in range(4):
            coordinator.acknowledge_ready(role, rank, coordinator.policy)
    coordinator.begin_collection()

    def invalid(*args, **kwargs):
        raise ValueError("match digest differs")

    monkeypatch.setattr(module, "verify_completed_batch", invalid)
    with pytest.raises(CycleError, match="match digest differs"):
        coordinator.commit_batch(Path("unused"))
    assert coordinator.phase is Phase.FAILED


def test_manifest_handoff_checks_outcome_credit(monkeypatch):
    config = {"rollout": {"games_per_update": 4}}
    adapter_sha = "a" * 64
    manifest = PilotManifest.create(
        config=config,
        adapter_sha256=adapter_sha,
        policy_version="pilot:test",
        base_seed=11,
        target_games=4,
        replay_tolerance=2e-4,
    )
    entries = []
    matches = []
    for index in range(4):
        game_id = pilot_game_id(manifest.policy_version, index, 11 + index)
        entry = PilotMatchEntry(
            index=index, game_id=game_id, seed=11 + index,
            path=pilot_match_relative_path(index), sha256="1" * 64,
            policy_version=manifest.policy_version,
            turns=1, owned_tokens=4,
            final_results=(1.0, 0.0, 0.0, 0.0),
            termination_reason="natural_win",
            max_abs_log_prob_error=0.0,
        )
        entries.append(entry)
        matches.append(SimpleNamespace(
            game_id=game_id, final_results=entry.final_results,
            turns=[SimpleNamespace(
                seat=0, joint_step=0,
                credit=SimpleNamespace(
                    outcome_advantage=1.7320508075688772,
                    training_advantage=1.7320508075688772,
                ),
            )],
        ))
    manifest.matches = entries
    manifest.validate()
    policy = PolicyDescriptor(
        version=manifest.policy_version, update_index=0,
        adapter_sha256=adapter_sha, config_sha256=manifest.config_sha256,
        model_revision="pinned", tokenizer_sha256="2" * 64,
        grammar_sha256="3" * 64, run_kind="production",
    )
    monkeypatch.setattr(module, "read_pilot_manifest", lambda root: manifest)
    monkeypatch.setattr(module, "directory_sha256", lambda root: adapter_sha)
    monkeypatch.setattr(module, "read_and_replay_pilot_match", lambda root, m, i: (matches[i], "1" * 64))
    monkeypatch.setattr(module, "make_match_entry", lambda **kwargs: entries[kwargs["index"]])
    monkeypatch.setattr(module, "file_sha256", lambda path: "4" * 64)
    receipt = module.verify_completed_batch(
        Path("unused"), config=config, policy=policy, expected_games=4,
    )
    assert receipt.game_ids == tuple(entry.game_id for entry in entries)
    assert receipt.match_indices_by_rank == ((0,), (1,), (2,), (3,))
    matches[2].turns[0].credit.training_advantage = 0.0
    with pytest.raises(ValueError, match="Training credit differs"):
        module.verify_completed_batch(
            Path("unused"), config=config, policy=policy, expected_games=4,
        )
