"""Per-seat leave-one-out advantages remove turn-order credit without touching records."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rewards.outcome import FOUR_PLAYER_OUTCOME_SCALE, seat_baseline_advantages
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256
from self_play_grpo.training import trainer_handoff as module
from self_play_grpo.training.coordinator import BatchReceipt, PolicyDescriptor
from self_play_grpo.training.rollout_worker import prepare_frozen_batch


CONFIG = Path("self_play_grpo/configs/quoridor_outcome_64games.yaml")
WIN = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def test_leave_one_out_baseline_uses_only_other_matches():
    batch = [WIN[2], WIN[2], WIN[2], WIN[0]]
    advantages = seat_baseline_advantages(batch)
    # Seat 2 won the other two of three matches: winning again is worth 1 - 2/3.
    assert advantages[0][2] == pytest.approx((1.0 - 2.0 / 3.0) / FOUR_PLAYER_OUTCOME_SCALE)
    # Seat 2 losing the last match costs its full expected share.
    assert advantages[3][2] == pytest.approx((0.0 - 1.0) / FOUR_PLAYER_OUTCOME_SCALE)
    # A seat that never wins elsewhere gets zero for losing.
    assert advantages[0][1] == 0.0


def test_balanced_seats_reduce_to_the_fixed_game_baseline_in_expectation():
    batch = [WIN[seat] for seat in range(4)] * 8
    advantages = seat_baseline_advantages(batch)
    mean_by_seat = [sum(row[seat] for row in advantages) / len(advantages) for seat in range(4)]
    assert mean_by_seat == pytest.approx([0.0] * 4, abs=1e-12)


def test_invalid_batches_are_rejected():
    with pytest.raises(ValueError, match="at least two"):
        seat_baseline_advantages([WIN[0]])
    with pytest.raises(ValueError, match="sum to one"):
        seat_baseline_advantages([WIN[0], (0.5, 0.5, 0.5, 0.0)])


def test_config_rejects_unknown_baseline_and_keeps_old_digest():
    config = load_config(CONFIG)
    assert "advantage_baseline" not in config.to_dict()["training"]
    with pytest.raises(ValueError, match="advantage_baseline"):
        replace(config.training, advantage_baseline="median").validate(config.environment)


def _turn(seat, advantage):
    return SimpleNamespace(seat=seat, credit=SimpleNamespace(training_advantage=advantage))


def test_admission_rewrites_only_in_memory_training_advantages(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    config = replace(config, training=replace(config.training, advantage_baseline="seat_loo"))
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"cpu-only-adapter")
    policy = PolicyDescriptor(
        version="policy-000000", update_index=0, adapter_sha256=directory_sha256(source),
        config_sha256=canonical_sha256(config.to_dict()), model_revision=config.model.revision,
        tokenizer_sha256="b" * 64, grammar_sha256="c" * 64, run_kind="production",
    )
    root = tmp_path / "batch"
    prepare_frozen_batch(root, config=config, source_adapter=source, policy=policy,
                         base_seed=11, replay_tolerance=2e-4)
    digest = file_sha256(root / "manifest.json")
    indices = tuple(tuple(range(rank * 16, (rank + 1) * 16)) for rank in range(4))
    receipt = BatchReceipt(
        manifest_sha256=digest, policy_version=policy.version,
        adapter_sha256=policy.adapter_sha256, config_sha256=policy.config_sha256,
        game_ids=tuple(f"game-{index}" for index in range(64)), match_indices_by_rank=indices,
        turns_by_rank=(32, 32, 32, 32), owned_tokens_by_rank=(64, 64, 64, 64), max_replay_error=0.0,
    )
    # Seat 2 wins three quarters of the batch, seat 0 the rest.
    results = [WIN[2] if index % 4 else WIN[0] for index in range(64)]
    manifest = SimpleNamespace(
        matches=[SimpleNamespace(owned_tokens=4, final_results=row) for row in results],
        replay_tolerance=2e-4,
    )
    monkeypatch.setattr(module, "verify_completed_batch", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(module, "read_pilot_manifest", lambda path: manifest)
    monkeypatch.setattr(module, "load_trainer_match_shard", lambda path, value, shard: [
        SimpleNamespace(final_results=results[index], turns=[_turn(0, 9.0), _turn(2, 9.0)])
        for index in shard
    ])
    admitted = module.admit_trainer_shard(
        root, config=config, policy=policy, rank=0, expected_manifest_sha256=digest,
    )
    assert admitted.seat_result_means == pytest.approx((0.25, 0.0, 0.75, 0.0))
    expected = seat_baseline_advantages(results)
    for index, match in zip(indices[0], admitted.matches):
        assert [turn.credit.training_advantage for turn in match.turns] == [
            pytest.approx(expected[index][0]), pytest.approx(expected[index][2]),
        ]
    # A seat-2 win is now worth far less than a seat-0 win.
    assert expected[1][2] < expected[0][0] / 2
