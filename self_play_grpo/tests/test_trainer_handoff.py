"""CPU-only fail-closed contracts for D5 trainer batch admission."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256
from self_play_grpo.training import trainer_handoff as module
from self_play_grpo.training.coordinator import BatchReceipt, PolicyDescriptor
from self_play_grpo.training.rollout_worker import prepare_frozen_batch


CONFIG = Path("self_play_grpo/configs/quoridor_outcome_64games.yaml")


def _fixture(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"cpu-only-adapter")
    policy = PolicyDescriptor(
        version="policy-000000", update_index=0,
        adapter_sha256=directory_sha256(source),
        config_sha256=canonical_sha256(config.to_dict()),
        model_revision=config.model.revision,
        tokenizer_sha256="b" * 64, grammar_sha256="c" * 64,
        run_kind="production",
    )
    root = tmp_path / "batch"
    prepare_frozen_batch(
        root, config=config, source_adapter=source, policy=policy,
        base_seed=11, replay_tolerance=2e-4,
    )
    digest = file_sha256(root / "manifest.json")
    indices = tuple(tuple(range(rank * 16, (rank + 1) * 16)) for rank in range(4))
    receipt = BatchReceipt(
        manifest_sha256=digest, policy_version=policy.version,
        adapter_sha256=policy.adapter_sha256, config_sha256=policy.config_sha256,
        game_ids=tuple(f"game-{index}" for index in range(64)),
        match_indices_by_rank=indices,
        turns_by_rank=(16, 16, 16, 16),
        owned_tokens_by_rank=(64, 64, 64, 64),
        max_replay_error=0.0,
    )
    manifest = SimpleNamespace(
        matches=[SimpleNamespace(owned_tokens=4) for _ in range(64)],
        replay_tolerance=2e-4,
    )
    calls = []

    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return receipt

    monkeypatch.setattr(module, "verify_completed_batch", verify)
    monkeypatch.setattr(module, "read_pilot_manifest", lambda path: manifest)
    monkeypatch.setattr(
        module, "load_trainer_match_shard",
        lambda path, value, shard: [SimpleNamespace(turns=[object()]) for _ in shard],
    )
    return root, config, policy, digest, calls


def test_admits_exact_trainer_shard_only_after_complete_batch_verification(tmp_path, monkeypatch):
    root, config, policy, digest, calls = _fixture(tmp_path, monkeypatch)
    admitted = module.admit_trainer_shard(
        root, config=config, policy=policy, rank=2, expected_manifest_sha256=digest,
    )
    assert admitted.match_indices == tuple(range(32, 48))
    assert len(admitted.matches) == 16
    assert admitted.receipt.manifest_sha256 == digest
    assert len(calls) == 1
    assert calls[0][1]["expected_games"] == 64


def test_bad_manifest_digest_stops_before_global_verification(tmp_path, monkeypatch):
    root, config, policy, _, calls = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="coordinator receipt"):
        module.admit_trainer_shard(
            root, config=config, policy=policy, rank=0,
            expected_manifest_sha256="0" * 64,
        )
    assert calls == []


def test_stale_policy_descriptor_stops_before_global_verification(tmp_path, monkeypatch):
    root, config, policy, digest, calls = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="descriptor differs"):
        module.admit_trainer_shard(
            root, config=config,
            policy=replace(policy, tokenizer_sha256="d" * 64),
            rank=0, expected_manifest_sha256=digest,
        )
    assert calls == []


def test_validation_policy_and_wrong_rank_are_rejected(tmp_path, monkeypatch):
    root, config, policy, digest, calls = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="production policy"):
        module.admit_trainer_shard(
            root, config=config, policy=replace(policy, run_kind="validation"),
            rank=0, expected_manifest_sha256=digest,
        )
    with pytest.raises(ValueError, match="rank must be"):
        module.admit_trainer_shard(
            root, config=config, policy=policy, rank=4,
            expected_manifest_sha256=digest,
        )
    assert calls == []


def test_short_shard_or_changed_manifest_is_rejected(tmp_path, monkeypatch):
    root, config, policy, digest, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module, "load_trainer_match_shard",
        lambda path, value, shard: [SimpleNamespace(turns=[object()]) for _ in shard[:-1]],
    )
    with pytest.raises(ValueError, match="16 complete matches"):
        module.admit_trainer_shard(
            root, config=config, policy=policy, rank=1,
            expected_manifest_sha256=digest,
        )

    def mutate_manifest(path, manifest, shard):
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")
        return [SimpleNamespace(turns=[object()]) for _ in shard]

    monkeypatch.setattr(module, "load_trainer_match_shard", mutate_manifest)
    with pytest.raises(ValueError, match="changed during trainer admission"):
        module.admit_trainer_shard(
            root, config=config, policy=policy, rank=1,
            expected_manifest_sha256=digest,
        )
