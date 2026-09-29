"""CPU-only contracts for the D5 frozen-adapter rollout shard."""

from dataclasses import replace
from pathlib import Path

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.distributed import (
    DistributedRankReport, rank_match_indices, write_rank_report,
)
from self_play_grpo.rollouts.pilot import (
    PilotMatchEntry, canonical_sha256, directory_sha256,
    pilot_game_id, pilot_match_relative_path, read_pilot_manifest,
)
from self_play_grpo.training import rollout_worker as module
from self_play_grpo.training.coordinator import PolicyDescriptor


CONFIG = Path("self_play_grpo/configs/quoridor_outcome_64games.yaml")


def _fixture(tmp_path):
    config = load_config(CONFIG)
    adapter = tmp_path / "source_adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"cpu-only-fixture")
    policy = PolicyDescriptor(
        version="policy-000000", update_index=0,
        adapter_sha256=directory_sha256(adapter),
        config_sha256=canonical_sha256(config.to_dict()),
        model_revision=config.model.revision,
        tokenizer_sha256="b" * 64, grammar_sha256="c" * 64,
        run_kind="production",
    )
    return config, adapter, policy


def _prepared(tmp_path):
    config, adapter, policy = _fixture(tmp_path)
    root = tmp_path / "batch"
    manifest = module.prepare_frozen_batch(
        root, config=config, source_adapter=adapter, policy=policy,
        base_seed=11, replay_tolerance=2e-4,
    )
    return root, config, policy, manifest


def _synthetic_report(rank, manifest):
    entries = []
    for index in rank_match_indices(rank, 4, 16):
        seed = manifest.base_seed + index
        entries.append(PilotMatchEntry(
            index=index,
            game_id=pilot_game_id(manifest.policy_version, index, seed),
            seed=seed, path=pilot_match_relative_path(index),
            sha256=f"{index + 1:064x}",
            policy_version=manifest.policy_version,
            turns=1, owned_tokens=4,
            final_results=(1.0, 0.0, 0.0, 0.0),
            termination_reason="natural_win",
            max_abs_log_prob_error=0.0,
        ))
    return DistributedRankReport(
        rank=rank, local_rank=rank, world_size=4, games_per_rank=16,
        policy_version=manifest.policy_version,
        adapter_sha256=manifest.adapter_sha256,
        matches=tuple(entries),
    )


def test_prepare_copies_adapter_and_creates_unpublished_64_game_manifest(tmp_path):
    root, config, policy, manifest = _prepared(tmp_path)
    assert manifest.target_games == 64
    assert manifest.matches == []
    assert directory_sha256(root / "policy_adapter") == policy.adapter_sha256
    assert module._validate_prepared(root, config, policy).policy_version == policy.version
    with pytest.raises(FileExistsError, match="must be new"):
        module.prepare_frozen_batch(
            root, config=config, source_adapter=tmp_path / "source_adapter",
            policy=policy, base_seed=11, replay_tolerance=2e-4,
        )


def test_prepare_rejects_wrong_policy_or_profile(tmp_path):
    config, adapter, policy = _fixture(tmp_path)
    with pytest.raises(ValueError, match="configuration differs"):
        module.prepare_frozen_batch(
            tmp_path / "batch", config=config, source_adapter=adapter,
            policy=replace(policy, config_sha256="d" * 64),
            base_seed=11, replay_tolerance=2e-4,
        )
    invalid = load_config(Path("self_play_grpo/configs/quoridor_outcome.yaml"))
    with pytest.raises(ValueError, match="64 games"):
        module.prepare_frozen_batch(
            tmp_path / "batch", config=invalid, source_adapter=adapter,
            policy=policy, base_seed=11, replay_tolerance=2e-4,
        )
    assert not (tmp_path / "batch").exists()


def test_worker_binding_requires_explicit_physical_module(monkeypatch):
    for name, value in {
        "SP_GRPO_ROLE": "rollout", "SP_GRPO_RANK": "2",
        "SP_GRPO_MODULE_ID": "6", "HLS_MODULE_ID": "6",
        "HABANA_VISIBLE_MODULES": "2,4,6,7",
    }.items():
        monkeypatch.setenv(name, value)
    assert module._worker_binding(2) == 6
    monkeypatch.setenv("HLS_MODULE_ID", "5")
    with pytest.raises(RuntimeError, match="physical module"):
        module._worker_binding(2)


def test_aggregate_requires_four_reports_and_identical_tensor_hashes(tmp_path, monkeypatch):
    root, config, policy, manifest = _prepared(tmp_path)
    replayed = []
    monkeypatch.setattr(module, "validate_registered_matches", lambda *args: replayed.append(args))
    for rank in range(3):
        write_rank_report(root, _synthetic_report(rank, manifest), manifest)
        module._write_identity(root, rank=rank, module_id=rank, parameter_sha256="a" * 64)
    with pytest.raises(FileNotFoundError):
        module.aggregate_rank_shards(
            root, config=config, policy=policy, expected_modules=(0, 1, 2, 3),
        )
    assert read_pilot_manifest(root).matches == []

    write_rank_report(root, _synthetic_report(3, manifest), manifest)
    module._write_identity(root, rank=3, module_id=3, parameter_sha256="b" * 64)
    with pytest.raises(ValueError, match="different trainable tensor"):
        module.aggregate_rank_shards(
            root, config=config, policy=policy, expected_modules=(0, 1, 2, 3),
        )
    assert read_pilot_manifest(root).matches == []
    assert replayed == []


def test_aggregate_publishes_only_after_registered_match_validation(tmp_path, monkeypatch):
    root, config, policy, manifest = _prepared(tmp_path)
    for rank in range(4):
        write_rank_report(root, _synthetic_report(rank, manifest), manifest)
        module._write_identity(root, rank=rank, module_id=rank, parameter_sha256="a" * 64)

    def fail_replay(*args):
        raise ValueError("match artifact is invalid")

    monkeypatch.setattr(module, "validate_registered_matches", fail_replay)
    with pytest.raises(ValueError, match="match artifact"):
        module.aggregate_rank_shards(
            root, config=config, policy=policy, expected_modules=(0, 1, 2, 3),
        )
    assert read_pilot_manifest(root).matches == []

    observed = []
    monkeypatch.setattr(module, "validate_registered_matches", lambda path, value: observed.append(len(value.matches)))
    published = module.aggregate_rank_shards(
        root, config=config, policy=policy, expected_modules=(0, 1, 2, 3),
    )
    assert observed == [64]
    assert len(published.matches) == 64
    assert len(read_pilot_manifest(root).matches) == 64


def test_production_descriptor_is_frozen_with_batch(tmp_path):
    root, config, policy, _ = _prepared(tmp_path)
    with pytest.raises(ValueError, match="production policy"):
        module._validate_prepared(root, config, replace(policy, run_kind="validation"))
    with pytest.raises(ValueError, match="descriptor changed"):
        module._validate_prepared(root, config, replace(policy, tokenizer_sha256="e" * 64))
    assert (root / "policy_descriptor.json").is_file()
