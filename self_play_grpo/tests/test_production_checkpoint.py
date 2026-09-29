"""CPU-only format-3 production checkpoint-template contracts."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import (
    PilotMatchEntry, canonical_sha256, directory_sha256, file_sha256,
    pilot_game_id, pilot_match_relative_path, write_pilot_manifest,
)
from self_play_grpo.training.coordinator import BatchReceipt, PolicyDescriptor
from self_play_grpo.training.production_checkpoint import (
    build_production_checkpoint_template,
)
from self_play_grpo.training.rollout_worker import prepare_frozen_batch


CONFIG = Path("self_play_grpo/configs/quoridor_outcome_64games.yaml")
RUNTIME = {"python": "3.12", "torch": "test", "transformers": "test", "peft": "test", "habana": "test"}


def _fixture(tmp_path):
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
    manifest = prepare_frozen_batch(
        root, config=config, source_adapter=source, policy=policy,
        base_seed=11, replay_tolerance=2e-4,
    )
    for index in range(64):
        seed = 11 + index
        manifest.append(PilotMatchEntry(
            index=index, game_id=pilot_game_id(policy.version, index, seed),
            seed=seed, path=pilot_match_relative_path(index),
            sha256=f"{index + 1:064x}", policy_version=policy.version,
            turns=1, owned_tokens=4,
            final_results=(1.0, 0.0, 0.0, 0.0),
            termination_reason="natural_win", max_abs_log_prob_error=0.0,
        ))
    write_pilot_manifest(root, manifest)
    shards = tuple(tuple(range(rank * 16, (rank + 1) * 16)) for rank in range(4))
    receipt = BatchReceipt(
        manifest_sha256=file_sha256(root / "manifest.json"),
        policy_version=policy.version, adapter_sha256=policy.adapter_sha256,
        config_sha256=policy.config_sha256,
        game_ids=tuple(entry.game_id for entry in manifest.matches),
        match_indices_by_rank=shards,
        turns_by_rank=(16, 16, 16, 16),
        owned_tokens_by_rank=(64, 64, 64, 64),
        max_replay_error=0.0,
    )
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(_attn_implementation="sdpa")
    trainer = SimpleNamespace(
        config=config, policy=SimpleNamespace(model=model),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
        update_index=1, policy_version="policy-000001",
    )
    return root, receipt, policy, trainer


def _build(root, receipt, policy, trainer, **kwargs):
    return build_production_checkpoint_template(
        run_id="one-cycle", rollout_root=root, receipt=receipt,
        policy=policy, trainer=trainer, trainer_module_ids=(4, 5, 6, 7),
        runtime_identity=RUNTIME, **kwargs,
    )


def test_template_binds_64_games_and_four_rank_rng_paths(tmp_path):
    root, receipt, policy, trainer = _fixture(tmp_path)
    template = _build(root, receipt, policy, trainer)
    assert template.run_kind == "production"
    assert template.policy_version == "policy-000001"
    assert template.source_rollout_manifest_sha256 == receipt.manifest_sha256
    assert template.files == ()  # Completed atomically by save_checkpoint.
    assert tuple(rank.module_id for rank in template.ranks) == ("4", "5", "6", "7")
    assert tuple(rank.rng_path for rank in template.ranks) == tuple(
        f"distributed/rng/rank-{rank:03d}.pt" for rank in range(4)
    )
    assert tuple(rank.match_indices for rank in template.ranks) == receipt.match_indices_by_rank
    assert template.code_identity.startswith("source-sha256:")


def test_template_rejects_stale_or_validation_only_policy(tmp_path):
    root, receipt, policy, trainer = _fixture(tmp_path)
    with pytest.raises(ValueError, match="production policy"):
        _build(root, receipt, replace(policy, run_kind="validation"), trainer)
    with pytest.raises(ValueError, match="exactly one"):
        trainer.update_index = 2
        _build(root, receipt, policy, trainer)


def test_template_rejects_changed_manifest_or_receipt_ids(tmp_path):
    root, receipt, policy, trainer = _fixture(tmp_path)
    with pytest.raises(ValueError, match="digest differs"):
        _build(root, replace(receipt, manifest_sha256="0" * 64), policy, trainer)
    ids = list(receipt.game_ids)
    ids[1] = "altered-game-id"
    with pytest.raises(ValueError, match="match IDs differ"):
        _build(root, replace(receipt, game_ids=tuple(ids)), policy, trainer)


def test_template_rejects_incomplete_runtime_or_wrong_modules(tmp_path):
    root, receipt, policy, trainer = _fixture(tmp_path)
    with pytest.raises(ValueError, match="runtime identity"):
        build_production_checkpoint_template(
            run_id="one-cycle", rollout_root=root, receipt=receipt,
            policy=policy, trainer=trainer, trainer_module_ids=(4, 5, 6, 7),
            runtime_identity={"torch": "test"},
        )
    with pytest.raises(ValueError, match="distinct trainer module"):
        build_production_checkpoint_template(
            run_id="one-cycle", rollout_root=root, receipt=receipt,
            policy=policy, trainer=trainer, trainer_module_ids=(4, 5, 5, 7),
            runtime_identity=RUNTIME,
        )
