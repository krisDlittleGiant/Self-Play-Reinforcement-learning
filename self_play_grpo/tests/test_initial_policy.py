"""CPU-only initial policy provenance and adapter-copy tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import directory_sha256
from self_play_grpo.training import initial_policy as module


CONFIG = Path("self_play_grpo/configs/quoridor_outcome_64games.yaml")


def fixture(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    pilot = tmp_path / "pilot"
    adapter = pilot / "policy_adapter"
    adapter.mkdir(parents=True)
    (pilot / "manifest.json").write_text("fixture")
    (adapter / "adapter_model.safetensors").write_bytes(b"frozen initial pilot weights")
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": config.model.lora_rank,
        "lora_alpha": config.model.lora_alpha,
        "lora_dropout": config.model.dropout,
        "base_model_name_or_path": config.model.id,
    }))
    pilot_config = config.to_dict()
    pilot_config["rollout"]["games_per_update"] = 16
    pilot_config["rollout"]["parallel_games_per_rank"] = 1
    manifest = SimpleNamespace(
        policy_version="pilot:fixture", matches=(SimpleNamespace(sha256="f" * 64),),
        config=pilot_config, adapter_sha256=directory_sha256(adapter),
    )
    monkeypatch.setattr(module, "read_pilot_manifest", lambda path: manifest)
    monkeypatch.setattr(module, "read_and_replay_pilot_match", lambda *args: (
        SimpleNamespace(turns=(1,)), "f" * 64,
    ))
    monkeypatch.setattr(module, "_tokenizer_digest", lambda config: "a" * 64)
    return config, pilot, manifest


def test_recorded_pilot_produces_new_immutable_initial_bundle(tmp_path, monkeypatch):
    config, pilot, manifest = fixture(tmp_path, monkeypatch)
    target = tmp_path / "initial-policy"
    descriptor, provenance = module.prepare_initial_policy(
        pilot_root=pilot, config=config, output=target,
    )
    assert descriptor.version == "policy-000000"
    assert descriptor.adapter_sha256 == manifest.adapter_sha256
    assert directory_sha256(target / "policy_adapter") == manifest.adapter_sha256
    assert json.loads((target / "policy_descriptor.json").read_text())["run_kind"] == "production"
    assert provenance["source_pilot_policy_version"] == "pilot:fixture"
    with pytest.raises(FileExistsError, match="output must be new"):
        module.prepare_initial_policy(pilot_root=pilot, config=config, output=target)


def test_synthetic_or_changed_pilot_adapter_rejected(tmp_path, monkeypatch):
    config, pilot, manifest = fixture(tmp_path, monkeypatch)
    manifest.policy_version = "policy-000001"
    with pytest.raises(ValueError, match="recorded pilot"):
        module.prepare_initial_policy(pilot_root=pilot, config=config, output=tmp_path / "out")
    manifest.policy_version = "pilot:fixture"
    (pilot / "policy_adapter" / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="adapter digest differs"):
        module.prepare_initial_policy(pilot_root=pilot, config=config, output=tmp_path / "out")
