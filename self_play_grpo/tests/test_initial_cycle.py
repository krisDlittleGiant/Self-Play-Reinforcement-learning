"""CPU-only prelaunch and command-shape tests for the first D5 cycle."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256
from self_play_grpo.training import initial_cycle as module
from self_play_grpo.training.coordinator import PolicyDescriptor


CONFIG = Path("self_play_grpo/configs/quoridor_outcome_64games.yaml")


def test_worker_command_has_exact_role_rank_and_interpreter():
    commands = module._commands(
        "rollout", (2, 3, 5, 6),
        ["self_play_grpo.training.rollout_worker", "--root", "batch"],
    )
    assert len(commands) == 4
    assert [command.module_id for command in commands] == [2, 3, 5, 6]
    assert commands[2].argv == (
        sys.executable, "-m", "self_play_grpo.training.rollout_worker",
        "--root", "batch", "--rank", "2",
    )


def test_initial_descriptor_requires_exact_frozen_adapter(tmp_path):
    config = load_config(CONFIG)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fresh-initial-adapter")
    descriptor = PolicyDescriptor(
        version="policy-000000", update_index=0,
        adapter_sha256=directory_sha256(adapter),
        config_sha256=canonical_sha256(config.to_dict()),
        model_revision=config.model.revision, tokenizer_sha256="a" * 64,
        grammar_sha256="b" * 64, run_kind="production",
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(asdict(descriptor)))
    assert module._initial_policy(path, adapter, config) == descriptor
    (adapter / "adapter_model.safetensors").write_bytes(b"different")
    with pytest.raises(ValueError, match="adapter digest differs"):
        module._initial_policy(path, adapter, config)


def test_d4_failure_prevents_run_output_and_child_launch(tmp_path, monkeypatch):
    def reject(**kwargs):
        raise FileNotFoundError("D4 four-rank gate missing")

    def forbidden(*args, **kwargs):
        raise AssertionError("No D5 child launch or adapter load may occur")

    monkeypatch.setattr(module, "verify_d5_launch_prerequisites", reject)
    monkeypatch.setattr(module, "_initial_policy", forbidden)
    monkeypatch.setattr(module, "supervise_role_phase", forbidden)
    output = tmp_path / "run"
    args = SimpleNamespace(
        config=CONFIG, rollout_modules="0,1,2,3", trainer_modules="4,5,6,7",
        d4_two_summary=tmp_path / "two", d4_four_summary=tmp_path / "four",
        output=output,
    )
    with pytest.raises(FileNotFoundError, match="D4 four-rank"):
        module.run_initial_cycle(args)
    assert not output.exists()
