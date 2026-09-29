"""Verified reward export never writes from a changed rollout source."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.rewards import export as module


def test_export_verified_outcome_dataset(tmp_path, monkeypatch):
    root = tmp_path / "rollout"
    output = tmp_path / "labels.jsonl"
    entry = SimpleNamespace(
        path="matches/game-000000.jsonl",
        sha256="1" * 64,
        game_id="game-1",
    )
    manifest = SimpleNamespace(matches=[entry])
    policy = SimpleNamespace(version="policy-000000", adapter_sha256="a" * 64)
    receipt = SimpleNamespace(manifest_sha256="d" * 64, game_ids=("game-1",))
    credit = SimpleNamespace(
        outcome_advantage=3 ** 0.5,
        training_advantage=3 ** 0.5,
    )
    match = SimpleNamespace(
        validate=lambda: None,
        game_id="game-1", seed=11, policy_version=policy.version,
        final_results=(1.0, 0.0, 0.0, 0.0),
        turns=[SimpleNamespace(
            joint_step=0, seat=0, observation="Board",
            policy_sample=SimpleNamespace(chosen_label="MOVE_A2"),
            credit=credit,
        )],
    )
    monkeypatch.setattr(module, "verify_completed_batch", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(module, "read_pilot_manifest", lambda path: manifest)
    monkeypatch.setattr(module, "read_matches_jsonl", lambda path: [match])
    actual_sha256 = module.file_sha256

    def source_hash(path: Path) -> str:
        if path.name == "manifest.json":
            return receipt.manifest_sha256
        if path.parent.name == "matches":
            return entry.sha256
        return actual_sha256(path)

    monkeypatch.setattr(module, "file_sha256", source_hash)
    report = module.export_verified_outcome_dataset(
        root, output, config={}, policy=policy, expected_games=1,
    )
    assert report["status"] == "complete"
    assert report["examples"] == 1
    assert report["source_manifest_sha256"] == receipt.manifest_sha256
    assert output.is_file()

    with pytest.raises(FileExistsError, match="already exists"):
        module.export_verified_outcome_dataset(
            root, output, config={}, policy=policy, expected_games=1,
        )


def test_export_rejects_modified_match_before_creating_dataset(tmp_path, monkeypatch):
    root = tmp_path / "rollout"
    output = tmp_path / "labels.jsonl"
    entry = SimpleNamespace(
        path="matches/game-000000.jsonl", sha256="1" * 64,
        game_id="game-1",
    )
    monkeypatch.setattr(module, "verify_completed_batch", lambda *args, **kwargs: SimpleNamespace(manifest_sha256="d" * 64))
    monkeypatch.setattr(module, "read_pilot_manifest", lambda path: SimpleNamespace(matches=[entry]))
    monkeypatch.setattr(module, "file_sha256", lambda path: "2" * 64)
    with pytest.raises(ValueError, match="source match changed"):
        module.export_verified_outcome_dataset(
            root, output, config={}, policy=SimpleNamespace(), expected_games=1,
        )
    assert not output.exists()
