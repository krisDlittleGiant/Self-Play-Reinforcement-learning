"""Model-free format-3 checkpoint integrity and identity tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from self_play_grpo.training.distributed_checkpoint import (
    ABSENT_TRAINER_STATE,
    MANIFEST_PATH,
    SHARDING_RULE,
    CheckpointFile,
    DistributedCheckpointManifest,
    TrainerRankState,
    file_sha256,
    read_distributed_manifest,
    validate_rank_rng_record,
    validate_resume_identity,
    verify_checkpoint_files,
)


def _digest(character: str) -> str:
    return character * 64


def _fixture(root: Path) -> DistributedCheckpointManifest:
    ranks = tuple(
        TrainerRankState(
            rank=rank,
            local_rank=rank,
            module_id=str(rank),
            device=f"hpu:{rank}",
            rng_path=f"distributed/rng/rank-{rank:03d}.pt",
            match_indices=(rank * 2, rank * 2 + 1),
            match_ids=(f"game-{rank * 2}", f"game-{rank * 2 + 1}"),
            match_sha256=(_digest("a"), _digest("b")),
        )
        for rank in range(2)
    )
    paths = (
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "distributed/rng/rank-000.pt",
        "distributed/rng/rank-001.pt",
        "optimizer_state.pt",
        "torch_rng_state.pt",
        "trainer_state.json",
    )
    files = []
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{name}".encode("utf-8"))
        files.append(CheckpointFile(name, path.stat().st_size, file_sha256(path)))
    manifest = DistributedCheckpointManifest(
        run_kind="validation",
        run_id="d4-fixture",
        policy_version="policy-000001",
        update_index=1,
        config_sha256=_digest("1"),
        model_id="Qwen/Qwen3-4B",
        model_revision="1cfa9a7208912126459214e8b04321603b3df60c",
        adapter_schema_sha256=_digest("2"),
        optimizer_schema_sha256=_digest("3"),
        tokenizer_sha256=_digest("4"),
        grammar_sha256=_digest("5"),
        code_identity="source-tree-sha256:fixture",
        runtime_identity=(
            ("habana", "fixture"), ("peft", "0.20.0"), ("python", "3.12"),
            ("torch", "2.7.1+hpu"), ("transformers", "5.12.1"),
        ),
        attention_backend="sdpa",
        dtype="bfloat16",
        backend="nccl",
        trainer_world_size=2,
        source_rollout_manifest_sha256=_digest("6"),
        ranks=ranks,
        files=tuple(files),
    )
    manifest.validate()
    path = root / MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8")
    return manifest


def _expectation(manifest: DistributedCheckpointManifest) -> dict[str, object]:
    value = manifest.to_dict()
    value["rank_bindings"] = [
        {"rank": rank.rank, "local_rank": rank.local_rank,
         "module_id": rank.module_id, "device": rank.device}
        for rank in manifest.ranks
    ]
    value["match_shards"] = [
        {"match_indices": list(rank.match_indices), "match_ids": list(rank.match_ids),
         "match_sha256": list(rank.match_sha256)}
        for rank in manifest.ranks
    ]
    return {
        key: value[key]
        for key in (
            "run_id", "policy_version", "update_index", "config_sha256", "model_id",
            "model_revision", "adapter_schema_sha256", "optimizer_schema_sha256",
            "tokenizer_sha256", "grammar_sha256", "code_identity", "runtime_identity",
            "attention_backend", "dtype", "backend", "trainer_world_size",
            "source_rollout_manifest_sha256", "sharding_rule", "rank_bindings",
            "match_shards",
        )
    }


def test_format_three_round_trip_and_complete_file_inventory(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    assert read_distributed_manifest(tmp_path) == manifest
    assert manifest.checkpoint_format == 3
    assert manifest.absent_state == ABSENT_TRAINER_STATE
    assert manifest.sharding_rule == SHARDING_RULE
    verify_checkpoint_files(tmp_path, manifest)


def test_old_checkpoint_format_is_not_misread_as_distributed(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    payload = manifest.to_dict()
    payload["checkpoint_format"] = 2
    with pytest.raises(ValueError, match="Unsupported distributed checkpoint format"):
        DistributedCheckpointManifest.from_dict(payload)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../b", "a//b", "a\\b", "C:drive"])
def test_unsafe_manifest_paths_are_rejected(tmp_path: Path, path: str) -> None:
    manifest = _fixture(tmp_path)
    files = (replace(manifest.files[0], path=path),) + manifest.files[1:]
    with pytest.raises(ValueError, match="safe checkpoint-relative path"):
        replace(manifest, files=files).validate()


def test_duplicate_or_missing_rank_is_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(manifest, ranks=(manifest.ranks[0], manifest.ranks[0])).validate()
    with pytest.raises(ValueError, match="count differs"):
        replace(manifest, ranks=manifest.ranks[:1]).validate()


def test_unequal_match_shard_is_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    changed = replace(manifest.ranks[1], match_indices=(9, 10))
    with pytest.raises(ValueError, match="equal contiguous"):
        replace(manifest, ranks=(manifest.ranks[0], changed)).validate()


@pytest.mark.parametrize("name", ["config_sha256", "model_revision", "source_rollout_manifest_sha256"])
def test_resume_rejects_identity_drift(tmp_path: Path, name: str) -> None:
    manifest = _fixture(tmp_path)
    expected = _expectation(manifest)
    expected[name] = "different" if name == "model_revision" else _digest("f")
    with pytest.raises(ValueError, match=name):
        validate_resume_identity(manifest, expected=expected, allow_validation=True)


def test_resume_rejects_rank_binding_and_shard_drift(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    expected = _expectation(manifest)
    bindings = [dict(row) for row in expected["rank_bindings"]]
    bindings[1]["module_id"] = "7"
    expected["rank_bindings"] = bindings
    with pytest.raises(ValueError, match="rank_bindings"):
        validate_resume_identity(manifest, expected=expected, allow_validation=True)
    expected = _expectation(manifest)
    shards = [dict(row) for row in expected["match_shards"]]
    shards[1]["match_ids"] = ["other", "other2"]
    expected["match_shards"] = shards
    with pytest.raises(ValueError, match="match_shards"):
        validate_resume_identity(manifest, expected=expected, allow_validation=True)


def test_validation_only_checkpoint_is_not_a_production_resume(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    expected = _expectation(manifest)
    with pytest.raises(ValueError, match="validation-only"):
        validate_resume_identity(manifest, expected=expected)
    validate_resume_identity(manifest, expected=expected, allow_validation=True)


def test_corrupt_or_missing_checkpoint_file_is_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    target = tmp_path / "optimizer_state.pt"
    target.write_bytes(b"X" * target.stat().st_size)
    with pytest.raises(ValueError, match="digest differs: optimizer_state.pt"):
        verify_checkpoint_files(tmp_path, manifest)
    target.unlink()
    with pytest.raises(ValueError, match="inventory differs"):
        verify_checkpoint_files(tmp_path, manifest)


def test_unexpected_file_or_escaping_symlink_is_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    extra = tmp_path / "extra.pt"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory differs"):
        verify_checkpoint_files(tmp_path, manifest)
    extra.unlink()
    external = tmp_path.parent / "outside-checkpoint.pt"
    external.write_bytes(b"outside")
    link = tmp_path / "linked.pt"
    link.symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        verify_checkpoint_files(tmp_path, manifest)


def test_duplicate_json_keys_are_rejected_before_schema_parsing(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / MANIFEST_PATH).write_text('{"checkpoint_format":3,"checkpoint_format":3}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_distributed_manifest(tmp_path)


@pytest.mark.torch
def test_rank_rng_payload_must_match_filename_and_binding(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    manifest = _fixture(tmp_path)
    record = {
        "schema_version": 1,
        "rank": 0,
        "module_id": "0",
        "cpu": torch.get_rng_state(),
        "hpu": [torch.ones(8, dtype=torch.uint8)],
        "hpu_api": "torch.hpu.get_rng_state_all",
    }
    target = tmp_path / manifest.ranks[0].rng_path
    torch.save(record, target)
    validate_rank_rng_record(tmp_path, manifest.ranks[0])
    record["rank"] = 1
    torch.save(record, target)
    with pytest.raises(ValueError, match="filename rank"):
        validate_rank_rng_record(tmp_path, manifest.ranks[0])
