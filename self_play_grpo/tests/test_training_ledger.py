"""CPU-only restart-ledger contracts; no HPU workers are started."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.rollouts.pilot import file_sha256
from self_play_grpo.training import ledger as module
from self_play_grpo.training.ledger import CommitRecord, publish_commit, read_commits


def _checkpoint(root: Path, index: int, batch: str, run_id: str = "production-test") -> tuple[str, str]:
    relative = f"checkpoints/policy-{index:06d}"
    manifest = root / relative / "distributed" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"test": index}))
    return relative, file_sha256(manifest)


def _record(root: Path, index: int, batch: str, previous: str | None = None) -> CommitRecord:
    relative, digest = _checkpoint(root, index, batch)
    return CommitRecord(
        run_id="production-test", update_index=index,
        policy_version=f"policy-{index:06d}", source_manifest_sha256=batch,
        checkpoint_path=relative, checkpoint_manifest_sha256=digest,
        previous_record_sha256=previous,
    )


@pytest.fixture
def checkpoint_contract(monkeypatch):
    def reader(path):
        index = int(Path(path).name.split("-")[-1])
        return SimpleNamespace(
            run_kind="production", run_id="production-test", update_index=index,
            policy_version=f"policy-{index:06d}",
            source_rollout_manifest_sha256=("a" if index == 1 else "b") * 64,
        )

    monkeypatch.setattr(module, "read_distributed_manifest", reader)
    monkeypatch.setattr(module, "verify_checkpoint_files", lambda *args: None)


def test_publish_and_read_two_checkpoint_linked_updates(tmp_path, checkpoint_contract):
    first = _record(tmp_path, 1, "a" * 64)
    publish_commit(tmp_path, first)
    second = _record(tmp_path, 2, "b" * 64, previous=first.sha256)
    publish_commit(tmp_path, second)
    assert read_commits(tmp_path) == (first, second)
    with pytest.raises(ValueError, match="immediately follow"):
        publish_commit(tmp_path, second)


def test_reused_batch_rejected(tmp_path, checkpoint_contract):
    first = _record(tmp_path, 1, "a" * 64)
    publish_commit(tmp_path, first)
    second = _record(tmp_path, 2, "a" * 64, previous=first.sha256)
    with pytest.raises(ValueError, match="already been consumed"):
        publish_commit(tmp_path, second)


def test_tampered_record_and_checkpoint_rejected(tmp_path, checkpoint_contract):
    record = _record(tmp_path, 1, "a" * 64)
    path = publish_commit(tmp_path, record)
    checkpoint = tmp_path / record.checkpoint_path / "distributed" / "manifest.json"
    checkpoint.write_text('{"test":999}')
    with pytest.raises(ValueError, match="digest differs"):
        read_commits(tmp_path)
    checkpoint.write_text('{"test": 1}')
    data = json.loads(path.read_text())
    data["source_manifest_sha256"] = "c" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity differs"):
        read_commits(tmp_path)


def test_broken_chain_and_sparse_update_rejected(tmp_path, checkpoint_contract):
    first = _record(tmp_path, 1, "a" * 64)
    publish_commit(tmp_path, first)
    second = _record(tmp_path, 2, "b" * 64, previous="f" * 64)
    with pytest.raises(ValueError, match="predecessor"):
        publish_commit(tmp_path, second)
    sparse = _record(tmp_path, 3, "b" * 64, previous=first.sha256)
    with pytest.raises(ValueError, match="immediately follow"):
        publish_commit(tmp_path, sparse)


def test_validation_checkpoint_and_path_escape_rejected(tmp_path, checkpoint_contract, monkeypatch):
    record = _record(tmp_path, 1, "a" * 64)
    with pytest.raises(ValueError, match="safe root-relative"):
        replace(record, checkpoint_path="../elsewhere").validate()
    monkeypatch.setattr(module, "read_distributed_manifest", lambda _: SimpleNamespace(run_kind="validation"))
    with pytest.raises(ValueError, match="Validation-only"):
        publish_commit(tmp_path, record)
