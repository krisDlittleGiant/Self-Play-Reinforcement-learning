"""Append-only production update ledger for D5/D6 restart boundaries.

A record is published only after its format-3 checkpoint is complete.  The
ledger is deliberately not a launcher: it makes repeated batch consumption
and a stale/inconsistent restart visible before any optimizer is restored.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from self_play_grpo.rollouts.pilot import canonical_sha256, file_sha256
from self_play_grpo.training.distributed_checkpoint import (
    MANIFEST_PATH,
    read_distributed_manifest,
    verify_checkpoint_files,
)


_RECORD_NAME = re.compile(r"update-([0-9]{6})\.json\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _checkpoint_path(value: str) -> None:
    if (
        not isinstance(value, str) or not value
        or value.startswith("/") or "\\" in value or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or PurePosixPath(value).as_posix() != value
    ):
        raise ValueError("checkpoint_path must be a safe root-relative path")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate ledger key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class CommitRecord:
    run_id: str
    update_index: int
    policy_version: str
    source_manifest_sha256: str
    checkpoint_path: str
    checkpoint_manifest_sha256: str
    previous_record_sha256: str | None
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1 or type(self.update_index) is not int or self.update_index < 1:
            raise ValueError("Invalid production ledger schema or update index")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Ledger run_id must be nonempty")
        if self.policy_version != f"policy-{self.update_index:06d}":
            raise ValueError("Ledger policy version differs from update index")
        _digest(self.source_manifest_sha256, "source_manifest_sha256")
        _digest(self.checkpoint_manifest_sha256, "checkpoint_manifest_sha256")
        _checkpoint_path(self.checkpoint_path)
        if self.update_index == 1:
            if self.previous_record_sha256 is not None:
                raise ValueError("Initial ledger record must not name a predecessor")
        else:
            _digest(self.previous_record_sha256, "previous_record_sha256")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "update_index": self.update_index,
            "policy_version": self.policy_version,
            "source_manifest_sha256": self.source_manifest_sha256,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "previous_record_sha256": self.previous_record_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CommitRecord":
        required = set(cls("", 1, "", "", "", "", None).__dict__)
        if set(value) != required:
            raise ValueError("Ledger record has missing or unexpected fields")
        record = cls(**value)
        record.validate()
        return record

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _verify_checkpoint(root: Path, record: CommitRecord) -> None:
    checkpoint = root / record.checkpoint_path
    if checkpoint.is_symlink():
        raise ValueError("Ledger checkpoint path is a symlink")
    resolved_root = root.resolve(strict=True)
    resolved_checkpoint = checkpoint.resolve(strict=True)
    if not resolved_checkpoint.is_relative_to(resolved_root):
        raise ValueError("Ledger checkpoint escapes run root")
    manifest = read_distributed_manifest(checkpoint)
    if manifest.run_kind != "production":
        raise ValueError("Validation-only checkpoint cannot enter production ledger")
    if (
        manifest.run_id != record.run_id
        or manifest.update_index != record.update_index
        or manifest.policy_version != record.policy_version
        or manifest.source_rollout_manifest_sha256 != record.source_manifest_sha256
    ):
        raise ValueError("Ledger identity differs from checkpoint")
    if file_sha256(checkpoint / MANIFEST_PATH) != record.checkpoint_manifest_sha256:
        raise ValueError("Ledger checkpoint manifest digest differs")
    verify_checkpoint_files(checkpoint, manifest)


def read_commits(root: str | Path) -> tuple[CommitRecord, ...]:
    """Validate the entire contiguous chain and every referenced checkpoint."""

    root_path = Path(root)
    directory = root_path / "commits"
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Commit ledger directory is not a regular directory")
    records: list[CommitRecord] = []
    consumed: set[str] = set()
    files = sorted(directory.iterdir())
    for path in files:
        if path.is_symlink() or not path.is_file() or _RECORD_NAME.fullmatch(path.name) is None:
            raise ValueError(f"Unexpected ledger entry: {path.name}")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(raw, dict):
            raise ValueError("Ledger record must be a JSON object")
        record = CommitRecord.from_dict(raw)
        expected_index = len(records) + 1
        if record.update_index != expected_index or path.name != f"update-{expected_index:06d}.json":
            raise ValueError("Ledger updates must be contiguous and correctly named")
        if record.previous_record_sha256 != (records[-1].sha256 if records else None):
            raise ValueError("Ledger hash chain is broken")
        if records and record.run_id != records[0].run_id:
            raise ValueError("Ledger records mix run IDs")
        if record.source_manifest_sha256 in consumed:
            raise ValueError("Rollout batch was consumed more than once")
        _verify_checkpoint(root_path, record)
        consumed.add(record.source_manifest_sha256)
        records.append(record)
    return tuple(records)


def publish_commit(root: str | Path, record: CommitRecord) -> Path:
    """Publish one checkpoint-backed record without overwriting a predecessor.

    This function assumes a single coordinator writer.  Concurrent writers
    can race; the exclusive hard link prevents an overwrite but does not turn
    this file protocol into distributed consensus.
    """

    record.validate()
    root_path = Path(root)
    records = read_commits(root_path)
    if record.update_index != len(records) + 1:
        raise ValueError("New commit does not immediately follow the ledger head")
    if record.previous_record_sha256 != (records[-1].sha256 if records else None):
        raise ValueError("New commit predecessor differs from ledger head")
    if records and record.run_id != records[0].run_id:
        raise ValueError("New commit uses a different run ID")
    if any(item.source_manifest_sha256 == record.source_manifest_sha256 for item in records):
        raise ValueError("Rollout batch has already been consumed")
    _verify_checkpoint(root_path, record)
    directory = root_path / "commits"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Commit ledger directory cannot be a symlink")
    target = directory / f"update-{record.update_index:06d}.json"
    temporary = directory / f".{target.name}.{os.getpid()}.pending"
    payload = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target
