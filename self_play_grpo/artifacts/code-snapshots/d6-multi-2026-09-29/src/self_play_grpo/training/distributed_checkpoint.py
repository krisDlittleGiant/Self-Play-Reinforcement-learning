"""Format-3 distributed checkpoint identity and integrity contracts.

This module is deliberately model-free. It validates a checkpoint completely
before a future loader mutates the adapter, optimizer, or rank RNG state.
Format-2 single-process checkpoints remain owned by ``SynchronousTrainer``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


CHECKPOINT_FORMAT = 3
MANIFEST_PATH = "distributed/manifest.json"
SHARDING_RULE = "equal_contiguous_complete_matches_v1"
ABSENT_TRAINER_STATE = ("numpy_rng", "python_rng", "sampler", "scaler", "scheduler")
_REQUIRED_FILES = frozenset(
    {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "optimizer_state.pt",
        "torch_rng_state.pt",
        "trainer_state.json",
    }
)
_RUNTIME_KEYS = frozenset({"python", "torch", "transformers", "peft", "habana"})


def _keys(value: Mapping[str, Any], required: set[str], name: str) -> None:
    if set(value) != required:
        raise ValueError(
            f"{name} keys differ: missing={sorted(required - set(value))}, "
            f"unexpected={sorted(set(value) - required)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: Any, name: str) -> str:
    path = _text(value, name)
    if (
        "\\" in path
        or "\x00" in path
        or ":" in path
        or path.startswith("/")
        or any(part in ("", ".", "..") for part in path.split("/"))
        or PurePosixPath(path).as_posix() != path
    ):
        raise ValueError(f"{name} is not a safe checkpoint-relative path: {path!r}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    byte_count: int
    sha256: str

    def validate(self) -> None:
        _relative_path(self.path, "checkpoint file path")
        if self.path == MANIFEST_PATH:
            raise ValueError("The manifest cannot hash itself")
        _integer(self.byte_count, "checkpoint file byte_count", minimum=1)
        _digest(self.sha256, "checkpoint file sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "byte_count": self.byte_count, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointFile":
        _keys(value, {"path", "byte_count", "sha256"}, "checkpoint file")
        result = cls(value["path"], value["byte_count"], value["sha256"])
        result.validate()
        return result


@dataclass(frozen=True)
class TrainerRankState:
    rank: int
    local_rank: int
    module_id: str
    device: str
    rng_path: str
    match_indices: tuple[int, ...]
    match_ids: tuple[str, ...]
    match_sha256: tuple[str, ...]

    def validate(self) -> None:
        _integer(self.rank, "rank")
        _integer(self.local_rank, "local_rank")
        if self.local_rank != self.rank:
            raise ValueError("Single-node trainer requires local_rank == rank")
        _text(self.module_id, "module_id")
        if self.device != f"hpu:{self.local_rank}":
            raise ValueError("Rank device differs from its logical HPU binding")
        if self.rng_path != f"distributed/rng/rank-{self.rank:03d}.pt":
            raise ValueError("Rank RNG filename differs from its rank")
        if not self.match_indices or not (
            len(self.match_indices) == len(self.match_ids) == len(self.match_sha256)
        ):
            raise ValueError("Rank match indices, IDs, and digests must align")
        for index in self.match_indices:
            _integer(index, "match index")
        for match_id in self.match_ids:
            _text(match_id, "match ID")
        if len(set(self.match_ids)) != len(self.match_ids):
            raise ValueError("Rank match IDs must be unique")
        for digest in self.match_sha256:
            _digest(digest, "match sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "module_id": self.module_id,
            "device": self.device,
            "rng_path": self.rng_path,
            "match_indices": list(self.match_indices),
            "match_ids": list(self.match_ids),
            "match_sha256": list(self.match_sha256),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainerRankState":
        _keys(
            value,
            {
                "rank", "local_rank", "module_id", "device", "rng_path",
                "match_indices", "match_ids", "match_sha256",
            },
            "trainer rank",
        )
        for name in ("match_indices", "match_ids", "match_sha256"):
            if not isinstance(value[name], list):
                raise ValueError(f"{name} must be a JSON array")
        result = cls(
            rank=value["rank"],
            local_rank=value["local_rank"],
            module_id=value["module_id"],
            device=value["device"],
            rng_path=value["rng_path"],
            match_indices=tuple(value["match_indices"]),
            match_ids=tuple(value["match_ids"]),
            match_sha256=tuple(value["match_sha256"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DistributedCheckpointManifest:
    run_kind: str
    run_id: str
    policy_version: str
    update_index: int
    config_sha256: str
    model_id: str
    model_revision: str
    adapter_schema_sha256: str
    optimizer_schema_sha256: str
    tokenizer_sha256: str
    grammar_sha256: str
    code_identity: str
    runtime_identity: tuple[tuple[str, str], ...]
    attention_backend: str
    dtype: str
    backend: str
    trainer_world_size: int
    source_rollout_manifest_sha256: str
    ranks: tuple[TrainerRankState, ...]
    files: tuple[CheckpointFile, ...]
    absent_state: tuple[str, ...] = ABSENT_TRAINER_STATE
    sharding_rule: str = SHARDING_RULE
    checkpoint_format: int = CHECKPOINT_FORMAT

    def validate(self) -> None:
        if self.checkpoint_format != CHECKPOINT_FORMAT:
            raise ValueError("Unsupported distributed checkpoint format")
        if self.run_kind not in ("validation", "production"):
            raise ValueError("run_kind must be validation or production")
        for name in (
            "run_id", "policy_version", "model_id", "model_revision",
            "code_identity", "attention_backend", "dtype", "backend",
        ):
            _text(getattr(self, name), name)
        _integer(self.update_index, "update_index")
        if self.policy_version != f"policy-{self.update_index:06d}":
            raise ValueError("Policy version differs from update_index")
        _integer(self.trainer_world_size, "trainer_world_size", minimum=2)
        for name in (
            "config_sha256", "adapter_schema_sha256", "optimizer_schema_sha256",
            "tokenizer_sha256", "grammar_sha256", "source_rollout_manifest_sha256",
        ):
            _digest(getattr(self, name), name)
        runtime = dict(self.runtime_identity)
        if len(runtime) != len(self.runtime_identity) or set(runtime) != _RUNTIME_KEYS:
            raise ValueError("runtime_identity keys or uniqueness differ")
        for name, value in runtime.items():
            _text(value, f"runtime_identity.{name}")
        if self.sharding_rule != SHARDING_RULE:
            raise ValueError("Unsupported trainer match-sharding rule")
        if self.absent_state != ABSENT_TRAINER_STATE:
            raise ValueError("Trainer absent-state declaration differs")
        if len(self.ranks) != self.trainer_world_size:
            raise ValueError("Rank record count differs from trainer world size")
        if tuple(rank.rank for rank in self.ranks) != tuple(range(self.trainer_world_size)):
            raise ValueError("Rank records must be unique and sorted from zero")
        per_rank = len(self.ranks[0].match_indices)
        seen_ids: set[str] = set()
        for rank in self.ranks:
            rank.validate()
            expected = tuple(range(rank.rank * per_rank, (rank.rank + 1) * per_rank))
            if rank.match_indices != expected:
                raise ValueError(f"Rank {rank.rank} match shard is not equal contiguous")
            if seen_ids.intersection(rank.match_ids):
                raise ValueError("Match IDs are duplicated across trainer ranks")
            seen_ids.update(rank.match_ids)
        paths = tuple(file.path for file in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("Checkpoint files must have unique sorted paths")
        for file in self.files:
            file.validate()
        required = _REQUIRED_FILES | {rank.rng_path for rank in self.ranks}
        if not required.issubset(paths):
            raise ValueError(f"Checkpoint inventory is missing {sorted(required - set(paths))}")
        rng_files = {path for path in paths if path.startswith("distributed/rng/")}
        if rng_files != {rank.rng_path for rank in self.ranks}:
            raise ValueError("Checkpoint inventory has unexpected rank RNG files")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "checkpoint_format": self.checkpoint_format,
            "run_kind": self.run_kind,
            "run_id": self.run_id,
            "policy_version": self.policy_version,
            "update_index": self.update_index,
            "config_sha256": self.config_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_schema_sha256": self.adapter_schema_sha256,
            "optimizer_schema_sha256": self.optimizer_schema_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "grammar_sha256": self.grammar_sha256,
            "code_identity": self.code_identity,
            "runtime_identity": dict(self.runtime_identity),
            "attention_backend": self.attention_backend,
            "dtype": self.dtype,
            "backend": self.backend,
            "trainer_world_size": self.trainer_world_size,
            "source_rollout_manifest_sha256": self.source_rollout_manifest_sha256,
            "ranks": [rank.to_dict() for rank in self.ranks],
            "files": [file.to_dict() for file in self.files],
            "absent_state": list(self.absent_state),
            "sharding_rule": self.sharding_rule,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DistributedCheckpointManifest":
        required = set(cls.__dataclass_fields__)
        _keys(value, required, "distributed checkpoint manifest")
        if not isinstance(value["ranks"], list) or not isinstance(value["files"], list):
            raise ValueError("ranks and files must be JSON arrays")
        if not isinstance(value["runtime_identity"], Mapping):
            raise ValueError("runtime_identity must be a JSON object")
        if not isinstance(value["absent_state"], list):
            raise ValueError("absent_state must be a JSON array")
        result = cls(
            **{
                name: value[name]
                for name in required - {"runtime_identity", "ranks", "files", "absent_state"}
            },
            runtime_identity=tuple(sorted(value["runtime_identity"].items())),
            ranks=tuple(TrainerRankState.from_dict(item) for item in value["ranks"]),
            files=tuple(CheckpointFile.from_dict(item) for item in value["files"]),
            absent_state=tuple(value["absent_state"]),
        )
        result.validate()
        return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in checkpoint manifest: {key}")
        result[key] = value
    return result


def read_distributed_manifest(root: str | Path) -> DistributedCheckpointManifest:
    source = Path(root) / MANIFEST_PATH
    data = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(data, dict):
        raise ValueError("Distributed checkpoint manifest must be a JSON object")
    return DistributedCheckpointManifest.from_dict(data)


def verify_checkpoint_files(root: str | Path, manifest: DistributedCheckpointManifest) -> None:
    """Reject missing, extra, changed, or symlinked checkpoint files."""

    manifest.validate()
    original_root = Path(root)
    if original_root.is_symlink():
        raise ValueError("Checkpoint root must not be a symlink")
    resolved_root = original_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("Checkpoint root is not a directory")
    observed: set[str] = set()
    for candidate in resolved_root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"Checkpoint contains a symlink: {candidate}")
        if candidate.is_file():
            observed.add(candidate.relative_to(resolved_root).as_posix())
        elif not candidate.is_dir():
            raise ValueError(f"Checkpoint contains a non-file entry: {candidate}")
    expected = {file.path for file in manifest.files} | {MANIFEST_PATH}
    if observed != expected:
        raise ValueError(
            f"Checkpoint file inventory differs: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    for file in manifest.files:
        candidate = (resolved_root / file.path).resolve(strict=True)
        if not candidate.is_relative_to(resolved_root):
            raise ValueError(f"Checkpoint file escapes its root: {file.path}")
        if candidate.stat().st_size != file.byte_count:
            raise ValueError(f"Checkpoint file size differs: {file.path}")
        if file_sha256(candidate) != file.sha256:
            raise ValueError(f"Checkpoint file digest differs: {file.path}")


def validate_resume_identity(
    manifest: DistributedCheckpointManifest,
    *,
    expected: Mapping[str, Any],
    allow_validation: bool = False,
) -> None:
    """Fail before state mutation on every identity supplied by the caller."""

    manifest.validate()
    if manifest.run_kind == "validation" and not allow_validation:
        raise ValueError("Refusing to resume a validation-only distributed checkpoint")
    required = {
        "run_id", "policy_version", "update_index", "config_sha256", "model_id",
        "model_revision", "adapter_schema_sha256", "optimizer_schema_sha256",
        "tokenizer_sha256", "grammar_sha256", "code_identity", "runtime_identity",
        "attention_backend", "dtype", "backend", "trainer_world_size",
        "source_rollout_manifest_sha256", "sharding_rule", "rank_bindings",
        "match_shards",
    }
    _keys(expected, required, "resume expectation")
    actual = manifest.to_dict()
    for key in required - {"rank_bindings", "match_shards"}:
        if actual[key] != expected[key]:
            raise ValueError(f"Distributed checkpoint {key} differs from active run")
    bindings = [
        {"rank": rank.rank, "local_rank": rank.local_rank, "module_id": rank.module_id,
         "device": rank.device}
        for rank in manifest.ranks
    ]
    shards = [
        {"match_indices": list(rank.match_indices), "match_ids": list(rank.match_ids),
         "match_sha256": list(rank.match_sha256)}
        for rank in manifest.ranks
    ]
    if bindings != expected["rank_bindings"]:
        raise ValueError("Distributed checkpoint rank_bindings differ from active run")
    if shards != expected["match_shards"]:
        raise ValueError("Distributed checkpoint match_shards differ from active run")


def validate_rank_rng_record(
    root: str | Path,
    rank: TrainerRankState,
) -> None:
    """Safely load and verify the rank identity carried inside its RNG payload."""

    import torch

    record = torch.load(Path(root) / rank.rng_path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict):
        raise ValueError(f"Rank {rank.rank} RNG record must be a dictionary")
    _keys(record, {"schema_version", "rank", "module_id", "cpu", "hpu", "hpu_api"}, "rank RNG")
    if record["schema_version"] != 1 or record["rank"] != rank.rank:
        raise ValueError(f"Rank RNG payload differs from filename rank {rank.rank}")
    if record["module_id"] != rank.module_id:
        raise ValueError(f"Rank {rank.rank} RNG module binding differs")
    if record["hpu_api"] != "torch.hpu.get_rng_state_all":
        raise ValueError(f"Rank {rank.rank} RNG HPU API differs")
    if not torch.is_tensor(record["cpu"]) or record["cpu"].dtype != torch.uint8:
        raise ValueError(f"Rank {rank.rank} CPU RNG state is not a byte tensor")
    if not isinstance(record["hpu"], (list, tuple)) or not record["hpu"]:
        raise ValueError(f"Rank {rank.rank} HPU RNG state is missing")
    if any(not torch.is_tensor(state) or state.dtype != torch.uint8 for state in record["hpu"]):
        raise ValueError(f"Rank {rank.rank} HPU RNG state contains invalid tensors")


def load_rank_rng_record(root: str | Path, rank: TrainerRankState) -> dict[str, Any]:
    """Load a validated rank-local RNG record without restoring it yet."""

    import torch

    validate_rank_rng_record(root, rank)
    return torch.load(Path(root) / rank.rng_path, map_location="cpu", weights_only=True)


def write_rank_rng_record(root: str | Path, rank: TrainerRankState) -> Path:
    """Stage one rank's CPU and HPU RNG at a synchronized update boundary."""

    import torch

    rank.validate()
    if not hasattr(torch, "hpu") or not hasattr(torch.hpu, "get_rng_state_all"):
        raise RuntimeError("This runtime cannot capture rank HPU RNG state")
    if hasattr(torch.hpu, "is_initialized") and not torch.hpu.is_initialized():
        raise RuntimeError("HPU must be initialized before capturing rank RNG")
    if hasattr(torch.hpu, "synchronize"):
        torch.hpu.synchronize()
    states = torch.hpu.get_rng_state_all()
    if not isinstance(states, (list, tuple)) or not states:
        raise RuntimeError("HPU RNG capture returned no rank state")
    root_path = Path(root)
    target = root_path / rank.rng_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Rank RNG record already exists: {target}")
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary rank RNG record already exists: {temporary}")
    record = {
        "schema_version": 1,
        "rank": rank.rank,
        "module_id": rank.module_id,
        "cpu": torch.get_rng_state(),
        "hpu": list(states),
        "hpu_api": "torch.hpu.get_rng_state_all",
    }
    try:
        torch.save(record, temporary)
        temporary.replace(target)
        validate_rank_rng_record(root_path, rank)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def build_checkpoint_inventory(root: str | Path) -> tuple[CheckpointFile, ...]:
    """Hash every checkpoint file except the manifest to be written last."""

    root_path = Path(root).resolve(strict=True)
    files: list[CheckpointFile] = []
    for candidate in root_path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"Checkpoint contains a symlink: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(root_path).as_posix()
            if relative == MANIFEST_PATH:
                raise ValueError("Checkpoint manifest already exists before inventory")
            files.append(CheckpointFile(relative, candidate.stat().st_size, file_sha256(candidate)))
        elif not candidate.is_dir():
            raise ValueError(f"Checkpoint contains a non-file entry: {candidate}")
    result = tuple(sorted(files, key=lambda file: file.path))
    for file in result:
        file.validate()
    return result
