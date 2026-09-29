"""Crash-safe manifests and artifacts for frozen-policy pilot collection."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_play_grpo.rollouts.schema import (
    MatchRecord,
    read_matches_jsonl,
    write_matches_jsonl,
)


PILOT_SCHEMA_VERSION = 1
_ADDITIVE_SUMMARY_FIELDS = frozenset(
    {"fractional_results_by_seat", "mean_turns", "wins_by_seat"}
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data independently of whitespace and key order."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: str | Path) -> str:
    """Hash relative file names and contents for a persisted adapter directory."""

    root = Path(path)
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Cannot hash empty directory: {root}")
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def pilot_match_relative_path(index: int) -> str:
    if index < 0:
        raise ValueError("Pilot match index must be non-negative")
    return f"matches/game-{index:06d}.jsonl"


def pilot_game_id(policy_version: str, index: int, seed: int) -> str:
    return f"{policy_version}-game-{index:06d}-seed-{seed}"


@dataclass(frozen=True)
class PilotMatchEntry:
    index: int
    game_id: str
    seed: int
    path: str
    sha256: str
    policy_version: str
    turns: int
    owned_tokens: int
    final_results: tuple[float, float, float, float]
    termination_reason: str
    max_abs_log_prob_error: float
    illegal_action_substitutions: int = 0

    def validate(
        self,
        *,
        policy_version: str,
        base_seed: int,
        replay_tolerance: float,
    ) -> None:
        expected_seed = base_seed + self.index
        expected_path = pilot_match_relative_path(self.index)
        expected_game_id = pilot_game_id(policy_version, self.index, expected_seed)
        if self.index < 0:
            raise ValueError("Pilot match index must be non-negative")
        if self.seed != expected_seed:
            raise ValueError(
                f"Pilot match {self.index} has seed {self.seed}, expected {expected_seed}"
            )
        if self.path != expected_path:
            raise ValueError(
                f"Pilot match {self.index} has path {self.path!r}, expected {expected_path!r}"
            )
        if self.policy_version != policy_version:
            raise ValueError("Pilot manifest contains mixed policy versions")
        if self.game_id != expected_game_id:
            raise ValueError(
                f"Pilot match {self.index} has game_id {self.game_id!r}, "
                f"expected {expected_game_id!r}"
            )
        if not _is_sha256(self.sha256):
            raise ValueError("Pilot match is missing a SHA-256 artifact digest")
        if self.turns <= 0 or self.owned_tokens <= 0:
            raise ValueError("Pilot match must contain turns and model-owned tokens")
        if len(self.final_results) != 4:
            raise ValueError("Pilot match must contain four final results")
        if not math.isfinite(self.max_abs_log_prob_error) or self.max_abs_log_prob_error < 0:
            raise ValueError("Pilot match replay error must be finite and non-negative")
        if self.max_abs_log_prob_error > replay_tolerance:
            raise ValueError(
                f"Pilot match replay error {self.max_abs_log_prob_error:.6g} exceeds "
                f"{replay_tolerance:.6g}"
            )
        if self.illegal_action_substitutions != 0:
            raise ValueError("Pilot collection forbids illegal-action substitution")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotMatchEntry":
        return cls(
            index=int(value["index"]),
            game_id=str(value["game_id"]),
            seed=int(value["seed"]),
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            policy_version=str(value["policy_version"]),
            turns=int(value["turns"]),
            owned_tokens=int(value["owned_tokens"]),
            final_results=tuple(map(float, value["final_results"])),  # type: ignore[arg-type]
            termination_reason=str(value["termination_reason"]),
            max_abs_log_prob_error=float(value["max_abs_log_prob_error"]),
            illegal_action_substitutions=int(value.get("illegal_action_substitutions", 0)),
        )


@dataclass
class PilotManifest:
    config: dict[str, Any]
    config_sha256: str
    adapter_sha256: str
    policy_version: str
    base_seed: int
    target_games: int
    replay_tolerance: float
    matches: list[PilotMatchEntry] = field(default_factory=list)
    schema_version: int = PILOT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        config: Mapping[str, Any],
        adapter_sha256: str,
        policy_version: str,
        base_seed: int,
        target_games: int,
        replay_tolerance: float,
    ) -> "PilotManifest":
        normalized_config = json.loads(json.dumps(config, sort_keys=True))
        manifest = cls(
            config=normalized_config,
            config_sha256=canonical_sha256(normalized_config),
            adapter_sha256=adapter_sha256,
            policy_version=policy_version,
            base_seed=base_seed,
            target_games=target_games,
            replay_tolerance=replay_tolerance,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != PILOT_SCHEMA_VERSION:
            raise ValueError("Unsupported pilot manifest schema")
        if canonical_sha256(self.config) != self.config_sha256:
            raise ValueError("Pilot manifest configuration digest does not match its config")
        if not _is_sha256(self.config_sha256):
            raise ValueError("Pilot manifest is missing a configuration SHA-256 digest")
        if not _is_sha256(self.adapter_sha256):
            raise ValueError("Pilot manifest is missing an adapter SHA-256 digest")
        if not self.policy_version:
            raise ValueError("Pilot manifest is missing its policy version")
        if self.target_games <= 0:
            raise ValueError("Pilot target_games must be positive")
        if not math.isfinite(self.replay_tolerance) or self.replay_tolerance <= 0:
            raise ValueError("Pilot replay_tolerance must be positive")
        if self.target_games < len(self.matches):
            raise ValueError("Pilot target cannot be below its completed match count")
        for expected_index, entry in enumerate(self.matches):
            if entry.index != expected_index:
                raise ValueError("Pilot match entries must be contiguous from index zero")
            entry.validate(
                policy_version=self.policy_version,
                base_seed=self.base_seed,
                replay_tolerance=self.replay_tolerance,
            )

    def extend_target(self, target_games: int) -> None:
        if target_games < self.target_games:
            raise ValueError(
                f"Pilot target cannot shrink from {self.target_games} to {target_games}"
            )
        self.target_games = target_games
        self.validate()

    def append(self, entry: PilotMatchEntry) -> None:
        if entry.index != len(self.matches):
            raise ValueError(
                f"Expected pilot match index {len(self.matches)}, got {entry.index}"
            )
        entry.validate(
            policy_version=self.policy_version,
            base_seed=self.base_seed,
            replay_tolerance=self.replay_tolerance,
        )
        self.matches.append(entry)
        self.validate()

    def summary(self) -> dict[str, Any]:
        termination_counts: dict[str, int] = {}
        fractional_results = [0.0, 0.0, 0.0, 0.0]
        wins = [0, 0, 0, 0]
        for entry in self.matches:
            termination_counts[entry.termination_reason] = (
                termination_counts.get(entry.termination_reason, 0) + 1
            )
            for seat, result in enumerate(entry.final_results):
                fractional_results[seat] += result
                if abs(result - 1.0) <= 1e-9:
                    wins[seat] += 1
        completed = len(self.matches)
        return {
            "completed_games": completed,
            "draw_games": sum(
                all(abs(result - 0.25) <= 1e-9 for result in entry.final_results)
                for entry in self.matches
            ),
            "fractional_results_by_seat": fractional_results,
            "illegal_action_substitutions": sum(
                entry.illegal_action_substitutions for entry in self.matches
            ),
            "max_abs_log_prob_error": max(
                (entry.max_abs_log_prob_error for entry in self.matches), default=0.0
            ),
            "mean_turns": (
                sum(entry.turns for entry in self.matches) / completed
                if completed
                else 0.0
            ),
            "owned_tokens": sum(entry.owned_tokens for entry in self.matches),
            "policy_versions": sorted({entry.policy_version for entry in self.matches}),
            "status": (
                "complete" if len(self.matches) == self.target_games else "incomplete"
            ),
            "target_games": self.target_games,
            "termination_counts": termination_counts,
            "turns": sum(entry.turns for entry in self.matches),
            "wins_by_seat": wins,
        }

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "config": self.config,
            "config_sha256": self.config_sha256,
            "adapter_sha256": self.adapter_sha256,
            "policy_version": self.policy_version,
            "base_seed": self.base_seed,
            "target_games": self.target_games,
            "replay_tolerance": self.replay_tolerance,
            "matches": [asdict(entry) for entry in self.matches],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotManifest":
        manifest = cls(
            schema_version=int(value.get("schema_version", 0)),
            config=dict(value["config"]),
            config_sha256=str(value["config_sha256"]),
            adapter_sha256=str(value["adapter_sha256"]),
            policy_version=str(value["policy_version"]),
            base_seed=int(value["base_seed"]),
            target_games=int(value["target_games"]),
            replay_tolerance=float(value["replay_tolerance"]),
            matches=[PilotMatchEntry.from_dict(item) for item in value.get("matches", ())],
        )
        manifest.validate()
        recorded_summary = value.get("summary")
        if recorded_summary is not None:
            if not isinstance(recorded_summary, Mapping):
                raise ValueError("Pilot manifest summary must be a mapping")
            recorded = dict(recorded_summary)
            expected = manifest.summary()
            extra = set(recorded) - set(expected)
            missing = set(expected) - set(recorded)
            if extra or missing - _ADDITIVE_SUMMARY_FIELDS or any(
                recorded[key] != expected[key] for key in recorded
            ):
                raise ValueError("Pilot manifest summary does not match its game entries")
        return manifest


def write_pilot_manifest(root: str | Path, manifest: PilotManifest) -> Path:
    root_path = Path(root)
    target = root_path / "manifest.json"
    root_path.mkdir(parents=True, exist_ok=True)
    temporary = root_path / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def read_pilot_manifest(root: str | Path) -> PilotManifest:
    source = Path(root) / "manifest.json"
    return PilotManifest.from_dict(json.loads(source.read_text(encoding="utf-8")))


def save_initial_adapter(model: Any, destination: str | Path) -> str:
    """Atomically persist the exact adapter defining a pilot's frozen policy."""

    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(f"Refusing to overwrite adapter: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise TypeError("Pilot collection requires a PEFT model with save_pretrained")
    with tempfile.TemporaryDirectory(
        dir=destination_path.parent,
        prefix=".policy-adapter-incomplete-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        model.save_pretrained(temporary, safe_serialization=True)
        digest = directory_sha256(temporary)
        temporary.replace(destination_path)
    return digest


def restore_initial_adapter(model: Any, source: str | Path, expected_sha256: str) -> None:
    source_path = Path(source)
    if directory_sha256(source_path) != expected_sha256:
        raise ValueError("Frozen pilot adapter digest does not match the manifest")
    if not hasattr(model, "load_adapter"):
        raise TypeError("Pilot resume requires a PEFT model with load_adapter")
    result = model.load_adapter(
        source_path,
        adapter_name="default",
        is_trainable=True,
        torch_device=str(next(model.parameters()).device),
        local_files_only=True,
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Frozen pilot adapter keys differ from the active model: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def initialize_pilot_root(
    root: str | Path,
    *,
    model: Any,
    config: Mapping[str, Any],
    model_revision: str,
    base_seed: int,
    target_games: int,
    replay_tolerance: float,
) -> PilotManifest:
    """Create adapter, manifest, and match directory as one directory rename."""

    root_path = Path(root)
    if root_path.exists():
        raise FileExistsError(
            f"Pilot output already exists without a usable manifest: {root_path}"
        )
    root_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=root_path.parent,
        prefix=f".{root_path.name}-initializing-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        (temporary / "matches").mkdir()
        adapter_sha256 = save_initial_adapter(model, temporary / "policy_adapter")
        policy_version = f"pilot:{model_revision[:12]}:{adapter_sha256[:12]}"
        manifest = PilotManifest.create(
            config=config,
            adapter_sha256=adapter_sha256,
            policy_version=policy_version,
            base_seed=base_seed,
            target_games=target_games,
            replay_tolerance=replay_tolerance,
        )
        write_pilot_manifest(temporary, manifest)
        temporary.replace(root_path)
    return manifest


def validate_manifest_request(
    manifest: PilotManifest,
    *,
    config: Mapping[str, Any],
    base_seed: int,
    replay_tolerance: float,
) -> None:
    if canonical_sha256(config) != manifest.config_sha256:
        raise ValueError("Active configuration differs from the frozen pilot configuration")
    if base_seed != manifest.base_seed:
        raise ValueError(
            f"Pilot base seed is {manifest.base_seed}; requested {base_seed}"
        )
    if replay_tolerance != manifest.replay_tolerance:
        raise ValueError(
            f"Pilot replay tolerance is {manifest.replay_tolerance}; "
            f"requested {replay_tolerance}"
        )
    model_config = manifest.config.get("model")
    if not isinstance(model_config, Mapping) or not model_config.get("revision"):
        raise ValueError("Frozen pilot configuration has no model revision")
    expected_policy_version = (
        f"pilot:{str(model_config['revision'])[:12]}:{manifest.adapter_sha256[:12]}"
    )
    if manifest.policy_version != expected_policy_version:
        raise ValueError("Pilot policy version does not match its model and adapter digests")


def validate_match_identity(
    match: MatchRecord,
    manifest: PilotManifest,
    index: int,
) -> None:
    expected_seed = manifest.base_seed + index
    expected_game_id = pilot_game_id(manifest.policy_version, index, expected_seed)
    if match.game_id != expected_game_id:
        raise ValueError(
            f"Pilot match {index} game_id is {match.game_id!r}, expected {expected_game_id!r}"
        )
    if match.seed != expected_seed:
        raise ValueError(
            f"Pilot match {index} seed is {match.seed}, expected {expected_seed}"
        )
    if match.policy_version != manifest.policy_version:
        raise ValueError("Pilot match uses a different policy version")
    if canonical_sha256(match.environment_config) != canonical_sha256(
        manifest.config["environment"]
    ):
        raise ValueError("Pilot match environment differs from the frozen configuration")


def read_and_replay_pilot_match(
    root: str | Path,
    manifest: PilotManifest,
    index: int,
) -> tuple[MatchRecord, str]:
    """Validate one persisted record and deterministically replay its environment."""

    relative = pilot_match_relative_path(index)
    path = Path(root) / relative
    matches = read_matches_jsonl(path)
    if len(matches) != 1:
        raise ValueError(f"Pilot artifact {relative} must contain exactly one match")
    match = matches[0]
    validate_match_identity(match, manifest, index)
    from self_play_grpo.rollouts.replay import replay_match

    replay_match(match)
    return match, file_sha256(path)


def validate_registered_matches(root: str | Path, manifest: PilotManifest) -> None:
    for entry in manifest.matches:
        match, digest = read_and_replay_pilot_match(root, manifest, entry.index)
        if digest != entry.sha256:
            raise ValueError(f"Pilot artifact digest changed: {entry.path}")
        reconstructed = make_match_entry(
            index=entry.index,
            match=match,
            path=Path(root) / entry.path,
            max_abs_log_prob_error=entry.max_abs_log_prob_error,
        )
        if reconstructed != entry:
            raise ValueError(f"Pilot manifest metadata changed: {entry.path}")


def discover_match_indices(root: str | Path) -> list[int]:
    match_root = Path(root) / "matches"
    if not match_root.is_dir():
        raise FileNotFoundError(f"Pilot match directory is missing: {match_root}")
    indices: list[int] = []
    for path in sorted(match_root.glob("*.jsonl")):
        name = path.name
        if not (name.startswith("game-") and name.endswith(".jsonl")):
            raise ValueError(f"Unexpected pilot match artifact name: {name}")
        digits = name[len("game-") : -len(".jsonl")]
        if len(digits) != 6 or not digits.isdigit():
            raise ValueError(f"Unexpected pilot match artifact name: {name}")
        indices.append(int(digits))
    if indices != list(range(len(indices))):
        raise ValueError(f"Pilot match artifacts are not contiguous: {indices}")
    return indices


def write_pilot_match(root: str | Path, index: int, match: MatchRecord) -> Path:
    path = Path(root) / pilot_match_relative_path(index)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite pilot match: {path}")
    write_matches_jsonl(path, [match])
    return path


def make_match_entry(
    *,
    index: int,
    match: MatchRecord,
    path: str | Path,
    max_abs_log_prob_error: float,
) -> PilotMatchEntry:
    samples = [
        turn.policy_sample
        for turn in match.turns
        if turn.policy_sample.completion_token_ids
    ]
    return PilotMatchEntry(
        index=index,
        game_id=match.game_id,
        seed=match.seed,
        path=pilot_match_relative_path(index),
        sha256=file_sha256(path),
        policy_version=match.policy_version,
        turns=len(match.turns),
        owned_tokens=sum(len(sample.completion_token_ids) for sample in samples),
        final_results=match.final_results,
        termination_reason=match.termination_reason,
        max_abs_log_prob_error=max_abs_log_prob_error,
        illegal_action_substitutions=0,
    )


def model_samples(match: MatchRecord) -> Sequence[Any]:
    return tuple(
        turn.policy_sample
        for turn in match.turns
        if turn.policy_sample.completion_token_ids
    )
