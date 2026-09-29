"""Rank-sharded artifact contracts for distributed frozen-policy gameplay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_play_grpo.rollouts.pilot import PilotManifest, PilotMatchEntry


DISTRIBUTED_ROLLOUT_SCHEMA_VERSION = 1


def rank_match_indices(
    rank: int,
    world_size: int,
    games_per_rank: int,
) -> tuple[int, ...]:
    """Return the deterministic contiguous global match shard for one rank."""

    if world_size < 2:
        raise ValueError("Distributed rollout requires at least two ranks")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside [0, {world_size})")
    if games_per_rank <= 0:
        raise ValueError("games_per_rank must be positive")
    start = rank * games_per_rank
    return tuple(range(start, start + games_per_rank))


def rank_report_relative_path(rank: int) -> str:
    if rank < 0:
        raise ValueError("rank must be non-negative")
    return f"ranks/rank-{rank:03d}.json"


@dataclass(frozen=True)
class DistributedRankReport:
    rank: int
    local_rank: int
    world_size: int
    games_per_rank: int
    policy_version: str
    adapter_sha256: str
    matches: tuple[PilotMatchEntry, ...]
    schema_version: int = DISTRIBUTED_ROLLOUT_SCHEMA_VERSION

    def validate(self, manifest: PilotManifest) -> None:
        if self.schema_version != DISTRIBUTED_ROLLOUT_SCHEMA_VERSION:
            raise ValueError("Unsupported distributed rollout report schema")
        expected = rank_match_indices(
            self.rank,
            self.world_size,
            self.games_per_rank,
        )
        if self.local_rank != self.rank:
            raise ValueError("Single-node rollout requires local_rank == rank")
        if tuple(entry.index for entry in self.matches) != expected:
            raise ValueError(
                f"Rank {self.rank} match indices differ from expected {expected}"
            )
        if self.policy_version != manifest.policy_version:
            raise ValueError("Distributed rank report has a mixed policy version")
        if self.adapter_sha256 != manifest.adapter_sha256:
            raise ValueError("Distributed rank report has a different adapter")
        for entry in self.matches:
            entry.validate(
                policy_version=manifest.policy_version,
                base_seed=manifest.base_seed,
                replay_tolerance=manifest.replay_tolerance,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "games_per_rank": self.games_per_rank,
            "policy_version": self.policy_version,
            "adapter_sha256": self.adapter_sha256,
            "matches": [asdict(entry) for entry in self.matches],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DistributedRankReport":
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            rank=int(value["rank"]),
            local_rank=int(value["local_rank"]),
            world_size=int(value["world_size"]),
            games_per_rank=int(value["games_per_rank"]),
            policy_version=str(value["policy_version"]),
            adapter_sha256=str(value["adapter_sha256"]),
            matches=tuple(
                PilotMatchEntry.from_dict(item) for item in value.get("matches", ())
            ),
        )


def write_rank_report(
    root: str | Path,
    report: DistributedRankReport,
    manifest: PilotManifest,
) -> Path:
    report.validate(manifest)
    root_path = Path(root)
    target = root_path / rank_report_relative_path(report.rank)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def read_rank_report(
    root: str | Path,
    rank: int,
    manifest: PilotManifest,
) -> DistributedRankReport:
    path = Path(root) / rank_report_relative_path(rank)
    report = DistributedRankReport.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    report.validate(manifest)
    return report


def ordered_report_entries(
    reports: Sequence[DistributedRankReport],
    *,
    world_size: int,
    games_per_rank: int,
) -> tuple[PilotMatchEntry, ...]:
    if len(reports) != world_size:
        raise ValueError(f"Expected {world_size} rank reports, got {len(reports)}")
    by_rank = {report.rank: report for report in reports}
    if set(by_rank) != set(range(world_size)):
        raise ValueError("Distributed rank reports are missing or duplicated")
    for rank, report in by_rank.items():
        if report.world_size != world_size:
            raise ValueError(f"Rank {rank} reports a different world size")
        if report.games_per_rank != games_per_rank:
            raise ValueError(f"Rank {rank} reports a different games_per_rank")
    entries = tuple(
        entry
        for rank in range(world_size)
        for entry in by_rank[rank].matches
    )
    expected_indices = tuple(range(world_size * games_per_rank))
    if tuple(entry.index for entry in entries) != expected_indices:
        raise ValueError("Distributed match entries are not globally contiguous")
    return entries
