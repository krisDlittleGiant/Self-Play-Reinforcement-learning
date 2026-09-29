"""Read-only D4 hardware-evidence gate for a future D5 launcher.

This cannot manufacture D4 success: both two- and four-trainer gate reports,
their rank evidence, and the referenced format-3 checkpoints must already
exist.  Validation-only D4 checkpoints prove continuation but are never used
as production starting weights.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from self_play_grpo.rollouts.pilot import file_sha256
from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.distributed_checkpoint import (
    MANIFEST_PATH,
    read_distributed_manifest,
    verify_checkpoint_files,
)
from self_play_grpo.training.roles import role_layout_from_modules


_EQUALITY_FIELDS = (
    "cpu_rng_equal", "hpu_rng_equal", "parameters_equal",
    "optimizer_equal", "metrics_equal",
)


@dataclass(frozen=True)
class D4GateReceipt:
    world_size: int
    module_ids: tuple[int, ...]
    checkpoint_manifest_sha256: str
    source_rollout: str
    source_checkpoint: str
    summary_path: str


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"D4 gate report is not a JSON object: {path}")
    return data


def verify_d4_gate(summary_path: str | Path, *, world_size: int) -> D4GateReceipt:
    """Verify one successful, complete D4 gate without loading an HPU model."""

    if world_size not in (2, 4):
        raise ValueError("D4 acceptance requires a two- or four-rank gate")
    path = Path(summary_path)
    summary = _read_json(path)
    if (
        summary.get("status") != "ok"
        or summary.get("validation_only") is not True
        or summary.get("world_size") != world_size
    ):
        raise ValueError("D4 summary did not pass the requested world size")
    runtime = summary.get("runtime_contract")
    if not isinstance(runtime, dict) or runtime.get("status") != "ok" or runtime.get("world_size") != world_size:
        raise ValueError("D4 runtime collective contract is missing")
    reports = summary.get("rank_reports")
    if not isinstance(reports, list) or len(reports) != world_size:
        raise ValueError("D4 rank reports are incomplete")
    module_ids: list[int] = []
    for rank, row in enumerate(reports):
        if not isinstance(row, dict) or row.get("rank") != rank or row.get("status") != "ok":
            raise ValueError("D4 rank evidence is missing or out of order")
        if any(row.get(field) is not True for field in _EQUALITY_FIELDS):
            raise ValueError(f"D4 rank {rank} continuation equality failed")
        if row.get("checkpoint_manifest_sha256") != summary.get("checkpoint_manifest_sha256"):
            raise ValueError("D4 rank checkpoint digest differs from summary")
        try:
            module_ids.append(int(row["module_id"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("D4 rank physical module is missing") from exc
    if len(set(module_ids)) != world_size or any(module < 0 for module in module_ids):
        raise ValueError("D4 rank physical modules are not distinct")
    if runtime.get("module_ids") != [str(module) for module in module_ids]:
        raise ValueError("D4 runtime module mapping differs from rank reports")
    raw_checkpoint = summary.get("checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError("D4 summary has no checkpoint path")
    checkpoint = Path(raw_checkpoint)
    manifest = read_distributed_manifest(checkpoint)
    verify_checkpoint_files(checkpoint, manifest)
    if (
        manifest.run_kind != "validation"
        or manifest.trainer_world_size != world_size
        or manifest.run_id != summary.get("run_id")
        or tuple(rank.module_id for rank in manifest.ranks)
        != tuple(str(module) for module in module_ids)
    ):
        raise ValueError("D4 checkpoint identity differs from summary and rank reports")
    digest = file_sha256(checkpoint / MANIFEST_PATH)
    if digest != summary.get("checkpoint_manifest_sha256"):
        raise ValueError("D4 checkpoint manifest bytes changed")
    source_rollout = summary.get("source_rollout")
    source_checkpoint = summary.get("source_checkpoint")
    if not isinstance(source_rollout, str) or not source_rollout:
        raise ValueError("D4 source rollout is missing")
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise ValueError("D4 source checkpoint is missing")
    return D4GateReceipt(
        world_size=world_size, module_ids=tuple(module_ids),
        checkpoint_manifest_sha256=digest,
        source_rollout=source_rollout,
        source_checkpoint=source_checkpoint,
        summary_path=str(path),
    )


def verify_d4_acceptance(
    two_rank_summary: str | Path,
    four_rank_summary: str | Path,
    *,
    layout: RoleLayout,
) -> tuple[D4GateReceipt, D4GateReceipt]:
    """Require both D4 gates and the intended final trainer-module layout."""

    two = verify_d4_gate(two_rank_summary, world_size=2)
    four = verify_d4_gate(four_rank_summary, world_size=4)
    if four.module_ids != layout.trainer_modules:
        raise ValueError("Four-rank D4 gate did not use the D5 trainer modules")
    if (
        Path(two.source_rollout).resolve() != Path(four.source_rollout).resolve()
        or Path(two.source_checkpoint).resolve() != Path(four.source_checkpoint).resolve()
    ):
        raise ValueError("Two- and four-rank D4 gates used different source artifacts")
    return two, four


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only D4 acceptance preflight")
    parser.add_argument("--two-rank-summary", required=True, type=Path)
    parser.add_argument("--four-rank-summary", required=True, type=Path)
    parser.add_argument("--modules", required=True,
                        help="Eight explicit module IDs: four rollout, then four trainer")
    args = parser.parse_args(argv)
    layout = role_layout_from_modules(args.modules)
    two, four = verify_d4_acceptance(
        args.two_rank_summary, args.four_rank_summary, layout=layout
    )
    print(json.dumps({
        "status": "passed", "scope": "read_only_d4_evidence",
        "two_rank_summary": two.summary_path,
        "four_rank_summary": four.summary_path,
        "trainer_modules": list(four.module_ids),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
