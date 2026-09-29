"""CPU-only audit of a production complete-update restart boundary.

This module never loads a model or starts workers.  A committed checkpoint is
authoritative; refresh evidence is a separate gate before new collection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256
from self_play_grpo.training.d6_code_identity import d6_code_identity
from self_play_grpo.training.distributed_checkpoint import read_distributed_manifest
from self_play_grpo.training.ledger import read_commits
from self_play_grpo.training.production_checkpoint import production_code_identity


@dataclass(frozen=True)
class RecoveryPlan:
    status: str
    run_id: str
    committed_update: int
    policy_version: str
    checkpoint: str
    checkpoint_manifest_sha256: str
    source_manifest_sha256: str
    next_update: int
    next_collection_index: int
    next_policy_version: str
    next_base_seed: int
    seed_derivation: str
    consumed_batches: tuple[str, ...]


def _json_file(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked recovery evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Recovery evidence is not a JSON object: {path}")
    return value


def _next_seed(experiment_seed: int, collection_index: int, run_id: str) -> int:
    if type(experiment_seed) is not int or experiment_seed < 0:
        raise ValueError("D6 experiment seed must be a non-negative integer")
    digest = canonical_sha256({
        "namespace": "self_play_grpo/production_collection/v1",
        "run_id": run_id,
        "experiment_seed": experiment_seed,
        "collection_index": collection_index,
    })
    return int(digest[:15], 16)


def audit_recovery(
    root: str | Path, *, config_path: str | Path, experiment_seed: int,
    expected_run_id: str | None = None,
) -> RecoveryPlan:
    """Verify a full ledger head and report whether collection may resume.

    A published checkpoint with absent or incomplete refresh remains committed,
    but returns ``refresh_required``.  Callers must not collect in that state.
    """

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("D6 run root must be a real directory")
    config = load_config(config_path)
    config_digest = canonical_sha256(config.to_dict())
    records = read_commits(root)
    if not records:
        raise ValueError("D6 restart requires at least one committed production update")
    head = records[-1]
    if expected_run_id is not None and head.run_id != expected_run_id:
        raise ValueError("D6 run ID differs from the requested run")
    checkpoint = root / head.checkpoint_path
    manifest = read_distributed_manifest(checkpoint)
    expected_code = (production_code_identity() if head.update_index == 1 else
                     d6_code_identity() if head.update_index == 2 else None)
    if expected_code is None:
        raise ValueError("D6 recovery does not yet support this checkpoint generation")
    if (manifest.config_sha256 != config_digest
            or manifest.model_id != config.model.id
            or manifest.model_revision != config.model.revision
            or manifest.code_identity != expected_code
            or manifest.trainer_world_size != 4):
        raise ValueError("D6 committed checkpoint config, code or topology differs")
    adapter_digest = directory_sha256(checkpoint / "adapter")
    collection_index = head.update_index - 1
    source = root / f"rollout-{collection_index:06d}" / "manifest.json"
    if source.is_symlink() or not source.is_file() or file_sha256(source) != head.source_manifest_sha256:
        raise ValueError("D6 committed batch is absent or differs from checkpoint")

    refresh_dir = root / f"refresh-{head.update_index:06d}"
    refresh_summary_path = refresh_dir / "refresh_summary.json"
    status = "refresh_required"
    if refresh_summary_path.exists() or refresh_summary_path.is_symlink():
        summary = _json_file(refresh_summary_path)
        expected = {
            "status": "refresh_reports_verified",
            "policy_version": head.policy_version,
            "checkpoint_manifest_sha256": head.checkpoint_manifest_sha256,
            "source_manifest_sha256": head.source_manifest_sha256,
            "adapter_sha256": adapter_digest,
            "refresh_ranks": 4,
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError("D6 refresh summary differs from committed checkpoint")
        for rank in range(4):
            report = _json_file(refresh_dir / f"rank-{rank:03d}.json")
            if (report.get("status") != "refresh_verified"
                    or report.get("rank") != rank
                    or report.get("policy_version") != head.policy_version
                    or report.get("checkpoint_manifest_sha256") != head.checkpoint_manifest_sha256
                    or report.get("source_manifest_sha256") != head.source_manifest_sha256
                    or report.get("adapter_sha256") != adapter_digest):
                raise ValueError(f"D6 refresh rank {rank} differs from commit")
        status = "ready_for_next_collection"
    next_collection_index = head.update_index
    return RecoveryPlan(
        status=status, run_id=head.run_id, committed_update=head.update_index,
        policy_version=head.policy_version, checkpoint=head.checkpoint_path,
        checkpoint_manifest_sha256=head.checkpoint_manifest_sha256,
        source_manifest_sha256=head.source_manifest_sha256,
        next_update=head.update_index + 1,
        next_collection_index=next_collection_index,
        next_policy_version=head.policy_version,
        next_base_seed=_next_seed(experiment_seed, next_collection_index, head.run_id),
        seed_derivation="self_play_grpo/production_collection/v1",
        consumed_batches=tuple(record.source_manifest_sha256 for record in records),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a D6 production restart boundary without HPUs")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    plan = audit_recovery(args.root, config_path=args.config,
                          experiment_seed=args.seed, expected_run_id=args.run_id)
    print(json.dumps(asdict(plan), sort_keys=True))
    return 0 if plan.status == "ready_for_next_collection" else 2


if __name__ == "__main__":
    raise SystemExit(main())
