"""D5 checkpoint publication and policy-refresh evidence boundary.

The live launcher must call these functions after four trainer update reports
and before permitting any new rollout.  No model or HPU is loaded here.
"""

from __future__ import annotations

import math
from pathlib import Path

from self_play_grpo.rollouts.pilot import directory_sha256, file_sha256
from self_play_grpo.training.coordinator import (
    CycleCoordinator,
    Phase,
    PolicyDescriptor,
)
from self_play_grpo.training.distributed_checkpoint import MANIFEST_PATH
from self_play_grpo.training.ledger import CommitRecord, publish_commit, read_commits


def publish_cycle_checkpoint(
    coordinator: CycleCoordinator,
    *,
    run_root: str | Path,
    checkpoint_path: str,
    run_id: str,
) -> CommitRecord:
    """Bind the matching updated adapter, checkpoint and consumed batch.

    If ledger publication succeeds but the coordinator later fails, the ledger
    remains authoritative for restart.  It must never be rolled back merely
    because refresh acknowledgement was incomplete.
    """

    if coordinator.phase is not Phase.UPDATED:
        coordinator._fail("Checkpoint publication requires four matching trainer updates")
    assert coordinator.batch is not None and coordinator.next_policy is not None
    if coordinator.policy.run_kind != "production":
        coordinator._fail("Validation-only update cannot publish a production commit")
    root = Path(run_root)
    try:
        checkpoint = root / checkpoint_path
        adapter_digest = directory_sha256(checkpoint / "adapter")
        if adapter_digest != coordinator.next_policy.adapter_sha256:
            raise ValueError("Checkpoint adapter differs from trainer-reported policy")
        records = read_commits(root)
        if records:
            head = records[-1]
            if (
                head.run_id != run_id
                or head.update_index != coordinator.policy.update_index
                or head.policy_version != coordinator.policy.version
                or directory_sha256(root / head.checkpoint_path / "adapter")
                != coordinator.policy.adapter_sha256
            ):
                raise ValueError("Coordinator policy differs from committed ledger head")
        elif coordinator.policy.update_index != 0:
            raise ValueError("Non-initial coordinator requires a committed ledger head")
        manifest_digest = file_sha256(checkpoint / MANIFEST_PATH)
        record = CommitRecord(
            run_id=run_id,
            update_index=coordinator.next_policy.update_index,
            policy_version=coordinator.next_policy.version,
            source_manifest_sha256=coordinator.batch.manifest_sha256,
            checkpoint_path=checkpoint_path,
            checkpoint_manifest_sha256=manifest_digest,
            previous_record_sha256=records[-1].sha256 if records else None,
        )
        publish_commit(root, record)
        coordinator.commit_checkpoint(
            manifest_sha256=record.source_manifest_sha256,
            checkpoint_sha256=record.checkpoint_manifest_sha256,
            policy=coordinator.next_policy,
        )
    except Exception as exc:
        coordinator._fail(f"D5 checkpoint handoff failed: {exc}")
    return record


def acknowledge_refresh_with_identity(
    coordinator: CycleCoordinator,
    rank: int,
    *,
    policy: PolicyDescriptor,
    loaded_parameter_sha256: str,
    probe_error: float,
    probe_tolerance: float,
) -> None:
    """Require loaded tensor identity as well as the numerical probe."""

    if coordinator.phase is not Phase.REFRESHING:
        coordinator._fail("Refresh acknowledgement is outside the refresh phase")
    if len(coordinator.updated) != 4:
        coordinator._fail("Refresh requires four matching trainer state reports")
    expected = coordinator.updated[0][0]
    if loaded_parameter_sha256 != expected:
        coordinator._fail("Rollout worker loaded different trainable tensor values")
    if not math.isfinite(probe_error) or not math.isfinite(probe_tolerance):
        coordinator._fail("Rollout refresh probe is non-finite")
    coordinator.acknowledge_refreshed(
        rank, policy=policy,
        probe_error=probe_error, probe_tolerance=probe_tolerance,
    )
