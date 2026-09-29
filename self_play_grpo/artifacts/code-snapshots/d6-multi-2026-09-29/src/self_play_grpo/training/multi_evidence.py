"""Checkpoint-inventoried evidence closes the save-before-acknowledgement gap."""
from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from self_play_grpo.rollouts.pilot import directory_sha256, file_sha256
from self_play_grpo.training.coordinator import PolicyDescriptor, verify_completed_batch
from self_play_grpo.training.d6_recovery import _json_file
from self_play_grpo.training.distributed_checkpoint import read_distributed_manifest, verify_checkpoint_files
from self_play_grpo.training.ledger import read_commits
from self_play_grpo.training.loop import UpdateMetrics
from self_play_grpo.training.multi_identity import multi_code_identity
from self_play_grpo.training.policy_refresh import validate_probe, source_sample


@dataclass
class CommittedMetrics(UpdateMetrics):
    # Saved inside trainer_state.json and covered by the format-3 inventory.
    continuation_evidence: dict[str, Any] = field(default_factory=dict)


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically create evidence, or verify identical existing evidence."""
    if path.is_symlink():
        raise ValueError(f"Symlinked evidence: {path}")
    if path.exists():
        if _json_file(path) != value:
            raise ValueError(f"Existing evidence differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_checkpoint(*, root: Path, output: Path, rollout_root: Path,
                        config: Any, layout: Any, run_id: str, update: int) -> dict[str, Any]:
    """Reconstruct missing worker reports from evidence inside a valid checkpoint.

    A present but invalid checkpoint is an error. Never discard it and retrain.
    This function works before or after ledger commitment and starts no model.
    """
    if update < 3:
        raise ValueError("Multi-update checkpoint must be update 3 or later")
    prior = PolicyDescriptor(**_json_file(rollout_root / "policy_descriptor.json"))
    if prior.update_index != update - 1 or prior.version != f"policy-{update - 1:06d}":
        raise ValueError("Checkpoint source policy is not its predecessor")
    receipt = verify_completed_batch(rollout_root, config=config.to_dict(), policy=prior, expected_games=64)
    checkpoint = output / "trainer" / "checkpoints" / f"policy-{update:06d}"
    if checkpoint.is_symlink():
        raise ValueError("Symlinked checkpoint")
    manifest = read_distributed_manifest(checkpoint)
    verify_checkpoint_files(checkpoint, manifest)
    if (manifest.run_kind != "production" or manifest.run_id != run_id
            or manifest.update_index != update or manifest.policy_version != f"policy-{update:06d}"
            or manifest.code_identity != multi_code_identity()
            or manifest.config_sha256 != prior.config_sha256
            or manifest.model_id != config.model.id or manifest.model_revision != prior.model_revision
            or manifest.tokenizer_sha256 != prior.tokenizer_sha256
            or manifest.grammar_sha256 != prior.grammar_sha256
            or manifest.source_rollout_manifest_sha256 != receipt.manifest_sha256
            or manifest.trainer_world_size != 4
            or tuple(row.module_id for row in manifest.ranks) != tuple(map(str, layout.trainer_modules))):
        raise ValueError("Published checkpoint identity differs from the requested update")
    for rank, row in enumerate(manifest.ranks):
        if row.match_indices != receipt.match_indices_by_rank[rank]:
            raise ValueError("Checkpoint shard differs from the verified batch")
    commits = read_commits(root)
    if len(commits) not in (update - 1, update):
        raise ValueError("Checkpoint does not adjoin the ledger head")
    previous = commits[update - 2]
    if (previous.policy_version != prior.version or previous.run_id != run_id
            or directory_sha256(root / previous.checkpoint_path / "adapter") != prior.adapter_sha256):
        raise ValueError("Checkpoint predecessor differs from the committed policy")
    if any(row.source_manifest_sha256 == receipt.manifest_sha256 for row in commits[:update - 1]):
        raise ValueError("Checkpoint batch was already consumed")
    state = _json_file(checkpoint / "trainer_state.json")
    metrics = dict(state["last_update"])
    evidence = metrics.pop("continuation_evidence")
    if (state.get("update_index") != update or state.get("policy_version") != manifest.policy_version
            or state.get("validation_only") is not False
            or metrics.get("update") != update or metrics.get("optimizer_steps") != update
            or metrics.get("policy_version") != manifest.policy_version or metrics.get("games") != 64
            or metrics.get("owned_tokens") != sum(receipt.owned_tokens_by_rank)):
        raise ValueError("Checkpoint trainer counters or coverage differ")
    for key in ("loss", "policy_loss", "kl_loss", "grad_norm", "mean_ratio_before_step",
                "clip_fraction_before_step", "replay_max_abs_error"):
        if type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key]):
            raise ValueError(f"Nonfinite checkpoint metric: {key}")
    from self_play_grpo.rollouts.pilot import read_pilot_manifest
    tolerance = read_pilot_manifest(rollout_root).replay_tolerance
    if not 0 <= metrics["replay_max_abs_error"] <= tolerance:
        raise ValueError("Checkpoint behavior replay failed")
    probe = validate_probe(evidence["policy_probe"])
    sample, match_digest = source_sample(rollout_root)
    if (probe["match_sha256"] != match_digest or
            probe["completion_token_ids"] != list(sample.completion_token_ids)):
        raise ValueError("Checkpoint probe differs from source action")
    reports = evidence["rank_reports"]
    if len(reports) != 4:
        raise ValueError("Checkpoint lacks four trainer reports")
    parameters, optimizers = set(), set()
    for rank, report in enumerate(reports):
        result, restored = report["update"], report["restored"]
        if (report.get("rank") != rank or report.get("module_id") != str(layout.trainer_modules[rank])
                or restored.get("status") != "restored_no_optimizer_step"
                or restored.get("policy_version") != prior.version
                or restored.get("checkpoint_manifest_sha256") != previous.checkpoint_manifest_sha256
                or result.get("rank") != rank or result.get("source_policy_version") != prior.version
                or result.get("initial_parameter_sha256") != restored.get("parameter_sha256")
                or result.get("next_policy_version") != manifest.policy_version
                or result.get("consumed_manifest_sha256") != receipt.manifest_sha256
                or result.get("optimizer_steps") != update or result.get("gradient_sync_phases") != 1
                or result.get("owned_tokens") != receipt.owned_tokens_by_rank[rank]):
            raise ValueError(f"Checkpoint rank {rank} continuation evidence differs")
        from self_play_grpo.training.policy_refresh import _digest
        parameters.add(_digest(result.get("parameter_sha256"), "parameters"))
        optimizers.add(_digest(result.get("optimizer_sha256"), "optimizer"))
    if len(parameters) != 1 or len(optimizers) != 1:
        raise ValueError("Checkpoint trainer replicas differ")
    digest = file_sha256(checkpoint / "distributed" / "manifest.json")
    if len(commits) == update and commits[-1].checkpoint_manifest_sha256 != digest:
        raise ValueError("Checkpoint differs from the committed head")
    adapter = directory_sha256(checkpoint / "adapter")
    for rank, report in enumerate(reports):
        publish_json(output / "ranks" / f"rank-{rank:03d}.json", {
            **report, "status": "updated_checkpoint_committed_refresh_pending",
            "source_policy_version": prior.version, "source_manifest_sha256": receipt.manifest_sha256,
            "checkpoint_manifest_sha256": digest, "checkpoint_adapter_sha256": adapter,
        })
    summary = {
        "status": "updated_checkpoint_committed_refresh_pending", "run_id": run_id,
        "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": digest,
        "checkpoint_adapter_sha256": adapter, "parameter_sha256": next(iter(parameters)),
        "source_policy_version": prior.version, "source_manifest_sha256": receipt.manifest_sha256,
        "policy_probe": probe, "metrics": metrics,
    }
    publish_json(output / "trainer_summary.json", summary)
    return summary
