"""Prepare the next immutable 64-game collection from a committed checkpoint.

This does not start any HPU worker.  It is deliberately separate from the
one-update D5 driver so an incomplete refresh can never launch new games.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256
from self_play_grpo.training.coordinator import PolicyDescriptor
from self_play_grpo.training.d6_recovery import RecoveryPlan, audit_recovery
from self_play_grpo.training.distributed_checkpoint import read_distributed_manifest
from self_play_grpo.training.rollout_worker import prepare_frozen_batch


def descriptor_from_commit(
    root: str | Path, plan: RecoveryPlan, *, config: Any,
) -> PolicyDescriptor:
    """Bind the next collection to the head checkpoint and original grammar."""

    root = Path(root)
    if plan.status != "ready_for_next_collection":
        raise RuntimeError("D6 collection is forbidden until all refresh ranks verify")
    prior_path = root / f"rollout-{plan.committed_update - 1:06d}" / "policy_descriptor.json"
    if prior_path.is_symlink() or not prior_path.is_file():
        raise ValueError("D6 preceding policy descriptor is missing or symlinked")
    prior = PolicyDescriptor(**json.loads(prior_path.read_text(encoding="utf-8")))
    if (prior.run_kind != "production"
            or prior.update_index != plan.committed_update - 1
            or prior.version != f"policy-{prior.update_index:06d}"):
        raise ValueError("D6 preceding policy descriptor has the wrong version")
    checkpoint = root / plan.checkpoint
    manifest = read_distributed_manifest(checkpoint)
    if (manifest.run_kind != "production"
            or manifest.run_id != plan.run_id
            or manifest.policy_version != plan.policy_version
            or manifest.update_index != plan.committed_update
            or manifest.tokenizer_sha256 != prior.tokenizer_sha256
            or manifest.grammar_sha256 != prior.grammar_sha256
            or manifest.config_sha256 != prior.config_sha256
            or prior.config_sha256 != canonical_sha256(config.to_dict())):
        raise ValueError("D6 checkpoint and preceding policy identities differ")
    adapter_digest = directory_sha256(checkpoint / "adapter")
    descriptor = PolicyDescriptor(
        version=plan.policy_version, update_index=plan.committed_update,
        adapter_sha256=adapter_digest, config_sha256=prior.config_sha256,
        model_revision=prior.model_revision,
        tokenizer_sha256=prior.tokenizer_sha256,
        grammar_sha256=prior.grammar_sha256, run_kind="production",
    )
    return descriptor


def prepare_next_collection(
    root: str | Path, *, config_path: str | Path, experiment_seed: int,
    expected_run_id: str, replay_tolerance: float,
) -> dict[str, Any]:
    """Copy the committed adapter into a fresh frozen-batch directory."""

    root = Path(root)
    config = load_config(config_path)
    plan = audit_recovery(
        root, config_path=config_path, experiment_seed=experiment_seed,
        expected_run_id=expected_run_id,
    )
    descriptor = descriptor_from_commit(root, plan, config=config)
    output = root / f"rollout-{plan.next_collection_index:06d}"
    prepare_frozen_batch(
        output, config=config, source_adapter=root / plan.checkpoint / "adapter",
        policy=descriptor, base_seed=plan.next_base_seed,
        replay_tolerance=replay_tolerance,
    )
    return {
        "status": "prepared_not_collected",
        "run_id": plan.run_id,
        "output": str(output),
        "collection_index": plan.next_collection_index,
        "next_update": plan.next_update,
        "base_seed": plan.next_base_seed,
        "policy_descriptor": asdict(descriptor),
    }
