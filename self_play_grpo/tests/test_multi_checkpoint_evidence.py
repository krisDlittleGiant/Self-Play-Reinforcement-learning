"""Real CPU format-3 checkpoint saves cover the crash-before-worker-report gap."""
import json
from dataclasses import asdict, replace
from types import SimpleNamespace as NS

import pytest
import torch

from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256
from self_play_grpo.training import multi_evidence as m
from self_play_grpo.training.coordinator import PolicyDescriptor, RoleLayout
from self_play_grpo.training.distributed_checkpoint import TrainerRankState, read_distributed_manifest
from self_play_grpo.training.multi_identity import multi_code_identity
from test_distributed_checkpoint_io import _trainer, _template, _metrics, _expected


@pytest.fixture
def saved_update(tmp_path, monkeypatch):
    root = tmp_path
    output = root / "trainer-000003"
    trainer = _trainer(output / "trainer")
    trainer.update_index, trainer.policy_version = 3, "policy-000003"
    adapter = root / "trainer-000002/trainer/checkpoints/policy-000002/adapter"
    adapter.mkdir(parents=True)
    (adapter / "weights").write_bytes(b"previous")
    prior = PolicyDescriptor("policy-000002", 2, directory_sha256(adapter),
                             canonical_sha256(trainer.config.to_dict()), "revision", "4" * 64, "5" * 64, "production")
    rollout = root / "rollout-000002"
    rollout.mkdir()
    (rollout / "policy_descriptor.json").write_text(json.dumps(asdict(prior)))
    receipt = NS(manifest_sha256="6" * 64, match_indices_by_rank=((0,), (1,), (2,), (3,)),
                 owned_tokens_by_rank=(1, 1, 1, 1))
    monkeypatch.setattr(m, "verify_completed_batch", lambda *a, **kw: receipt)
    previous = NS(policy_version=prior.version, run_id="test",
                  checkpoint_path=str(adapter.parent.relative_to(root)), checkpoint_manifest_sha256="a" * 64,
                  source_manifest_sha256="b" * 64)
    monkeypatch.setattr(m, "read_commits", lambda _: (NS(source_manifest_sha256="c" * 64), previous))
    monkeypatch.setattr(m, "source_sample", lambda _: (NS(completion_token_ids=(10,)), "d" * 64))
    monkeypatch.setattr("self_play_grpo.rollouts.pilot.read_pilot_manifest", lambda _: NS(replay_tolerance=2e-4))
    rows = []
    rank_states = []
    staging = root / "rng"
    for rank in range(4):
        rows.append({"rank": rank, "module_id": str(rank + 4),
                     "restored": {"status": "restored_no_optimizer_step", "policy_version": prior.version,
                                  "checkpoint_manifest_sha256": "a" * 64, "parameter_sha256": "e" * 64},
                     "update": {"rank": rank, "source_policy_version": prior.version,
                                "initial_parameter_sha256": "e" * 64, "next_policy_version": "policy-000003",
                                "consumed_manifest_sha256": "6" * 64, "optimizer_steps": 3,
                                "gradient_sync_phases": 1, "owned_tokens": 1,
                                "parameter_sha256": "f" * 64, "optimizer_sha256": "1" * 64}})
        record = TrainerRankState(rank, rank, str(rank + 4), f"hpu:{rank}",
                                  f"distributed/rng/rank-{rank:03d}.pt", (rank,), (f"game-{rank}",), (f"{rank+1:064x}",))
        rank_states.append(record)
        path = staging / record.rng_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 1, "rank": rank, "module_id": str(rank + 4),
                    "cpu": torch.get_rng_state(), "hpu": [torch.tensor([rank], dtype=torch.uint8)],
                    "hpu_api": "torch.hpu.get_rng_state_all"}, path)
    template = replace(_template(), run_kind="production", run_id="test", update_index=3,
                       policy_version="policy-000003", trainer_world_size=4, ranks=tuple(rank_states),
                       code_identity=multi_code_identity())
    metrics = m.CommittedMetrics(**asdict(replace(_metrics(), update=3, policy_version="policy-000003",
                                                 optimizer_steps=3, games=64, owned_tokens=4)),
                                 continuation_evidence={"rank_reports": rows, "policy_probe": {
                                     "game_index": 0, "joint_step": 0, "match_sha256": "d" * 64,
                                     "completion_token_ids": [10], "log_probs": [0.0]}})
    checkpoint = trainer.save_checkpoint(metrics, distributed_manifest=template, rank_rng_staging=staging)
    kwargs = dict(root=root, output=output, rollout_root=rollout, config=trainer.config,
                  layout=RoleLayout((0, 1, 2, 3), (4, 5, 6, 7)), run_id="test", update=3)
    return trainer, checkpoint, kwargs


def test_published_checkpoint_rebuilds_missing_reports_idempotently(saved_update):
    trainer, checkpoint, kwargs = saved_update
    assert not (kwargs["output"] / "ranks").exists()
    result = m.finalize_checkpoint(**kwargs)
    assert result["metrics"]["optimizer_steps"] == 3
    assert len(list((kwargs["output"] / "ranks").glob("rank-*.json"))) == 4
    assert m.finalize_checkpoint(**kwargs) == result


def test_embedded_evidence_is_hashed_and_cannot_be_tampered(saved_update):
    _, checkpoint, kwargs = saved_update
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["last_update"]["continuation_evidence"]["rank_reports"][0]["update"]["optimizer_steps"] = 999
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        m.finalize_checkpoint(**kwargs)
    assert not (kwargs["output"] / "trainer_summary.json").exists()


def test_existing_loader_restores_checkpoint_with_embedded_evidence(saved_update, monkeypatch):
    trainer, checkpoint, _ = saved_update
    manifest = read_distributed_manifest(checkpoint)
    weights = trainer.policy.model.weight.detach().clone()
    moment = next(iter(trainer.optimizer.state.values()))["exp_avg"].clone()
    trainer.policy.model.weight.data.add_(10)
    next(iter(trainer.optimizer.state.values()))["exp_avg"].add_(10)
    restored = []
    monkeypatch.setattr(torch, "hpu", NS(set_rng_state_all=restored.append), raising=False)
    trainer.load_checkpoint(checkpoint, distributed_expected=_expected(manifest), distributed_rank=2)
    assert trainer.update_index == 3
    assert torch.equal(weights, trainer.policy.model.weight)
    assert torch.equal(moment, next(iter(trainer.optimizer.state.values()))["exp_avg"])
    assert restored[0][0].item() == 2
