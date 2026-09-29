"""No-HPU tests for strict D4 evidence preflight."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.rollouts.pilot import file_sha256
from self_play_grpo.training import d4_acceptance as module
from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.d4_acceptance import verify_d4_acceptance, verify_d4_gate


LAYOUT = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))


def fixture(tmp_path: Path, size: int) -> Path:
    root = tmp_path / f"gate-{size}"
    checkpoint = root / "checkpoints" / "policy-000002"
    manifest = checkpoint / "distributed" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"fixture": size}))
    digest = file_sha256(manifest)
    modules = tuple(range(4, 8)) if size == 4 else (4, 5)
    reports = [
        {
            "rank": rank, "status": "ok", "module_id": str(module_id),
            "checkpoint_manifest_sha256": digest,
            "cpu_rng_equal": True, "hpu_rng_equal": True,
            "parameters_equal": True, "optimizer_equal": True,
            "metrics_equal": True,
        }
        for rank, module_id in enumerate(modules)
    ]
    summary = {
        "status": "ok", "validation_only": True, "world_size": size,
        "run_id": f"gate-{size}", "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": digest,
        "source_rollout": "shared-rollout", "source_checkpoint": "shared-source",
        "runtime_contract": {
            "status": "ok", "world_size": size,
            "module_ids": [str(module) for module in modules],
        },
        "rank_reports": reports,
    }
    source = root / "summary.json"
    source.write_text(json.dumps(summary))
    return source


@pytest.fixture
def checkpoint_reader(monkeypatch):
    def reader(path):
        size = int(Path(path).parents[1].name.split("-")[-1])
        modules = tuple(range(4, 8)) if size == 4 else (4, 5)
        return SimpleNamespace(
            run_kind="validation", trainer_world_size=size,
            run_id=f"gate-{size}",
            ranks=tuple(SimpleNamespace(module_id=str(module)) for module in modules),
        )
    monkeypatch.setattr(module, "read_distributed_manifest", reader)
    monkeypatch.setattr(module, "verify_checkpoint_files", lambda *args: None)


def test_both_d4_gates_are_required(tmp_path, checkpoint_reader):
    two = fixture(tmp_path, 2)
    four = fixture(tmp_path, 4)
    pair = verify_d4_acceptance(two, four, layout=LAYOUT)
    assert [receipt.world_size for receipt in pair] == [2, 4]
    assert pair[1].module_ids == LAYOUT.trainer_modules


def test_missing_or_failed_rank_equality_rejected(tmp_path, checkpoint_reader):
    summary = fixture(tmp_path, 4)
    data = json.loads(summary.read_text())
    data["rank_reports"][2]["optimizer_equal"] = False
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="continuation equality failed"):
        verify_d4_gate(summary, world_size=4)


def test_changed_checkpoint_manifest_rejected(tmp_path, checkpoint_reader):
    summary = fixture(tmp_path, 4)
    checkpoint = Path(json.loads(summary.read_text())["checkpoint"])
    (checkpoint / "distributed" / "manifest.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="manifest bytes changed"):
        verify_d4_gate(summary, world_size=4)


def test_d4_layout_must_match_future_trainer_roles(tmp_path, checkpoint_reader):
    two = fixture(tmp_path, 2)
    four = fixture(tmp_path, 4)
    wrong_layout = RoleLayout((4, 5, 6, 7), (0, 1, 2, 3))
    with pytest.raises(ValueError, match="D5 trainer modules"):
        verify_d4_acceptance(two, four, layout=wrong_layout)


def test_validation_only_checkpoint_is_accepted_as_evidence_not_training_source(tmp_path, checkpoint_reader):
    summary = fixture(tmp_path, 2)
    receipt = verify_d4_gate(summary, world_size=2)
    assert receipt.world_size == 2
    assert receipt.source_checkpoint == "shared-source"
