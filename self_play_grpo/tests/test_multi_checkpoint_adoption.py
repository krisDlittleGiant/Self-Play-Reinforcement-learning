from types import SimpleNamespace as NS
import pytest
from self_play_grpo.training import multi_run as m
from self_play_grpo.training.multi_evidence import publish_json
from test_multi_recovery_flow import setup_case, LAYOUT


@pytest.mark.parametrize("checkpoint_published", [False, True])
def test_retry_retrains_only_when_no_checkpoint_was_published(tmp_path, monkeypatch, checkpoint_published):
    root, config, prior, plan, args = setup_case(tmp_path)
    output = root / "trainer-000003"
    output.mkdir()
    (output / "interrupted.txt").write_text("evidence")
    if checkpoint_published:
        (output / "trainer/checkpoints/policy-000003").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(m, "launch", lambda *a: calls.append("trainer"))
    def finalize(**kw):
        for rank in range(4):
            publish_json(output / "ranks" / f"rank-{rank:03d}.json", {"rank": rank})
        return {"checkpoint_adapter_sha256": "f" * 64, "metrics": {"optimizer_steps": 3}}
    monkeypatch.setattr(m, "finalize_checkpoint", finalize)
    monkeypatch.setattr(m, "_reconstruct_coordinator", lambda **kw: object())
    def commit(*a, **kw):
        calls.append("commit")
        return NS(checkpoint_manifest_sha256="e" * 64)
    monkeypatch.setattr(m, "publish_cycle_checkpoint", commit)
    m.update_next(args, config, LAYOUT, plan, 0, root / "rollout-000002", prior, NS(manifest_sha256="d" * 64))
    assert calls == (["commit"] if checkpoint_published else ["trainer", "commit"])
    if checkpoint_published:
        assert (output / "interrupted.txt").exists()
    else:
        assert len(list((root / "recovery_archive").glob("*/interrupted.txt"))) == 1
