import pytest

from self_play_grpo.cli import _canonical_action_labels
from self_play_grpo.policies import llm
from self_play_grpo.policies.llm import TokenTrie, summarize_log_prob_replay


def test_token_trie_exposes_only_legal_prefix_continuations() -> None:
    trie = TokenTrie({"MOVE_A1": (10, 20), "MOVE_A2": (10, 21), "WALL_A1H": (30, 40)})
    assert trie.allowed(()) == (10, 30)
    assert trie.allowed((10,)) == (20, 21)
    assert trie.completed_label((10, 20)) == "MOVE_A1"
    assert trie.completed_label((10,)) is None


def test_token_trie_rejects_out_of_grammar_prefix() -> None:
    trie = TokenTrie({"MOVE_A1": (10, 20)})
    with pytest.raises(ValueError, match="outside"):
        trie.allowed((99,))


def test_complete_nine_by_nine_action_label_family() -> None:
    labels = _canonical_action_labels(9)
    assert len(labels) == 81 + 8 * 8 * 2
    assert len(set(labels)) == len(labels)
    assert "MOVE_A1" in labels
    assert "MOVE_I9" in labels
    assert "WALL_A1H" in labels
    assert "WALL_H8V" in labels
    assert "PASS" not in labels


def test_authoritative_log_probs_use_sampling_cache_path(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(llm, "constrained_log_probs_cached", lambda model, sample: sentinel)
    assert llm.constrained_log_probs(object(), object()) is sentinel


def test_authoritative_log_probs_dispatch_recorded_batched_shape(monkeypatch) -> None:
    sentinel = object()
    sample = type(
        "Sample",
        (),
        {"sampling_config": {"probability_path": "batched_kv_cache"}},
    )()
    monkeypatch.setattr(
        llm, "constrained_log_probs_batched_shape", lambda model, row: sentinel
    )
    assert llm.constrained_log_probs(object(), sample) is sentinel


def test_finite_log_prob_replay_summary_reports_max_error() -> None:
    row = summarize_log_prob_replay((-0.5, 0.0), (-0.25, 0.0))
    assert row["finite"] is True
    assert row["max_abs_error"] == 0.25
    assert row["nonfinite_behavior_positions"] == []
    assert row["nonfinite_replayed_positions"] == []


@pytest.mark.parametrize(
    ("behavior", "replayed", "bad_side"),
    (
        ((float("nan"),), (0.0,), "behavior"),
        ((0.0,), (float("inf"),), "replayed"),
    ),
)
def test_nonfinite_log_prob_replay_summary_fails_closed(
    behavior, replayed, bad_side
) -> None:
    row = summarize_log_prob_replay(behavior, replayed)
    assert row["finite"] is False
    assert row["max_abs_error"] is None
    assert row[f"nonfinite_{bad_side}_positions"] == [0]
