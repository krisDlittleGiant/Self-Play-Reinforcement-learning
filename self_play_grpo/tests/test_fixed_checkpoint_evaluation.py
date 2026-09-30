from pathlib import Path
from types import SimpleNamespace

import pytest

from self_play_grpo.config import load_config
from self_play_grpo.evaluation.fixed_checkpoint import (
    LINEUPS, build_suite, compare, evaluate, summarize,
)
from self_play_grpo.rollouts.pilot import canonical_sha256


CONFIG = Path(__file__).parents[1] / "configs/quoridor_outcome_64games.yaml"
FIXTURE = Path(__file__).parents[1] / "configs/quoridor_fixture.yaml"


def test_frozen_schedule_is_seat_balanced_reproducible_and_separate_from_training():
    config = load_config(CONFIG)
    suite = build_suite(config, games_per_seat=4, seed=20260929)
    assert suite == build_suite(config, games_per_seat=4, seed=20260929)
    assert suite["lineups"] == list(LINEUPS)
    assert len(suite["games"]) == 4 * 4 * len(LINEUPS)
    assert len({game["seed"] for game in suite["games"]}) == len(suite["games"])
    for opponent in LINEUPS:
        for seat in range(4):
            assert sum(game["opponent"] == opponent and game["seat"] == seat for game in suite["games"]) == 4


def test_game_seeds_depend_only_on_opponent_seat_and_repeat():
    config = load_config(CONFIG)
    small = build_suite(config, games_per_seat=1, seed=5, lineups=("shortest",))
    large = build_suite(config, games_per_seat=2, seed=5, lineups=("random", "shortest"))
    identity = {"namespace": "self_play_grpo/evaluation/v1", "seed": 5,
                "opponent": "shortest", "seat": 0, "repeat": 0}
    assert small["games"][0]["seed"] == int(canonical_sha256(identity)[:15], 16)
    assert small["games"][0]["seed"] in {game["seed"] for game in large["games"]}


def test_invalid_lineups_are_rejected():
    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="Line-ups"):
        build_suite(config, games_per_seat=1, seed=1, lineups=("random", "random"))
    with pytest.raises(ValueError, match="Line-ups"):
        build_suite(config, games_per_seat=1, seed=1, lineups=("grandmaster",))


def test_modified_suite_is_rejected_before_model_load(tmp_path):
    config = load_config(CONFIG)
    suite = build_suite(config, games_per_seat=1, seed=20)
    suite["games"][0]["seed"] += 1
    with pytest.raises(ValueError, match="Frozen evaluation suite differs"):
        evaluate(SimpleNamespace(run_root=tmp_path, output=tmp_path / "output", update=0), config, suite)
    assert not (tmp_path / "output").exists()


@pytest.mark.engine
def test_sharded_scripted_candidate_evaluation_summarizes_and_compares(tmp_path):
    pytest.importorskip("pyspiel")
    config = load_config(FIXTURE)
    suite = build_suite(config, games_per_seat=1, seed=3, lineups=("random", "noisy-shortest"))

    def args(shard):
        return SimpleNamespace(run_root=tmp_path, output=tmp_path / "eval", update=None,
                               candidate_bot="shortest", shard_index=shard, num_shards=2,
                               save_matches=True)

    first = evaluate(args(0), config, suite)
    assert first == {"status": "shard_complete", "shard": 0, "num_shards": 2, "games": 4}
    summary = evaluate(args(1), config, suite)
    assert summary["status"] == "evaluation_complete"
    assert summary["overall"]["games"] == 8
    for lineup in ("random", "noisy-shortest"):
        row = summary["by_opponent"][lineup]
        assert row["games"] == 4 and 0.0 <= row["mean_fractional_result"] <= 1.0
        assert row["mean_final_distance"] >= 0.0 and 1.0 <= row["mean_placement"] <= 4.0
    assert len(list((tmp_path / "eval" / "matches").glob("game-*.jsonl"))) == 8
    assert summarize(suite, tmp_path / "eval") == summary
    paired = compare(suite, tmp_path / "eval", tmp_path / "eval")
    assert paired["overall"]["result_difference"] == 0.0
    assert paired["overall"]["final_distance_difference"] == 0.0
    # A re-run of a finished shard reuses its games and does not replay them.
    assert evaluate(args(0), config, suite)["status"] == "evaluation_complete"


@pytest.mark.engine
def test_noisy_shortest_bot_is_legal_and_reduces_to_shortest_without_noise():
    pytest.importorskip("pyspiel")
    import random

    from self_play_grpo.envs.quoridor import QuoridorEnv
    from self_play_grpo.policies.bots import NoisyShortestPathPolicy, ShortestPathPolicy

    env = QuoridorEnv(load_config(FIXTURE).environment)
    for seed in range(5):
        noisy = NoisyShortestPathPolicy(epsilon=0.0).select_action(env, game_id="g", rng=random.Random(seed))
        exact = ShortestPathPolicy().select_action(env, game_id="g", rng=random.Random(seed))
        assert noisy.chosen_label == exact.chosen_label
        wild = NoisyShortestPathPolicy(epsilon=1.0).select_action(env, game_id="g", rng=random.Random(seed))
        assert wild.chosen_label in {action.label for action in env.legal_actions() if action.kind == "move"}


def test_concurrent_shards_accept_an_identical_shared_file_and_reject_a_different_one(tmp_path, monkeypatch):
    import json

    from self_play_grpo.evaluation import fixed_checkpoint as module

    target = tmp_path / "candidate.json"

    def lose_race(path, value):
        # Another shard links its identical copy first; our own link then fails.
        path.write_text(json.dumps(winner), encoding="utf-8")
        raise FileExistsError(path)

    monkeypatch.setattr(module, "publish_json", lose_race)
    winner = {"policy_version": "policy-000000"}
    module._publish_shared(target, {"policy_version": "policy-000000"})
    winner = {"policy_version": "policy-000001"}
    with pytest.raises(FileExistsError):
        module._publish_shared(target, {"policy_version": "policy-000000"})
