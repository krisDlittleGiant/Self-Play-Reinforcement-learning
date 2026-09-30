"""Freeze a seat-balanced opponent suite and evaluate immutable actor adapters.

Evaluation runs in separate processes and never constructs a trainer or an
optimizer. The candidate always occupies one rotating seat; the other three
seats are one opponent line-up:

- ``random``: uniformly random legal actions;
- ``noisy-shortest``: shortest-path racer with random pawn moves half the time;
- ``shortest``: shortest-path racer;
- ``wall-aware``: frozen heuristic weighing progress against obstruction;
- ``initial-llm``: three copies of the run's frozen initial policy (a clean
  25% baseline for the candidate).

Besides the win/draw result, every game records continuous metrics (final
path distance to goal, finishing place, forward and backward moves), so an
evaluation is informative even where one line-up is never beaten. A suite can
be split into shards that run concurrently on different HPUs; ``summarize``
merges them and ``compare`` pairs two checkpoints game by game.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import platform
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from self_play_grpo.config import load_config
from self_play_grpo.evaluation.tournament import EvaluationGame, bootstrap_match_mean
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256, read_pilot_manifest
from self_play_grpo.training.d6_recovery import _json_file
from self_play_grpo.training.ledger import read_commits
from self_play_grpo.training.multi_evidence import publish_json


LINEUPS = ("random", "noisy-shortest", "shortest", "wall-aware", "initial-llm")
BOTS = ("random", "noisy-shortest", "shortest", "wall-aware")
_SOURCES = ("evaluation/fixed_checkpoint.py", "evaluation/tournament.py", "policies/bots.py",
            "policies/llm.py", "envs/quoridor.py", "envs/observations.py", "rollouts/collector.py",
            "rewards/progress.py", "rollouts/analysis.py")


def build_suite(config, *, games_per_seat: int, seed: int, lineups: Sequence[str] = LINEUPS) -> dict:
    if type(games_per_seat) is not int or games_per_seat <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Evaluation game count and seed must be valid")
    lineups = tuple(lineups)
    if not lineups or len(set(lineups)) != len(lineups) or any(item not in LINEUPS for item in lineups):
        raise ValueError(f"Line-ups must be distinct members of {LINEUPS}")
    package = Path(__file__).resolve().parents[1]
    games = []
    for opponent in lineups:
        for seat in range(4):
            for repeat in range(games_per_seat):
                identity = {"namespace": "self_play_grpo/evaluation/v1", "seed": seed,
                            "opponent": opponent, "seat": seat, "repeat": repeat}
                game_seed = int(canonical_sha256(identity)[:15], 16)
                games.append({"index": len(games), "opponent": opponent, "seat": seat,
                              "repeat": repeat, "seed": game_seed})
    return {"schema_version": 2, "config_sha256": canonical_sha256(config.to_dict()),
            "seed": seed, "games_per_seat_per_opponent": games_per_seat, "lineups": list(lineups),
            "source_sha256": {name: file_sha256(package / name) for name in _SOURCES},
            "bot_parameters": {"wall_aware": {"opponent_weight": 0.25, "wall_cost": 0.05},
                               "noisy_shortest": {"epsilon": 0.5}},
            "initial_board": "standard", "games": games}


def _verify_suite(config, suite: dict) -> None:
    expected = build_suite(config, games_per_seat=suite.get("games_per_seat_per_opponent"),
                           seed=suite.get("seed"), lineups=tuple(suite.get("lineups", ())))
    if suite != expected:
        raise ValueError("Frozen evaluation suite differs from the current code/config")


def _initial_adapter(root: Path, config) -> tuple[Path, str]:
    manifest = read_pilot_manifest(root / "rollout-000000")
    if manifest.config_sha256 != canonical_sha256(config.to_dict()):
        raise ValueError("Initial policy config differs")
    adapter = root / "rollout-000000" / "policy_adapter"
    if directory_sha256(adapter) != manifest.adapter_sha256:
        raise ValueError("Initial adapter content differs")
    return adapter, manifest.adapter_sha256


def candidate_identity(root: Path, config, update: int):
    if type(update) is not int or update < 0:
        raise ValueError("Candidate update must be non-negative")
    if update == 0:
        adapter, expected = _initial_adapter(root, config)
        checkpoint_hash = None
    else:
        records = read_commits(root)
        if update > len(records):
            raise ValueError("Candidate checkpoint has not been committed")
        from self_play_grpo.training.distributed_checkpoint import read_distributed_manifest
        record = records[update - 1]
        checkpoint = root / record.checkpoint_path
        manifest = read_distributed_manifest(checkpoint)
        if (manifest.config_sha256 != canonical_sha256(config.to_dict())
                or manifest.model_id != config.model.id or manifest.model_revision != config.model.revision):
            raise ValueError("Candidate checkpoint configuration differs")
        adapter = checkpoint / "adapter"
        expected = directory_sha256(adapter)
        checkpoint_hash = record.checkpoint_manifest_sha256
    if directory_sha256(adapter) != expected:
        raise ValueError("Candidate adapter content differs")
    return adapter, {"policy_version": f"policy-{update:06d}", "adapter_sha256": expected,
                     "checkpoint_manifest_sha256": checkpoint_hash}


def _bot(name: str):
    from self_play_grpo.policies.bots import (
        NoisyShortestPathPolicy, RandomPolicy, ShortestPathPolicy, WallAwarePolicy,
    )
    return {"random": RandomPolicy, "noisy-shortest": NoisyShortestPathPolicy,
            "shortest": ShortestPathPolicy, "wall-aware": WallAwarePolicy}[name]()


def game_metrics(match, seat: int) -> dict[str, Any]:
    """Continuous candidate metrics from one complete match record."""

    from self_play_grpo.rewards.progress import path_distances
    from self_play_grpo.rollouts.analysis import goal_progress_delta

    final = match.turns[-1].state_after if match.turns else match.initial_state
    distances = [int(value) for value in path_distances(final)]
    own = [turn for turn in match.turns if turn.seat == seat]
    moves = [turn for turn in own if turn.policy_sample.chosen_label.startswith("MOVE_")]
    return {
        "final_distance": distances[seat], "final_distances": distances,
        "placement": 1 + sum(distances[other] < distances[seat] for other in range(4) if other != seat),
        "turns": len(own), "moves": len(moves), "walls": len(own) - len(moves),
        "forward_moves": sum(goal_progress_delta(turn) > 0 for turn in moves),
        "backward_moves": sum(goal_progress_delta(turn) < 0 for turn in moves),
    }


def _publish_shared(path: Path, value: dict) -> None:
    """Publish evidence that concurrent shards write with identical content."""

    try:
        publish_json(path, value)
    except FileExistsError:
        # Another shard linked the same file between the existence check and
        # our link; accept it only if the content is identical.
        if _json_file(path) != value:
            raise


@contextmanager
def _shard_lock(output: Path, shard: int, shards: int) -> Iterator[None]:
    path = output / f".eval-shard-{shard:03d}-of-{shards:03d}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Evaluation shard {shard} of {shards} is already running") from exc
        yield
    finally:
        os.close(fd)


def evaluate(args, config, suite: dict) -> dict:
    """Play this shard's scheduled games; publish the summary once all exist."""

    # Validate the entire frozen schedule before loading any model.
    _verify_suite(config, suite)
    shards, shard = getattr(args, "num_shards", 1), getattr(args, "shard_index", 0)
    if type(shards) is not int or shards <= 0 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError("Shard index must be in [0, num_shards)")
    selected = [game for game in suite["games"] if game["index"] % shards == shard]
    bot_candidate = getattr(args, "candidate_bot", None)
    if bot_candidate is not None:
        if bot_candidate not in BOTS:
            raise ValueError(f"Candidate bot must be one of {BOTS}")
        identity = {"policy_version": f"bot:{bot_candidate}", "adapter_sha256": None,
                    "checkpoint_manifest_sha256": None}
        adapter = None
    else:
        adapter, identity = candidate_identity(args.run_root, config, args.update)
    uses_llm = adapter is not None or any(game["opponent"] == "initial-llm" for game in suite["games"])
    torch = None
    if uses_llm:
        import torch
        from self_play_grpo.training.distributed_resume_gate import _runtime_identity
        runtime = dict(_runtime_identity(torch))
    else:
        runtime = {"python": platform.python_version()}
    metadata = {**identity, "suite_sha256": canonical_sha256(suite), "optimizer_steps": 0,
                "runtime_identity": runtime}
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    with _shard_lock(output, shard, shards):
        _publish_shared(output / "candidate.json", metadata)
        from self_play_grpo.envs.quoridor import QuoridorEnv
        from self_play_grpo.rollouts.collector import MatchCollector
        candidate = _bot(bot_candidate) if bot_candidate is not None else None
        initial = None
        before = None
        if uses_llm:
            from self_play_grpo.policies.llm import ConstrainedLLMPolicy
            from self_play_grpo.rollouts.pilot import restore_initial_adapter
            from self_play_grpo.training.identity import trainable_parameter_sha256
            if adapter is not None:
                candidate = ConstrainedLLMPolicy.load(config.model, config.rollout)
                restore_initial_adapter(candidate.model, adapter, identity["adapter_sha256"])
                candidate.model.eval()
                candidate.name = identity["policy_version"]
                before = trainable_parameter_sha256(candidate.model)
            if any(game["opponent"] == "initial-llm" for game in selected):
                initial_adapter, initial_digest = _initial_adapter(args.run_root, config)
                initial = ConstrainedLLMPolicy.load(config.model, config.rollout)
                restore_initial_adapter(initial.model, initial_adapter, initial_digest)
                initial.model.eval()
                initial.name = "initial_llm"
        collector = MatchCollector(collect_progress=False)
        for game in selected:
            path = output / "games" / f"game-{game['index']:06d}.json"
            game_id = f"eval:{metadata['suite_sha256']}:{game['index']}:{metadata['policy_version']}"
            if path.exists():
                recorded = _json_file(path)
                if (recorded.get("candidate") != metadata or recorded.get("schedule") != game
                        or recorded["result"].get("game_id") != game_id):
                    raise ValueError("Existing evaluation game identity differs")
                continue
            opponents = [initial] * 3 if game["opponent"] == "initial-llm" else [_bot(game["opponent"]) for _ in range(3)]
            policies = list(opponents)
            policies.insert(game["seat"], candidate)
            if torch is not None:
                torch.manual_seed(game["seed"])
                if config.model.device.startswith("hpu"):
                    torch.hpu.manual_seed_all(game["seed"])
            env = QuoridorEnv(config.environment)
            if torch is not None:
                with torch.no_grad():
                    match = collector.collect(env, policies, game_id=game_id, seed=game["seed"],
                                              policy_version=metadata["policy_version"])
            else:
                match = collector.collect(env, policies, game_id=game_id, seed=game["seed"],
                                          policy_version=metadata["policy_version"])
            result = EvaluationGame(
                game_id=game_id, seed=game["seed"], candidate_name=metadata["policy_version"],
                candidate_seat=game["seat"], opponent_names=tuple(policy.name for policy in opponents),
                candidate_result=match.final_results[game["seat"]], final_results=match.final_results,
                termination_reason=match.termination_reason, joint_actions=len(match.turns),
            )
            if getattr(args, "save_matches", True):
                record = output / "matches" / f"game-{game['index']:06d}.jsonl"
                record.parent.mkdir(parents=True, exist_ok=True)
                pending = record.with_name(f".{record.name}.pending")
                pending.write_text(match.to_json() + "\n", encoding="utf-8")
                pending.replace(record)
            metrics = game_metrics(match, game["seat"])
            publish_json(path, json.loads(json.dumps(
                {"candidate": metadata, "schedule": game, "result": asdict(result), "metrics": metrics})))
            print(json.dumps({"event": "evaluation_game_complete", "index": game["index"],
                              "seat": game["seat"], "opponent": game["opponent"],
                              "result": result.candidate_result,
                              "final_distance": metrics["final_distance"]}), flush=True)
        if before is not None:
            from self_play_grpo.training.identity import trainable_parameter_sha256
            if trainable_parameter_sha256(candidate.model) != before:
                raise RuntimeError("Evaluation changed trainable parameters")
    done = all((output / "games" / f"game-{game['index']:06d}.json").exists() for game in suite["games"])
    if not done:
        return {"status": "shard_complete", "shard": shard, "num_shards": shards, "games": len(selected)}
    return summarize(suite, output)


def _summary(rows: Sequence[dict], seed: int) -> dict[str, Any]:
    results = [row["result"]["candidate_result"] for row in rows]
    distances = [row["metrics"]["final_distance"] for row in rows]
    moves = sum(row["metrics"]["moves"] for row in rows)
    turns = sum(row["metrics"]["turns"] for row in rows)
    by_seat = {}
    for seat in range(4):
        selected = [row["result"]["candidate_result"] for row in rows if row["schedule"]["seat"] == seat]
        by_seat[str(seat)] = sum(selected) / len(selected) if selected else None
    return {
        "games": len(rows),
        "mean_fractional_result": sum(results) / len(results),
        "result_95_interval": list(bootstrap_match_mean(results, seed=seed)),
        "win_rate": sum(value == 1.0 for value in results) / len(results),
        "draw_rate": sum(row["result"]["termination_reason"] != "natural_win" for row in rows) / len(rows),
        "mean_final_distance": sum(distances) / len(distances),
        "final_distance_95_interval": list(bootstrap_match_mean(distances, seed=seed)),
        "mean_placement": sum(row["metrics"]["placement"] for row in rows) / len(rows),
        "forward_move_rate": None if moves == 0 else sum(row["metrics"]["forward_moves"] for row in rows) / moves,
        "backward_move_rate": None if moves == 0 else sum(row["metrics"]["backward_moves"] for row in rows) / moves,
        "wall_rate": None if turns == 0 else sum(row["metrics"]["walls"] for row in rows) / turns,
        "mean_game_length": sum(row["result"]["joint_actions"] for row in rows) / len(rows),
        "result_by_seat": by_seat,
    }


def _load_games(suite: dict, output: Path) -> list[dict]:
    rows = []
    for game in suite["games"]:
        row = _json_file(output / "games" / f"game-{game['index']:06d}.json")
        if row.get("schedule") != game:
            raise ValueError(f"Evaluation game {game['index']} does not match the suite")
        rows.append(row)
    candidates = {canonical_sha256(row["candidate"]) for row in rows}
    if len(candidates) != 1:
        raise ValueError("Evaluation games come from different candidates")
    return rows


def summarize(suite: dict, output: Path) -> dict:
    """CPU-only merge of every shard's games into one summary."""

    rows = _load_games(suite, output)
    summary = {"status": "evaluation_complete", "candidate": rows[0]["candidate"],
               "suite_sha256": canonical_sha256(suite),
               "overall": _summary(rows, suite["seed"]),
               "by_opponent": {lineup: _summary([row for row in rows if row["schedule"]["opponent"] == lineup],
                                                suite["seed"]) for lineup in suite["lineups"]}}
    summary = json.loads(json.dumps(summary))
    _publish_shared(output / "summary.json", summary)
    return summary


def compare(suite: dict, baseline: Path, candidate: Path) -> dict:
    """Pair two evaluations of one suite game by game (candidate minus baseline)."""

    before, after = _load_games(suite, baseline), _load_games(suite, candidate)

    def paired(selected: list[int]) -> dict[str, Any]:
        result = [after[i]["result"]["candidate_result"] - before[i]["result"]["candidate_result"] for i in selected]
        distance = [after[i]["metrics"]["final_distance"] - before[i]["metrics"]["final_distance"] for i in selected]
        return {"games": len(selected),
                "result_difference": sum(result) / len(result),
                "result_difference_95_interval": list(bootstrap_match_mean(result, seed=suite["seed"])),
                "final_distance_difference": sum(distance) / len(distance),
                "final_distance_difference_95_interval": list(bootstrap_match_mean(distance, seed=suite["seed"]))}

    indices = list(range(len(suite["games"])))
    return {"baseline": before[0]["candidate"]["policy_version"],
            "candidate": after[0]["candidate"]["policy_version"],
            "overall": paired(indices),
            "by_opponent": {lineup: paired([i for i in indices if suite["games"][i]["opponent"] == lineup])
                            for lineup in suite["lineups"]}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="CPU-only immutable schedule creation")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--games-per-seat", type=int, default=8)
    freeze.add_argument("--seed", type=int, default=20260929)
    freeze.add_argument("--lineups", default=",".join(LINEUPS))
    run = sub.add_parser("run", help="Read-only evaluation of one checkpoint (one HPU per shard)")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--update", type=int)
    group.add_argument("--candidate-bot", choices=BOTS, help="CPU pipeline check with a scripted candidate")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--no-save-matches", dest="save_matches", action="store_false")
    merge = sub.add_parser("summarize", help="CPU-only merge of all shards")
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--suite", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    pair = sub.add_parser("compare", help="CPU-only paired comparison of two evaluations")
    pair.add_argument("--config", type=Path, required=True)
    pair.add_argument("--suite", type=Path, required=True)
    pair.add_argument("--baseline", type=Path, required=True)
    pair.add_argument("--candidate", type=Path, required=True)
    pair.add_argument("--output", type=Path, help="also write the comparison JSON here")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "freeze":
        suite = build_suite(config, games_per_seat=args.games_per_seat, seed=args.seed,
                            lineups=tuple(item for item in args.lineups.split(",") if item))
        publish_json(args.output, suite)
        print(json.dumps({"status": "suite_frozen", "games": len(suite["games"]),
                          "suite": str(args.output), "sha256": canonical_sha256(suite)}))
        return 0
    suite = _json_file(args.suite)
    _verify_suite(config, suite)
    if args.command == "run":
        result = evaluate(args, config, suite)
    elif args.command == "summarize":
        result = summarize(suite, args.output)
    else:
        result = compare(suite, args.baseline, args.candidate)
        if args.output is not None:
            publish_json(args.output, json.loads(json.dumps(result)))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
