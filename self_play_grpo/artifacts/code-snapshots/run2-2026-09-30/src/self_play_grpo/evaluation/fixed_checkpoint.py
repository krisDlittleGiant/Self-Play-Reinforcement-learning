"""Freeze a seat-balanced bot suite and evaluate immutable actor adapters.

Evaluation runs in a separate process and never constructs a trainer/optimizer.
The first suite uses the ordinary initial board; opening generalization and a
frozen initial-Qwen opponent are later evaluation extensions.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from self_play_grpo.config import load_config
from self_play_grpo.evaluation.tournament import EvaluationGame, TournamentRunner
from self_play_grpo.rollouts.pilot import canonical_sha256, directory_sha256, file_sha256, read_pilot_manifest
from self_play_grpo.training.d6_recovery import _json_file
from self_play_grpo.training.ledger import read_commits
from self_play_grpo.training.multi_evidence import publish_json


def build_suite(config, *, games_per_seat: int, seed: int) -> dict:
    if type(games_per_seat) is not int or games_per_seat <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Evaluation game count and seed must be valid")
    package = Path(__file__).resolve().parents[1]
    files = ("evaluation/fixed_checkpoint.py", "evaluation/tournament.py", "policies/bots.py",
             "policies/llm.py", "envs/quoridor.py", "rollouts/collector.py")
    games = []
    for opponent in ("random", "shortest", "wall-aware"):
        for seat in range(4):
            for repeat in range(games_per_seat):
                identity = {"namespace": "self_play_grpo/evaluation/v1", "seed": seed,
                            "opponent": opponent, "seat": seat, "repeat": repeat}
                game_seed = int(canonical_sha256(identity)[:15], 16)
                games.append({"index": len(games), "opponent": opponent, "seat": seat,
                              "repeat": repeat, "seed": game_seed})
    return {"schema_version": 1, "config_sha256": canonical_sha256(config.to_dict()),
            "seed": seed, "games_per_seat_per_opponent": games_per_seat,
            "source_sha256": {name: file_sha256(package / name) for name in files},
            "wall_aware_parameters": {"opponent_weight": 0.25, "wall_cost": 0.05},
            "initial_board": "standard", "games": games}


def candidate_identity(root: Path, config, update: int):
    if type(update) is not int or update < 0:
        raise ValueError("Candidate update must be non-negative")
    if update == 0:
        manifest = read_pilot_manifest(root / "rollout-000000")
        if manifest.config_sha256 != canonical_sha256(config.to_dict()):
            raise ValueError("Initial policy config differs")
        adapter = root / "rollout-000000" / "policy_adapter"
        expected = manifest.adapter_sha256
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


def evaluate(args, config, suite: dict) -> dict:
    # Validate the entire frozen schedule before loading any model.
    expected = build_suite(config, games_per_seat=suite["games_per_seat_per_opponent"], seed=suite["seed"])
    if suite != expected:
        raise ValueError("Frozen evaluation suite differs from the current code/config")
    adapter, identity = candidate_identity(args.run_root, config, args.update)
    import torch
    from self_play_grpo.envs.quoridor import QuoridorEnv
    from self_play_grpo.policies.bots import RandomPolicy, ShortestPathPolicy, WallAwarePolicy
    from self_play_grpo.policies.llm import ConstrainedLLMPolicy
    from self_play_grpo.rollouts.collector import MatchCollector
    from self_play_grpo.rollouts.pilot import restore_initial_adapter
    from self_play_grpo.training.identity import trainable_parameter_sha256
    from self_play_grpo.training.distributed_resume_gate import _runtime_identity
    metadata = {**identity, "suite_sha256": canonical_sha256(suite), "optimizer_steps": 0,
                "runtime_identity": dict(_runtime_identity(torch))}
    args.output.mkdir(parents=True, exist_ok=True)
    from self_play_grpo.training.multi_run import run_lock
    with run_lock(args.output):
        publish_json(args.output / "candidate.json", metadata)
        candidate = ConstrainedLLMPolicy.load(config.model, config.rollout)
        restore_initial_adapter(candidate.model, adapter, identity["adapter_sha256"])
        candidate.model.eval()
        candidate.name = identity["policy_version"]
        before = trainable_parameter_sha256(candidate.model)
        collector = MatchCollector(collect_progress=False)
        results = []
        for game in suite["games"]:
            path = args.output / "games" / f"game-{game['index']:06d}.json"
            game_id = f"eval:{metadata['suite_sha256']}:{game['index']}:{candidate.name}"
            if path.exists():
                recorded = _json_file(path)
                if (recorded.get("candidate") != metadata or recorded.get("schedule") != game
                        or recorded["result"].get("game_id") != game_id):
                    raise ValueError("Existing evaluation game identity differs")
                result = EvaluationGame(**recorded["result"])
            else:
                constructor = {"random": RandomPolicy, "shortest": ShortestPathPolicy,
                               "wall-aware": WallAwarePolicy}[game["opponent"]]
                opponents = [constructor() for _ in range(3)]
                policies = list(opponents)
                policies.insert(game["seat"], candidate)
                torch.manual_seed(game["seed"])
                if config.model.device.startswith("hpu"):
                    torch.hpu.manual_seed_all(game["seed"])
                with torch.no_grad():
                    match = collector.collect(QuoridorEnv(config.environment), policies, game_id=game_id,
                                              seed=game["seed"], policy_version=candidate.name)
                result = EvaluationGame(
                    game_id=game_id, seed=game["seed"], candidate_name=candidate.name,
                    candidate_seat=game["seat"], opponent_names=tuple(bot.name for bot in opponents),
                    candidate_result=match.final_results[game["seat"]], final_results=match.final_results,
                    termination_reason=match.termination_reason, joint_actions=len(match.turns),
                )
                publish_json(path, {"candidate": metadata, "schedule": game, "result": asdict(result)})
                print(json.dumps({"event": "evaluation_game_complete", "index": game["index"],
                                  "seat": game["seat"], "opponent": game["opponent"],
                                  "result": result.candidate_result}), flush=True)
            results.append(result)
        if trainable_parameter_sha256(candidate.model) != before:
            raise RuntimeError("Evaluation changed trainable parameters")
        overall = asdict(TournamentRunner.summarize(results, bootstrap_seed=suite["seed"]))
        by_opponent = {}
        for opponent in ("random", "shortest", "wall-aware"):
            selected = [result for game, result in zip(suite["games"], results) if game["opponent"] == opponent]
            by_opponent[opponent] = asdict(TournamentRunner.summarize(selected, bootstrap_seed=suite["seed"]))
        summary = {"status": "evaluation_complete", "candidate": metadata, "overall": overall,
                   "by_opponent": by_opponent, "parameters_unchanged": True}
        # Normalize tuple and integer-key fields to their JSON representation.
        summary = json.loads(json.dumps(summary))
        publish_json(args.output / "summary.json", summary)
        return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="CPU-only immutable schedule creation")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--games-per-seat", type=int, default=4)
    freeze.add_argument("--seed", type=int, default=20260929)
    run = sub.add_parser("run", help="Read-only checkpoint evaluation on one HPU")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--update", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "freeze":
        suite = build_suite(config, games_per_seat=args.games_per_seat, seed=args.seed)
        publish_json(args.output, suite)
        print(json.dumps({"status": "suite_frozen", "games": len(suite["games"]),
                          "suite": str(args.output), "sha256": canonical_sha256(suite)}))
    else:
        print(json.dumps(evaluate(args, config, _json_file(args.suite)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
