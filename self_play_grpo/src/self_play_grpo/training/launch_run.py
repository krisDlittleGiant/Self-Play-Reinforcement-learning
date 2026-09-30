"""One resumable command for a complete 4+4 self-play run with evaluations.

Stages, each skipped when its committed evidence already exists:

1. pilot: a two-game frozen-policy pilot that seeds policy-000000;
2. update 1 (``initial_cycle``) and update 2 (``d6_cycle``);
3. updates 3..N (``multi_run``) in segments that end at evaluation points;
4. evaluation of policy-000000 and every ``--eval-every`` updates, sharded
   over all eight HPUs, plus a paired comparison against policy-000000;
5. ``run_report.json``: per-update metrics, phase timings, batch indicators
   and evaluation summaries.

Re-running the same command after an interruption (for example an expired
allocation) continues from the last committed update. Interrupted, unpublished
work is moved to ``recovery_archive`` rather than deleted. The training code
must not change between invocations: checkpoints bind to its source hashes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from self_play_grpo.config import load_config
from self_play_grpo.rollouts.pilot import read_pilot_manifest
from self_play_grpo.training.ledger import read_commits


def event(name: str, **values: Any) -> None:
    print(json.dumps({"event": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **values},
                     sort_keys=True, default=str), flush=True)


def eval_points(until_update: int, every: int) -> list[int]:
    if every <= 0:
        return []
    return sorted({0, until_update, *range(every, until_update + 1, every)})


def committed_updates(root: Path) -> int:
    """-1 when the run root does not exist yet."""

    if not root.exists():
        return -1
    if not (root / "commits").exists():
        return 0
    return len(read_commits(root))


@dataclass(frozen=True)
class Step:
    kind: str  # pilot | archive_root | update1 | update2 | evaluate | multi | done
    target: int = 0


def next_step(args: Any, *, pilot_ready: bool, commits: int, evaluated: set[int]) -> Step:
    """Pure decision of the next stage; see the module docstring."""

    if not pilot_ready and commits < 1:
        return Step("pilot")
    for point in eval_points(args.until_update, args.eval_every):
        if point <= max(commits, 0) and commits >= 1 and point not in evaluated:
            return Step("evaluate", point)
    if commits >= args.until_update:
        return Step("done", commits)
    if commits == -1:
        return Step("update1", 1)
    if commits == 0:
        return Step("archive_root")
    if commits == 1:
        return Step("update2", 2)
    later = [point for point in eval_points(args.until_update, args.eval_every) if point > commits]
    target = min(later) if later else args.until_update
    return Step("multi", max(3, min(target, args.until_update)))


def _driver_arguments(args: Any) -> list[str]:
    return ["--config", str(args.config), "--run-id", args.run_id,
            "--rollout-modules", args.rollout_modules, "--trainer-modules", args.trainer_modules,
            "--d4-two-summary", str(args.d4_two_summary), "--d4-four-summary", str(args.d4_four_summary),
            "--seed", str(args.seed), "--replay-tolerance", str(args.replay_tolerance),
            "--rollout-timeout-seconds", str(args.rollout_timeout_seconds),
            "--trainer-timeout-seconds", str(args.trainer_timeout_seconds),
            "--refresh-timeout-seconds", str(args.refresh_timeout_seconds)]


def _run(args: Any, name: str, command: list[str], *, env: dict[str, str] | None = None,
         log: Path | None = None) -> float:
    event("step_started", step=name, command=" ".join(command))
    start = time.monotonic()
    if log is None:
        result = subprocess.run(command, env=env)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    seconds = time.monotonic() - start
    record = {"step": name, "seconds": seconds, "returncode": result.returncode,
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.run_root.exists():
        with (args.run_root / "launcher_timings.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    event("step_finished", **record)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see its log above")
    return seconds


def _pilot_ready(args: Any) -> bool:
    try:
        manifest = read_pilot_manifest(args.pilot_root)
    except FileNotFoundError:
        return False
    return len(manifest.matches) >= args.pilot_games


def _archive(root: Path, path: Path) -> None:
    from self_play_grpo.training.multi_run import archive
    archive(root, path)


def _prepare_update2(root: Path) -> None:
    """Move aside unpublished, interrupted update-2 work before ``d6_cycle``."""

    refresh = root / "refresh-000001" / "refresh_summary.json"
    if not refresh.is_file():
        raise RuntimeError(
            "Update 1 is committed but its refresh never completed. The update-1 driver cannot "
            "resume a refresh; inspect logs-refresh-000001 before continuing.")
    trainer = root / "trainer-000002"
    if trainer.exists():
        if list((trainer / "trainer" / "checkpoints").glob("policy-*")):
            raise RuntimeError("trainer-000002 holds a published but uncommitted checkpoint; inspect it")
        _archive(root, trainer)
    rollout = root / "rollout-000001"
    if rollout.exists() and len(read_pilot_manifest(rollout).matches) < 64:
        _archive(root, rollout)


def _evaluated(args: Any) -> set[int]:
    done = set()
    for point in eval_points(args.until_update, args.eval_every):
        if (args.run_root / "eval" / f"policy-{point:06d}" / "summary.json").is_file():
            done.add(point)
    return done


def _evaluate(args: Any, update: int) -> None:
    module = [sys.executable, "-m", "self_play_grpo.evaluation.fixed_checkpoint"]
    suite = args.run_root / "eval" / "suite.json"
    if not suite.exists():
        _run(args, "freeze_eval_suite", [*module, "freeze", "--config", str(args.config),
                                         "--output", str(suite),
                                         "--games-per-seat", str(args.eval_games_per_seat),
                                         "--seed", str(args.eval_seed)])
    output = args.run_root / "eval" / f"policy-{update:06d}"
    modules = [*args.rollout_modules.split(","), *args.trainer_modules.split(",")]
    event("evaluation_started", update=update, shards=len(modules))
    start = time.monotonic()
    processes = []
    for shard, device in enumerate(modules):
        env = {**os.environ, "HABANA_VISIBLE_MODULES": device}
        log = output / "logs" / f"shard-{shard:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        command = [*module, "run", "--config", str(args.config), "--suite", str(suite),
                   "--run-root", str(args.run_root), "--update", str(update), "--output", str(output),
                   "--shard-index", str(shard), "--num-shards", str(len(modules))]
        processes.append((shard, subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
    failed = []
    for shard, process, handle in processes:
        if process.wait() != 0:
            failed.append(shard)
        handle.close()
    seconds = time.monotonic() - start
    with (args.run_root / "launcher_timings.jsonl").open("a", encoding="utf-8") as timings:
        timings.write(json.dumps({"step": f"evaluate_{update:06d}", "seconds": seconds,
                                  "returncode": 1 if failed else 0}, sort_keys=True) + "\n")
    if failed:
        raise RuntimeError(f"Evaluation shards {failed} failed; see {output / 'logs'}")
    _run(args, f"summarize_eval_{update:06d}", [*module, "summarize", "--config", str(args.config),
                                                "--suite", str(suite), "--output", str(output)],
         log=output / "logs" / "summarize.log")
    if update != 0:
        comparison = output / "compare_with_policy-000000.json"
        _run(args, f"compare_eval_{update:06d}", [*module, "compare", "--config", str(args.config),
                                                  "--suite", str(suite),
                                                  "--baseline", str(args.run_root / "eval" / "policy-000000"),
                                                  "--candidate", str(output), "--output", str(comparison)],
             log=output / "logs" / "compare.log")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    event("evaluation_complete", update=update, seconds=seconds,
          by_opponent={name: {key: row[key] for key in ("mean_fractional_result", "win_rate",
                                                        "mean_final_distance", "mean_placement")}
                       for name, row in summary["by_opponent"].items()})


def write_report(root: Path) -> dict[str, Any]:
    """CPU-only aggregation of everything a run has produced so far."""

    from self_play_grpo.rollouts.indicators import batch_indicators, read_match_directory

    updates = []
    for record in read_commits(root) if (root / "commits").exists() else ():
        u = record.update_index
        name = ("cycle_summary.json" if u == 1 else "cycle_summary_000002.json" if u == 2
                else f"multi_cycle_summary_{u:06d}.json")
        row: dict[str, Any] = {"update": u, "policy_version": record.policy_version}
        if (root / name).is_file():
            row["trainer_metrics"] = json.loads((root / name).read_text(encoding="utf-8")).get("trainer_metrics")
        rank = root / f"trainer-{u:06d}" / "ranks" / "rank-000.json"
        if rank.is_file():
            update = json.loads(rank.read_text(encoding="utf-8")).get("update", {})
            row["minibatches"] = update.get("minibatches", [])
            row["seat_result_means"] = update.get("seat_result_means", [])
        updates.append(row)
    timings = []
    for path in sorted(root.glob("logs-multi-*/timing.json")):
        timings.append(json.loads(path.read_text(encoding="utf-8")))
    launcher = []
    if (root / "launcher_timings.jsonl").is_file():
        launcher = [json.loads(line) for line in (root / "launcher_timings.jsonl").read_text().splitlines() if line]
    indicators = []
    for rollout in sorted(root.glob("rollout-[0-9]*")):
        manifest = read_pilot_manifest(rollout)
        if len(manifest.matches) == manifest.target_games:
            indicators.append({"rollout": rollout.name, **batch_indicators(read_match_directory(rollout))})
    evaluations = {}
    for summary in sorted(root.glob("eval/policy-*/summary.json")):
        entry = json.loads(summary.read_text(encoding="utf-8"))
        comparison = summary.parent / "compare_with_policy-000000.json"
        evaluations[summary.parent.name] = {
            "overall": entry["overall"], "by_opponent": entry["by_opponent"],
            "compare_with_policy_000000": json.loads(comparison.read_text()) if comparison.is_file() else None,
        }
    report = {"run_root": str(root), "updates": updates, "phase_timings": timings,
              "launcher_timings": launcher, "indicators": indicators, "evaluations": evaluations}
    (root / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def print_report(report: dict[str, Any]) -> None:
    keys = ("mean_game_length", "draw_rate", "forward_move_rate", "first_listed_rate",
            "wall_rate", "entropy_estimate_nats")
    print("rollout      " + "  ".join(f"{key[:14]:>14s}" for key in keys) + "  wins_by_seat")
    for row in report["indicators"]:
        print(f"{row['rollout'][-6:]:12s} " + "  ".join(
            f"{row[key]:14.3f}" if isinstance(row[key], float) else f"{str(row[key]):>14s}" for key in keys)
            + f"  {row['wins_by_seat']}")
    for name, entry in report["evaluations"].items():
        print(name, {lineup: (round(row["mean_fractional_result"], 3), round(row["mean_final_distance"], 2))
                     for lineup, row in entry["by_opponent"].items()})


def run(args: Any) -> dict[str, Any]:
    args.config = args.config.resolve()
    args.run_root = args.run_root.resolve()
    args.pilot_root = args.pilot_root.resolve()
    config = load_config(args.config)
    if config.rollout.games_per_update != 64 or args.until_update < 1:
        raise ValueError("The 4+4 launcher needs the 64-game profile and --until-update >= 1")
    base = args.trainer_master_port
    python = [sys.executable, "-m"]
    while True:
        commits = committed_updates(args.run_root)
        step = next_step(args, pilot_ready=_pilot_ready(args), commits=commits, evaluated=_evaluated(args))
        event("next_step", step=step.kind, target=step.target, committed_update=max(commits, 0))
        if args.dry_run:
            return {"status": "dry_run", "next_step": step.kind, "target": step.target}
        if step.kind == "done":
            break
        if step.kind == "pilot":
            env = {**os.environ, "HABANA_VISIBLE_MODULES": args.rollout_modules.split(",")[0]}
            _run(args, "pilot", [*python, "self_play_grpo.cli", "collect-policy-pilot",
                                 "--config", str(args.config), "--output", str(args.pilot_root),
                                 "--games", str(args.pilot_games), "--seed", str(args.seed),
                                 "--parallel-games", "2"], env=env)
        elif step.kind == "archive_root":
            target = args.run_root.with_name(f"{args.run_root.name}.interrupted-{uuid.uuid4().hex[:8]}")
            args.run_root.rename(target)
            event("archived_uncommitted_run_root", archive=str(target))
        elif step.kind == "update1":
            _run(args, "update_000001", [*python, "self_play_grpo.training.initial_cycle",
                                         *_driver_arguments(args), "--pilot-root", str(args.pilot_root),
                                         "--output", str(args.run_root),
                                         "--trainer-master-port", str(base)])
        elif step.kind == "update2":
            _prepare_update2(args.run_root)
            _run(args, "update_000002", [*python, "self_play_grpo.training.d6_cycle",
                                         *_driver_arguments(args), "--run-root", str(args.run_root),
                                         "--trainer-master-port", str(base + 1)])
        elif step.kind == "evaluate":
            _evaluate(args, step.target)
        elif step.kind == "multi":
            _run(args, f"updates_to_{step.target:06d}", [*python, "self_play_grpo.training.multi_run",
                                                         *_driver_arguments(args),
                                                         "--run-root", str(args.run_root),
                                                         "--until-update", str(step.target),
                                                         "--trainer-master-port", str(base + 2)])
        if args.run_root.exists():
            try:
                write_report(args.run_root)
            except Exception as exc:  # A report problem must never stop training.
                event("report_failed", error=str(exc))
    report = write_report(args.run_root)
    print_report(report)
    return {"status": "run_complete", "committed_update": committed_updates(args.run_root),
            "report": str(args.run_root / "run_report.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--pilot-games", type=int, default=2)
    parser.add_argument("--until-update", type=int, required=True)
    parser.add_argument("--eval-every", type=int, default=2, help="0 disables evaluation")
    parser.add_argument("--eval-games-per-seat", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20260929)
    parser.add_argument("--rollout-modules", default="0,1,2,3")
    parser.add_argument("--trainer-modules", default="4,5,6,7")
    parser.add_argument("--d4-two-summary", required=True, type=Path)
    parser.add_argument("--d4-four-summary", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--replay-tolerance", type=float, default=2e-4)
    parser.add_argument("--trainer-master-port", type=int, default=29660,
                        help="update 1 uses this port, update 2 the next, later updates the one after")
    parser.add_argument("--rollout-timeout-seconds", type=int, default=7200)
    parser.add_argument("--trainer-timeout-seconds", type=int, default=10800)
    parser.add_argument("--refresh-timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true", help="print the next stage and exit")
    parser.add_argument("--report-only", action="store_true", help="rebuild run_report.json (CPU) and exit")
    args = parser.parse_args(argv)
    try:
        if args.report_only:
            report = write_report(args.run_root.resolve())
            print_report(report)
            return 0
        print(json.dumps(run(args), sort_keys=True), flush=True)
    except BaseException as exc:
        event("launcher_stopped", error_type=type(exc).__name__, error=str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
