# D5 one-update launch handoff — 2026-09-28

D4 hardware acceptance has passed on the two- and four-trainer gates, as
recorded in `D4_FOUR_RANK_ACCEPTANCE_2026-09-28.md`. The user then ran the
read-only D5 preflight on `gaudi002` with the 64-game config, both real D4
summaries, and the 0–3 rollout / 4–7 trainer layout. It returned
`status: preflight_passed_not_launched`, `games_per_update: 64`,
`games_per_rollout_rank: 16`, and `parallel_games_per_rollout_rank: 2`.
No D5 worker or HPU run has yet been launched.

The intended next test is **one** complete production-path outcome-reward
GRPO update and four rollout refresh probes. It is not a five-update run
and does not involve the optional learned reward model. The initial policy
comes from the recorded, immutable pilot at
`self_play_grpo/artifacts/pilot-outcome-relative-shape-v1-qwen3-4b-seed-11`;
the D4 validation-only checkpoint is evidence, not the starting policy.

## Launch command (pending; run only on the already allocated eight-HPU node)

The proposed output path was absent and TCP port 29641 was not listening
when checked on `gaudi002`. These conditions can change; choose a different
fresh output path or free port if necessary. Do not delete a partial run.
The driver itself rechecks D4 evidence, allocation, config, output freshness,
and port range before starting any workers.

```bash
time bash env/shell.sh python -m self_play_grpo.training.initial_cycle \
  --config self_play_grpo/configs/quoridor_outcome_64games.yaml \
  --pilot-root self_play_grpo/artifacts/pilot-outcome-relative-shape-v1-qwen3-4b-seed-11 \
  --output self_play_grpo/artifacts/d5-one-update-64games-qwen3-4b-seed-11 \
  --run-id d5-one-update-64games-qwen3-4b-seed-11 \
  --rollout-modules 0,1,2,3 \
  --trainer-modules 4,5,6,7 \
  --d4-two-summary self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741/summary.json \
  --d4-four-summary self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741/summary.json \
  --seed 11 --replay-tolerance 2e-4 \
  --trainer-master-port 29641 \
  --rollout-timeout-seconds 7200 \
  --trainer-timeout-seconds 10800 \
  --refresh-timeout-seconds 1800
```

The three phase deadlines are two hours for rollout, three hours for
training, and 30 minutes for refresh. There is no external `timeout` wrapper:
the phase supervisor terminates its worker process groups on a deadline or
failure and preserves each worker log. The foreground parent may print little
while a phase runs because worker stdout/stderr is written to
`logs-rollout-000000/`, `logs-trainer-000001/`, or `logs-refresh-000001/`
inside the new run directory. Do not mistake quiet console output for a
completed phase. On failure the directory is diagnostic evidence and must
not be reused; the driver writes `cycle_failure.json` when it can. Success
requires `cycle_summary.json` with `status: one_update_complete`, 64 games,
policy `policy-000001`, one committed production checkpoint, four refresh
reports, and coordinator READY(1). A zero exit from a worker alone is not
sufficient.
