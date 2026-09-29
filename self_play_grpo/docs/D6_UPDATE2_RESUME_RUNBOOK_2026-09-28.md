# D6 update-2 continuation runbook (2026-09-28)

This is the first runnable cold-process continuation from the completed D5
`policy-000001` production checkpoint. It is a **single** additional 64-game
update, not the final five-update D6 restart gate. The code has CPU contract
tests; the real eight-HPU update-2 path has **not yet been run**.

## Scope and state safety

`training/d6_cycle.py` uses the existing D5 run root. It first audits the
hash-linked ledger and four-rank refresh evidence. It freezes the committed
adapter as `policy-000001` for fresh `rollout-000001`, collects 16 games on
each of rollout modules 0–3, and verifies the 64-game batch. Four trainer
processes on modules 4–7 load the complete format-3 update-1 checkpoint,
including AdamW state and per-rank CPU/HPU RNG, then take one synchronized
step. A new format-3 checkpoint is fingerprinted with the versioned D6 source
identity, committed as update 2, and probed on all four rollout modules.

Expected new files under the same run root:

- `rollout-000001/` — immutable 64-game source batch, policy-000001.
- `trainer-000002/trainer/checkpoints/policy-000002/` — production checkpoint.
- `commits/update-000002.json` — append-only batch-consumption record.
- `refresh-000002/` — four rank reports and verified refresh summary.
- `cycle_summary_000002.json` — final ready-state summary.

The prior D5 checkpoint, ledger record, and rollout are not overwritten. If
update-2 trainer or refresh output already exists, the command refuses to
retrain; inspect the state rather than deleting evidence. An empty prepared
`rollout-000001` may be used. A complete published batch is verified and
reused after an interruption. A partially collected rollout is preserved and
rejected, not silently replayed. Recovery from interruption inside update 2
is a later D6 task. The ledger is authoritative after checkpoint commit;
refresh failure must never cause that batch to be trained again.

## Preflight and launch

Run only from the repository root on the existing allocated eight-HPU node.
This command does not request a new Slurm allocation. First verify the node
and allocation, and that `trainer-000002`/`refresh-000002` do not already
exist. The driver also performs these fail-closed checks.

```bash
printf 'node=%s allocated_hpus=%s\n' "$(hostname)" "${SLURM_GPUS_ON_NODE:-unset}"
bash env/shell.sh python -m self_play_grpo.training.d6_recovery \
  --root self_play_grpo/artifacts/d5-one-update-64games-qwen3-4b-seed-11 \
  --config self_play_grpo/configs/quoridor_outcome_64games.yaml \
  --seed 11 --run-id d5-one-update-64games-qwen3-4b-seed-11
```

Expected audit status before launch: `ready_for_next_collection`, committed
update 1, next update 2. If this differs, do not launch.

```bash
time bash env/shell.sh python -m self_play_grpo.training.d6_cycle \
  --config self_play_grpo/configs/quoridor_outcome_64games.yaml \
  --run-root self_play_grpo/artifacts/d5-one-update-64games-qwen3-4b-seed-11 \
  --run-id d5-one-update-64games-qwen3-4b-seed-11 \
  --rollout-modules 0,1,2,3 \
  --trainer-modules 4,5,6,7 \
  --d4-two-summary self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741/summary.json \
  --d4-four-summary self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741/summary.json \
  --seed 11 --replay-tolerance 2e-4 \
  --trainer-master-port 29642 \
  --rollout-timeout-seconds 7200 \
  --trainer-timeout-seconds 10800 \
  --refresh-timeout-seconds 1800
```

The first D5 update took about 97 minutes, so this may also be long. Phase
logs appear in `logs-rollout-000001`, `logs-trainer-000002`, and
`logs-refresh-000002`. The launcher supervises process groups and requires
all four ranks to exit cleanly at each phase. Success is
`"status": "second_update_complete"`, policy-000002, two ledger records,
four refresh ranks, and finite trainer metrics. HPU memory and runtime for
this specific continuation are unknown until the real run.

## Verification after a successful run

```bash
bash env/shell.sh python -m self_play_grpo.training.d6_recovery \
  --root self_play_grpo/artifacts/d5-one-update-64games-qwen3-4b-seed-11 \
  --config self_play_grpo/configs/quoridor_outcome_64games.yaml \
  --seed 11 --run-id d5-one-update-64games-qwen3-4b-seed-11
```

Expected: committed update 2, `policy-000002`, next update 3, and
`ready_for_next_collection`. Do not interpret a successful loss or checkpoint
as a demonstrated gameplay-strength improvement; that needs held-out
evaluation. This command does not train a learned reward model.
