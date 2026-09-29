# D6 restart boundary: implementation and status (2026-09-28)

The first integrated D5 update completed on the 8-HPU gaudi002 allocation. It
collected 64 games on rollout modules 0–3, ran one synchronized optimizer step
on trainer modules 4–7, published a format-3 production checkpoint, committed
one ledger record, and verified refresh on all four rollout ranks. The run was
`d5-one-update-64games-qwen3-4b-seed-11`; see
`D5_ONE_UPDATE_RESULT_2026-09-28.md` for measured metrics and artifact paths.

## Implemented in this segment

`training/d6_recovery.py` is a **read-only, CPU-only** recovery audit. It reads
the entire hash-linked production ledger, verifies every referenced checkpoint
through the existing ledger contract, checks the head checkpoint against the
active config/model/source-code identity and four-trainer topology, checks the
head's consumed rollout manifest, and distinguishes a committed checkpoint
from a refresh-complete checkpoint. It reports the next logical update and
collection index, the consumed batch digest set, and a stable new collection
seed. An absent refresh returns `refresh_required` (exit code 2); malformed or
contradictory evidence fails. It does not start a rollout or trainer.

The seed derivation is SHA-256 over a versioned namespace, run ID, experiment
seed, and collection index. It does **not** claim exact per-game reproducibility
yet; the rollout worker's per-game RNG/scheduler behavior still needs a D6
reproducibility test. Policy version plus collection index must remain part of
new game IDs. The existing ledger already rejects repeated source-manifest
digests and discontinuous update indices.

Read-only audit command for the completed run:

```bash
bash env/shell.sh python -m self_play_grpo.training.d6_recovery \
  --root self_play_grpo/artifacts/d5-one-update-64games-qwen3-4b-seed-11 \
  --config self_play_grpo/configs/quoridor_outcome_64games.yaml \
  --seed 11 \
  --run-id d5-one-update-64games-qwen3-4b-seed-11
```

Observed: `ready_for_next_collection`, committed update 1,
`policy-000001`, next update 2, next collection index 1, checkpoint manifest
SHA-256 `53ca9b7ca0afecdfe1c3748bd0dfa33e3cbd8a222697a462142b6723d2e7a115`,
and consumed rollout manifest SHA-256
`ecd20330220924ae34e87c6ebd0325bdc02f759ed0405274a055b4fe7b84c06b`.
The next base seed reported by this exact code and run ID is
`600283395805802079`.

CPU test command:

```bash
bash env/shell.sh python -m pytest self_play_grpo/tests/test_d6_recovery.py -q
```

Observed: 5 passed. These tests cover complete versus absent refresh, a stable
seed, a missing/tampered rank report, mismatched run ID/source batch, and bad
seed input. This is a metadata gate, not a cold-start optimizer test.

## Still required before a five-update run

1. A D6 trainer worker must load the preceding **production** format-3
   checkpoint using `SynchronousTrainer.load_checkpoint` with exact distributed
   identity and per-rank RNG restoration, then advance from update `n` to
   `n+1`. The current `trainer_worker.py` explicitly accepts only
   `policy-000000` and hardcodes `policy-000001`.
2. The integrated driver must loop at committed boundaries, consume a fresh
   64-game manifest for each update, publish a new checkpoint/ledger record,
   refresh all four rollout workers, and admit only the newly verified policy.
   It must recover a committed-but-unrefreshed update without training it again.
3. Add CPU fake-worker tests for interruption during collection, after batch
   publication, during update, after checkpoint commit, and during refresh.
   Then run a bounded real-worker termination/restart test.
4. Run D6-A cold-process identical-data optimizer/RNG continuation, and D6-B
   three updates, terminate all workers, then two updates in fresh processes.
   Record unique batch IDs, policy versions, optimizer counters, and evaluation
   isolation. No five-update HPU run has been launched or passed yet.

The original D6 plan's five updates / 80 games assumes 16 games **total** per
update. The verified D5 production profile is 16 games **per rollout rank** on
four ranks, i.e. 64 games/update. Keeping this profile means five updates / 320
training games; do not silently change batch size to make the old 80-game
count fit.

No learned reward model is implemented here. The current reward remains the
configured game-outcome signal. Strength improvement requires separate
evaluation, not merely a successful optimizer step.
