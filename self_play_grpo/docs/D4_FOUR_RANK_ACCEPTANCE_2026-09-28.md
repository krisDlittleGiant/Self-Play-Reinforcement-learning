# D4 four-trainer acceptance — 2026-09-28

This note supersedes the pending-D4 status in
`D4_D5_GATE_AUDIT_2026-09-28.md` and
`D4_TWO_RANK_GATE_RESULT_2026-09-28.md`. Both D4 hardware gates are now
complete. This is validation-only checkpoint continuation, not an eight-HPU
GRPO update or a production checkpoint.

## Four-trainer command and result

The user ran the bounded `validate-distributed-checkpoint-resume` gate on
`gaudi002` with `HABANA_VISIBLE_MODULES=4,5,6,7`, four local processes,
the recorded D2 rollout, the D3 format-2 checkpoint, seed 741, and a fresh
`self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741` output.
The command and source paths are recorded in
`D4_TWO_RANK_GATE_RESULT_2026-09-28.md`.

The reported summary is
`self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741/summary.json`:

- `status: ok`, `validation_only: true`, `world_size: 4`;
- physical trainer modules 4, 5, 6, 7, one per rank, with successful HCCL
  membership and both reductions (expected and observed sum 10.0);
- each rank reports exact equality for CPU RNG, HPU RNG, scalar metrics,
  parameters, and optimizer state after uninterrupted versus reloaded
  continuation;
- all four rank reports identify the same published format-3 checkpoint
  manifest SHA-256:
  `8e95e34cdd17665a210b3ac0fef421bd24ae59c8fabfab303e24419d31965f67`;
- a validation-only `policy-000002` checkpoint with its format-3 manifest
  and four distinct rank RNG files is present under the output directory.

The gate's behavior-policy replay differences are expected when continuing
from the post-D3 policy. This gate requires exact uninterrupted/resumed
comparison, not the pre-update on-policy replay tolerance. The usual Habana
eager-mode, Apex, OpenSpiel, and collective-device warnings did not prevent
the successful summary. Do not use either D4 validation checkpoint as the
starting policy for production training.

## Independent read-only acceptance check

The project verifier was run against both actual D4 summaries and the
planned 4+4 module layout. It completed with exit code 0 and
`status: passed`, `scope: read_only_d4_evidence`, `trainer_modules: [4,5,6,7]`.
It checks all rank equality flags, checkpoint inventory and hashes, the
shared D2/D3 source identity, and the four-rank trainer-module mapping.
The exact command was:

```bash
bash env/shell.sh python -m self_play_grpo.training.d4_acceptance \
  --two-rank-summary self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741/summary.json \
  --four-rank-summary self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741/summary.json \
  --modules 0,1,2,3,4,5,6,7
```

## Next segment: guarded D5 one-update run

The next hardware test is one complete **production** outcome-reward GRPO
update on the already allocated eight HPUs: rollout workers on modules
0–3 and trainer workers on modules 4–7. The dedicated config is
`self_play_grpo/configs/quoridor_outcome_64games.yaml`, not the older
`quoridor_outcome.yaml` (which specifies only 16 games). The D5 config
specifies 64 complete games, two active games per rollout rank, four-seat
outcome credit, and one optimizer epoch. The recorded pilot bootstrap is
`self_play_grpo/artifacts/pilot-outcome-relative-shape-v1-qwen3-4b-seed-11`;
the initial-policy gate permits its 16-game pilot setting to differ from
the target 64-game production setting while requiring the same pinned model,
environment, and action-decoding contract.

`self_play_grpo.training.initial_cycle` is an opt-in **one-update** driver.
It will not start on import; it preflights both D4 summaries, the current
allocation and 64-game config before launching workers. It has not been
run on eight HPUs. Its output must be a new directory, and partial failures
are retained, not resumed automatically. The eventual five-update,
fresh-process restart test and learned reward model remain separate work.

No eight-HPU D5 launch was started while preparing this note.
