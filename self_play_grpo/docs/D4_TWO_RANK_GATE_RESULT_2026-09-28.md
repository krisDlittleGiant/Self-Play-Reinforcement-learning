# D4 two-trainer continuation result — 2026-09-28

This result supersedes the initial “no D4 output found” status in
`D4_D5_GATE_AUDIT_2026-09-28.md`. It records the user-run two-trainer gate;
the four-trainer D4 acceptance gate is still pending. No HPU run was started
by the documentation update, and no additional allocation was requested.

## Command and inputs

The user reported `gaudi002`, `SLURM_GPUS_ON_NODE=8`, and
`HABANA_VISIBLE_MODULES` unset before running the bounded gate. The command
used physical trainer modules 4 and 5:

```bash
HABANA_VISIBLE_MODULES=4,5 timeout --signal=TERM --kill-after=30s 35m \
  bash env/shell.sh python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  -m self_play_grpo.cli validate-distributed-checkpoint-resume \
  --config self_play_grpo/configs/quoridor_outcome.yaml \
  --source-rollout self_play_grpo/artifacts/distributed-gameplay-4rollout-hpu-qwen3-4b-seed-11 \
  --source-checkpoint self_play_grpo/artifacts/distributed-update-4trainer-hpu-qwen3-4b-seed-11-attention-checkpoint-v1/checkpoints/policy-000001 \
  --output self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741 \
  --expected-world-size 2 --seed 741 --timeout-seconds 1800 \
  --allow-validation-source
```

## Observed result

- Summary: `self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741/summary.json`.
- `status: ok`, `validation_only: true`, `world_size: 2`.
- Rank-to-physical-module binding: rank 0 → 4, rank 1 → 5. Both HCCL
  membership checks and reductions passed (reported sum 3.0).
- Both ranks reported exact CPU RNG, HPU RNG, metrics, trainable parameters,
  and optimizer-state continuation equality. Their checkpoint manifest
  digest matched the summary.
- A format-3 validation checkpoint was published at
  `self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741/checkpoints/policy-000002`.
  The manifest declares `checkpoint_format: 3`, `trainer_world_size: 2`,
  `run_kind: validation`, rank-specific RNG files, and the D2 rollout-manifest
  digest. The summary's manifest SHA-256 is
  `537b1e382413daf018b33a24c0c75fbcac801dee88ffe5bac9978657c8396428`.
  The manifest and inventory files were present when inspected.
- The first/continuation steps reported nonzero behavior-policy replay
  differences. That is expected here: the source is the **post-D3** policy.
  D4 compares uninterrupted and reloaded continuation exactly; it does not
  enforce the pre-update on-policy replay tolerance on this bridge.
- The output contained the usual Habana eager-mode, Apex, and OpenSpiel
  warnings; no traceback or failed rank was reported.

This establishes the bounded two-trainer hardware gate, not four-trainer D4
acceptance, an eight-HPU training cycle, or a production policy.

## Next pending command: four-trainer D4 acceptance

Use the same already assigned eight-HPU node, with trainer modules 4–7. The
output directory below must be new; if it already exists, choose another
name rather than deleting evidence. This command is recorded for the user to
run; it has **not** been executed as part of this note.

```bash
HABANA_VISIBLE_MODULES=4,5,6,7 timeout --signal=TERM --kill-after=30s 35m \
  bash env/shell.sh python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 \
  -m self_play_grpo.cli validate-distributed-checkpoint-resume \
  --config self_play_grpo/configs/quoridor_outcome.yaml \
  --source-rollout self_play_grpo/artifacts/distributed-gameplay-4rollout-hpu-qwen3-4b-seed-11 \
  --source-checkpoint self_play_grpo/artifacts/distributed-update-4trainer-hpu-qwen3-4b-seed-11-attention-checkpoint-v1/checkpoints/policy-000001 \
  --output self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741 \
  --expected-world-size 4 --seed 741 --timeout-seconds 1800 \
  --allow-validation-source
```

After a successful four-rank summary, run the read-only
`python -m self_play_grpo.training.d4_acceptance` check described in
`D4_D5_GATE_AUDIT_2026-09-28.md`. Its two summary paths must point to the
actual successful outputs. Only then consider the guarded D5 one-update
driver, still subject to its eight-HPU allocation and 64-game preflight.
