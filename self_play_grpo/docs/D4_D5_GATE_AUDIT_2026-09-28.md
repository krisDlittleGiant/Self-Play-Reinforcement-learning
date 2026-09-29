# D4/D5 gate audit — 2026-09-28

This note records the current evidence for the next hardware gate. It takes
precedence over older **status** statements in the D4 design plan and D5
one-update runbook; those documents remain useful for their design contracts.
No HPU job was started for this audit, and no allocation was requested.

## What is verified

- The user ran `bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not model' -q`
  after the D5 initial-cycle integration and pilot-bootstrap changes. Result:
  **222 passed, 3 existing Habana warnings, 19.55 s**. This is CPU/non-model
  verification, not an HPU acceptance result.
- Both `d5_initial_cycle_integration_v2.patch` and
  `d5_pilot_bootstrap_driver_v2.patch` are **already applied** in the working
  tree. Their forward `git apply --unidiff-zero --check` fails because the old
  lines are gone; their combined `git apply --unidiff-zero --reverse --check`
  succeeds. Do not apply them again.
- The recorded 16-game distributed rollout exists at
  `self_play_grpo/artifacts/distributed-gameplay-4rollout-hpu-qwen3-4b-seed-11/`.
  Its `manifest.json`, four rank reports, policy adapter, and 16 match files
  were present at audit time. This is the D4 `--source-rollout` **directory**,
  not the manifest-file path.
- The D3 validation-only four-trainer update exists at
  `self_play_grpo/artifacts/distributed-update-4trainer-hpu-qwen3-4b-seed-11-attention-checkpoint-v1/`.
  Its `summary.json` reports `status: ok`, 16 games, 892 turns, 3,658 owned
  tokens, one gradient synchronization, one optimizer step, zero maximum
  behavior replay error, and a `policy-000001` checkpoint. The checkpoint is
  format 2 and validation-only. Its directory is the D4
  `--source-checkpoint` value:
  `self_play_grpo/artifacts/distributed-update-4trainer-hpu-qwen3-4b-seed-11-attention-checkpoint-v1/checkpoints/policy-000001`.
- The implemented D4 CLI is
  `python -m self_play_grpo.cli validate-distributed-checkpoint-resume`.
  It takes `--config`, `--source-rollout`, `--source-checkpoint`, `--output`,
  `--expected-world-size`, and `--allow-validation-source`, with optional
  `--joint-step`, `--match-offset`, `--seed`, and `--timeout-seconds`.
  The older plan's `sp-grpo --trainer-hpus --source-rollout-manifest` example
  is illustrative and does **not** match this implemented interface.

## What is not verified

No D4 two-rank or four-rank acceptance `summary.json`, and no format-3
distributed checkpoint manifest, was found under `self_play_grpo/artifacts`
when all files (including ignored files) were enumerated. This does not prove
that results do not exist elsewhere; provide their paths if they do.
The D3 checkpoint is **not** a substitute for D4 continuation evidence.
The D5 one-update driver has not been run on eight HPUs, and the 222 tests
do not establish an eight-HPU allocation.

## Next gate and recorded command shape

Run the bounded D4 two-trainer continuation first, then the four-trainer
continuation, using fresh output directories. These are **pending commands**,
not commands executed in this audit. They require the intended trainer HPUs
to be genuinely assigned to the current process. On the planned 0–3 rollout,
4–7 trainer layout, the trainer module sets are `4,5` and `4,5,6,7`.
Do not treat `hl-smi` or `/dev/accel` host visibility alone as allocation
proof, do not reserve extra HPUs, and do not start either gate in a known
one-HPU allocation. Use the existing assigned node only.

```bash
HABANA_VISIBLE_MODULES=4,5 bash env/shell.sh python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  -m self_play_grpo.cli validate-distributed-checkpoint-resume \
  --config self_play_grpo/configs/quoridor_outcome.yaml \
  --source-rollout self_play_grpo/artifacts/distributed-gameplay-4rollout-hpu-qwen3-4b-seed-11 \
  --source-checkpoint self_play_grpo/artifacts/distributed-update-4trainer-hpu-qwen3-4b-seed-11-attention-checkpoint-v1/checkpoints/policy-000001 \
  --output self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741 \
  --expected-world-size 2 --seed 741 --timeout-seconds 1800 \
  --allow-validation-source
```

Only after its summary and rank evidence pass, run the same command with
`HABANA_VISIBLE_MODULES=4,5,6,7`, `--nproc-per-node=4`,
`--expected-world-size 4`, and a **different new** output directory such as
`self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741`.
If either proposed output directory exists, choose a new name; never remove
or overwrite diagnostic artifacts. The D4 gate uses one saved action per
rank and synthetic credit solely to verify exact optimizer/RNG continuation.
It does not train a production policy or collect new games.

After both gates succeed, use the read-only acceptance check with the two
actual `summary.json` paths and `--modules 0,1,2,3,4,5,6,7`:

```bash
bash env/shell.sh python -m self_play_grpo.training.d4_acceptance \
  --two-rank-summary self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741/summary.json \
  --four-rank-summary self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741/summary.json \
  --modules 0,1,2,3,4,5,6,7
```

These exact summary paths become valid only if the proposed gates complete
successfully. The acceptance command verifies the reports, exact equality
flags, checkpoint inventory and hash, shared D2/D3 sources, and that the
four-rank D4 module IDs equal the D5 trainer half. Then the guarded D5
one-update driver can be considered, still subject to the current eight-HPU
allocation check. A five-update run and learned reward model remain later
work; neither is implied by D4 or the 222 non-model tests.

## Audit correction

An earlier response incorrectly said there were no useful D4 commands or
source paths because it searched only the artifacts directory. The D4 design
and D5 runbooks are in `self_play_grpo/docs`; the D2/D3 source artifacts are
present. What remains missing under the inspected artifact root is **D4
hardware acceptance output**, not its implementation plan or inputs.
