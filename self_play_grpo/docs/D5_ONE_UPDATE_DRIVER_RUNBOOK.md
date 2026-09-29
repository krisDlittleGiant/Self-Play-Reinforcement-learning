# D5 one-update driver: implementation and gate status

The opt-in `self_play_grpo.training.initial_cycle` command is implemented for
one production update, not five. It performs CPU-side D4/allocation preflight
before creating a run directory and never launches at import time. The parent
must already have eight assigned HPUs; this command does not request Slurm
resources, mount filesystems, or choose other nodes.

The starting policy is copied from a recorded pilot (`pilot:` manifest with
at least one replayable complete match), with exact adapter SHA-256, LoRA
configuration, pinned model and environment, decoding contract, tokenizer,
and action-grammar identities. An existing compatible source is
`self_play_grpo/artifacts/pilot-outcome-relative-shape-v1-qwen3-4b-seed-11`.
Its observed adapter metadata has LoRA rank 16, alpha 32, dropout 0, and the
pinned local Qwen3-4B base path. The adapter is never modified in place.

One cycle has these strictly ordered phases:

1. Copy the frozen initial pilot adapter to `initial-policy/`, prepare a new
   `rollout-000000/` batch, and run four rollout workers (16 complete games
   per worker, two active games per worker).
2. Aggregate exactly 64 games and replay the complete outcome-credit batch.
3. Run four HCCL trainer workers against that immutable batch. Verify the
   per-rank reports, adapter/tensor/optimizer identities, complete format-3
   checkpoint, and a fixed post-update probability probe.
4. Commit the production checkpoint to the append-only ledger before policy
   refresh. Run four rollout refresh workers, each loading the committed
   adapter and checking named trainable tensor SHA-256 plus exact-shape probe.
5. Require all four refresh reports to agree, record the refresh evidence,
   and publish `cycle_summary.json` only if the coordinator reaches READY(1).

Each phase has a deadline, process-group teardown, and a distinct log
directory. A failed phase leaves evidence in the unique run directory and
does not publish a successful cycle summary. The run directory must be new.
Interrupted runs are not yet automatically resumable; do not reuse a partial
directory for a new run.

## Apply and verify CPU-side implementation

From the repository root, apply only these checked patch revisions, in this
order. Earlier v1 drafts are not intended for application.

```bash
git apply --unidiff-zero --check self_play_grpo/patches/d5_initial_cycle_integration_v2.patch self_play_grpo/patches/d5_pilot_bootstrap_driver_v2.patch
git apply --unidiff-zero self_play_grpo/patches/d5_initial_cycle_integration_v2.patch self_play_grpo/patches/d5_pilot_bootstrap_driver_v2.patch
bash env/shell.sh python -m pytest self_play_grpo/tests/test_initial_cycle_flow.py self_play_grpo/tests/test_initial_policy.py self_play_grpo/tests/test_initial_cycle.py self_play_grpo/tests/test_refresh_gate.py self_play_grpo/tests/test_refresh_gate_artifacts.py self_play_grpo/tests/test_phase_supervisor.py -q
bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not model' -q
```

Before those patches, the bounded CPU suite passed 221 tests with one
driver-flow test deselected and the three known Habana warnings. The two
patches pass `git apply --unidiff-zero --check` together. No HPU launch or
post-patch full-suite result is claimed.

## Hardware gate before any one-update run

Do not launch the driver until the D4 two- and four-trainer summaries pass
`verify_d4_acceptance` for the same 4+4 module layout and the *current*
process allocation includes all eight HPUs. Host-wide `hl-smi` visibility or
`/dev/accel` entries do not by themselves prove allocation. A known
`SLURM_GPUS_ON_NODE=1` is rejected, even on an eight-HPU host. Do not allocate
extra devices for this project; use the already assigned node/allocation.

The eventual HPU command is `python -m self_play_grpo.training.initial_cycle`
inside `env/shell.sh`, with the 64-game config, the compatible pilot root,
new output directory, run ID, eight explicit module IDs, both D4 summary
paths, and a free trainer rendezvous port. It is intentionally not part of
the CPU verification commands above.

After a successful one-update hardware gate, the remaining D6 work is a
restartable multi-update driver: new seeds/game IDs per update, restoration
of the production optimizer and four RNG streams, immutable batch/ledger
continuity, and a five-update cold-restart test. A learned reward model is
a later experimental extension, not a prerequisite for this outcome-reward
baseline.
