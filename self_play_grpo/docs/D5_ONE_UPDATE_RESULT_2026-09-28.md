# D5 one-update result — 2026-09-28

The user ran the command recorded in `D5_ONE_UPDATE_LAUNCH_2026-09-28.md`
on the already allocated eight-HPU `gaudi002` node. The foreground command
finished in **97m18.165s** with `status: one_update_complete`. This was one
production-path, engine-outcome GRPO update, not a five-update training run.
No additional HPU allocation, mount, or separate job was requested.

## Completed 4+4 lifecycle

- Initial policy `policy-000000` was copied from the recorded player-relative
  pilot; the D4 validation-only policy was **not** used as training weights.
- Four rollout workers on the assigned rollout role produced **64 complete
  games**, 16 per rank, with two active games per rank. Their phase summary
  reports all four exits at code 0.
- The immutable batch contained **3,586 turns** and **14,704 owned tokens**.
  The saved manifest contains 63 natural wins and one horizon draw, natural
  wins by seat `[19, 20, 13, 11]`, zero illegal substitutions, and maximum
  recorded behavior replay error `0.0`.
- Four trainer workers consumed the complete batch and performed exactly
  **one synchronized optimizer step**. All four exited at code 0. The final
  trainer metrics were policy/total loss `-0.020898882299661636`, gradient
  norm `1.5080829858779907`, pre-step mean ratio `1.0`, pre-step clip
  fraction `0.0`, and replay maximum absolute error `0.0`. The output does
  not by itself establish improvement in playing strength.
- A format-3 **production** checkpoint for `policy-000001` was published at
  `self_play_grpo/artifacts/d5-one-update-64games-qwen3-4b-seed-11/trainer-000001/trainer/checkpoints/policy-000001`.
  Its manifest SHA-256 is
  `53ca9b7ca0afecdfe1c3748bd0dfa33e3cbd8a222697a462142b6723d2e7a115`;
  a separate read-only `sha256sum` matched the cycle summary exactly.
  The updated adapter directory digest is
  `5a6cdd70e01c84f499a97011aaa3918e696af36425cd82bfa0f4d60153b5bfdd`.
- The append-only production ledger has one record,
  `commits/update-000001.json`, binding run ID, source batch digest
  `ecd20330220924ae34e87c6ebd0325bdc02f759ed0405274a055b4fe7b84c06b`,
  checkpoint digest, and `policy-000001`.
- All **four rollout refresh probes** verified the new adapter and trainable
  parameter identity. `refresh-000001/refresh_summary.json` reports
  `status: refresh_reports_verified` and maximum probe error `0.0`.
- `cycle_summary.json` reports the coordinator in phase `ready`, update
  index 1, policy `policy-000001`, no failure, and `refresh_ranks: 4`.
  This is the completed one-update exit condition. The zero counters in that
  READY snapshot are cleared per-phase counters, not a claim that no trainer
  or refresh work occurred; their rank reports and summaries are preserved.

The artifact root is
`self_play_grpo/artifacts/d5-one-update-64games-qwen3-4b-seed-11/`.
The run directory contains the complete rollout manifest and 64 match files,
four rollout/trainer/refresh worker logs, the trainer summary and format-3
checkpoint, one ledger commit, four refresh reports, and the final cycle
summary. No matching D5 worker remained running after the command returned.

## What remains

The D5 one-update integration gate is complete. Do **not** rerun it into the
same output directory or silently treat its validation history as a repeated
training experiment. The next implementation segment is D6: a restartable
multi-update driver that begins from this **production** checkpoint, restores
all four trainers' optimizer and RNG state, generates new 64-game batches with
new IDs/seeds, preserves the append-only ledger and policy-version boundary,
and proves fresh-process continuation. Then a bounded five-update run can be
tested. External evaluation against a fixed opponent is required before any
claim that the policy became stronger. A learned reward model is a separate
later extension; this result uses only engine outcomes.

Operational observation: the trainer phase printed no per-action progress
after model load. During the run, the four trainer processes and HPUs remained
active, but the console appeared stuck until the phase completed. Add
low-overhead periodic progress reporting before longer D6 runs, without
altering this completed checkpoint's code identity or run artifacts.
