# verl Gaudi GRPO support

This workspace contains the compatibility changes and launch tooling for GRPO training on
Intel Gaudi HPUs.

Start here:

- [`GRPO_MILES_SGLANG_MIGRATION.md`](GRPO_MILES_SGLANG_MIGRATION.md) — current pinned
  Miles runtime migration and staged verification commands; full-run stability pending.
- [`GRPO_GSM8K_QWEN3_GAUDI_RUNBOOK.md`](GRPO_GSM8K_QWEN3_GAUDI_RUNBOOK.md) — canonical,
  verified commands for Qwen3-4B-Base on GSM8K, plus success criteria and troubleshooting.
- [`GAUDI_FAILURE_LOG.md`](GAUDI_FAILURE_LOG.md) — chronological failure history and wrong
  hypotheses retained for forensic context.
- [`GRPO_GSM8K_GAUDI_PLAN.md`](GRPO_GSM8K_GAUDI_PLAN.md) — original design and environment
  analysis.
- [`GAUDI_GRAPH_COMPILE_DESIGN.md`](GAUDI_GRAPH_COMPILE_DESIGN.md) — HPU graph-capture notes.

Previous-runtime verified milestone: one distributed Qwen3-4B-Base GRPO optimizer step completed on
8x HL-225 using four FSDP workers and four SGLang rollout servers. Habana FusedSDPA produced
a finite gradient norm (`8.92047`), nonzero policy loss (`-0.02729`), and mixed GSM8K reward.
The complete one-epoch 4B run remains to be recorded.
