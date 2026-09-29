# D6 first production trainer restore function (2026-09-28)

`training/d6_trainer_resume.py` implements
`restore_first_production_update`. It is intended for each of the four fresh
trainer processes after the D5 `policy-000001` commit and verified refresh.
It does **not** launch HPU processes, train, or publish `policy-000002` by
itself. It is a library function for the upcoming D6 trainer worker.

Before any adapter/optimizer mutation it re-reads the authoritative
hash-linked production ledger, verifies the complete format-3 checkpoint file
inventory, constructs a resume expectation from the active config, trainable
adapter/optimizer schemas, exact code identity, Python/Torch/Transformers/PEFT/
Habana runtime versions, attention backend, dtype, and the four requested
physical trainer modules, and requires the update-1 refresh summary. It
rejects validation checkpoints and refuses a source trainer that is already
advanced or has an evaluator attached.

It then calls `SynchronousTrainer.load_checkpoint` with distributed rank and
strict expected identity. That existing loader restores the PEFT adapter,
AdamW state, update index, model train/eval mode, and the rank's saved CPU and
HPU RNG state. The wrapper checks the restored trainable-parameter and
optimizer SHA-256 fingerprints against the committed refresh evidence and
reports `restored_no_optimizer_step`. Any mismatch is an error; the caller
must discard that process and not take a training step.

The current implementation deliberately supports **only** the transition from
committed update 1 toward update 2. This keeps the D5 checkpoint's original
code identity intact. A future D6 worker must publish a new versioned code
identity for its own checkpoint and the recovery gate must recognize that
identity under an explicit migration rule. Do not weaken the code-identity
check or relabel a new worker's code as the old D5 code.

CPU-only contract tests are in `tests/test_d6_trainer_resume.py` and cover
identity-before-load ordering, rank binding, missing refresh, and a restored
optimizer fingerprint mismatch. Full real-HPU cold-process continuation and
an update-2 production checkpoint are **not yet verified**. There is no new
safe CLI training-resume command at this point.

The compatibility patches
`patches/d6_resume_ledger_hardening_v2.patch` and
`patches/d6_resume_test_ledger_fixture_v2.patch` were applied locally. Do not
apply them a second time.
