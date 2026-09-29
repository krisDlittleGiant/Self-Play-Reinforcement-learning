# D5 checkpoint and rollout-refresh handoff

This increment connects the existing CPU-only cycle coordinator to the
production consumed-batch ledger. It has **not** launched or validated an
eight-HPU cycle.

`training/handoff.py` publishes a cycle checkpoint only after all four trainer
update acknowledgements agree. It verifies that the saved adapter directory
matches the next policy descriptor, checks the current ledger head, publishes
one checkpoint-backed consumed-batch record, and then advances the coordinator
to refresh. If publication succeeds but refresh later fails, the ledger entry
remains committed; recovery must use that checkpoint rather than reapply the
same rollout batch.

Every rollout refresh acknowledgement must include the SHA-256 of the loaded
trainable tensor values, matching the four trainers' reported fingerprint,
alongside the policy descriptor and numerical probability probe. The next
collection remains blocked until all four acknowledgements pass. This helper
does not load a model itself; the future live rollout worker must compute its
fingerprint using `training/identity.py` and supply a real probe error.

Verification on 28 September 2026: `test_training_handoff.py` passed 4/4;
the complete non-model suite passed 161 tests with three existing Habana
import/deprecation warnings. No HPU command or long run was started.

Remaining before the first integrated GRPO cycle: D4 real-Hardware
checkpoint continuation, a bounded launcher with separate four-rollout and
four-trainer roles, real probability replay and one synchronized trainer step,
real checkpoint publication, adapter reload and probes on every rollout rank,
and coordinated shutdown/failure handling. The 64-game configuration is a
collection target, not evidence that a 64-game training cycle has run.
