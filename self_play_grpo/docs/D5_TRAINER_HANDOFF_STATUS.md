# D5 trainer handoff status

`training/trainer_handoff.py` is a CPU-only admission gate for each of the four
trainer ranks. It accepts only the current production configuration (64 games,
16 games per rank, two active rollout games per rank, four-seat engine-outcome
credit, one optimizer epoch). It requires the exact frozen policy descriptor
recorded with the rollout batch and the coordinator-committed manifest SHA-256.

Before returning a shard, it invokes the existing complete-batch verifier,
which checks all 64 match files, engine replay, artifact hashes, and outcome
credit. It then loads and engine-replays only the requesting trainer's 16
matches, checks turn and owned-token counts against the receipt, and rechecks
the manifest digest after loading. No model or optimizer is constructed or
mutated by this gate.

`tests/test_trainer_handoff.py` covers exact shard selection, wrong digest,
stale policy identity, validation-only policy rejection, wrong rank, short
shard, and manifest mutation during admission. These are contract tests with a
mocked complete-batch verifier; the verifier's own integration tests remain in
the existing suite. The user-reported rollout worker guard test passed, and
the local non-model suite after this addition passed: **186 tests**, with the
three previously observed Habana import/deprecation warnings.

This is not a training run. The next implementation is four-rank HPU loading,
exact behavior probability verification, per-rank backward on the admitted
shards, one synchronized gradient reduction and optimizer step, replica
verification, format-3 checkpoint publication, and four-rollout policy
refresh. Do not start a D5 or full GRPO run until the D4 hardware gates pass
on an allocation that actually exposes the required HPUs.
