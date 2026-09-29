# D4 distributed checkpoint and resume implementation plan

Status: design only; D4 is not implemented or validated. This document is the
next implementation segment after the successful four-trainer D3 gate.

## Starting point and objective

D3 completed one real, validation-only Qwen3-4B LoRA update on four trainer
HPUs. All 16 saved games (892 turns, 3,658 owned tokens) were consumed; the
global behavior-replay error and all reported cross-rank parameter/Adam-state
differences were `0.0`. Exactly one gradient synchronization and optimizer
step completed. The maximum reported HPU allocation was 70,382,665,216 bytes.
The output is `artifacts/distributed-update-4trainer-hpu-qwen3-4b-seed-11-attention-checkpoint-v1`
under `self_play_grpo/`. Its `policy-000001` checkpoint is validation-only.

That checkpoint is not yet a distributed resume contract. The existing
`SynchronousTrainer.save_checkpoint` writes one adapter and optimizer state,
but its `torch_rng_state.pt` records only the saving process. It does not
record trainer world size, rank/device mapping, source rollout shards, or one
RNG state per rank. D4 must make those facts durable and prove that reloading
the checkpoint continues identically. It must not be described as a full
eight-HPU training run.

## Deliverables

1. A versioned distributed checkpoint format, preserving format-2
   single-process load behavior. Distributed resume must require the new
   format and reject validation-only checkpoints in production mode.
2. A coordinated save in which trainer rank 0 publishes one complete
   checkpoint only after all trainer ranks have supplied their RNG records
   and the existing replica-equality checks have passed.
3. A strict distributed load that validates metadata and every file hash
   before restoring model/optimizer state, then restores each rank's own CPU
   and HPU RNG state last.
4. A bounded `validate-distributed-checkpoint-resume` CLI gate: compare a
   controlled uninterrupted second optimizer step with the same step after
   save/reload, including RNG draws, parameters, optimizer tensors, and
   scalar metrics. The command never collects new games or publishes a
   production policy.
5. Model-free schema/failure tests, a two-rank hardware smoke, and a
   four-rank acceptance run matching the final trainer topology. Record
   measured results in `IMPLEMENTATION_AND_TEST_LOG.md` and `RUNBOOK.md`.

## Checkpoint layout and metadata

Extend the atomic checkpoint directory used by `SynchronousTrainer`:

```text
checkpoints/policy-000001/
  adapter/                         # PEFT adapter only, no frozen base model
  optimizer_state.pt               # rank-0 state after exact replica check
  torch_rng_state.pt               # existing format-2/rank-0 compatibility
  trainer_state.json
  distributed/
    manifest.json                  # format-3 distributed contract
    rng/rank-000.pt
    rng/rank-001.pt
    rng/rank-002.pt
    rng/rank-003.pt                 # four-rank acceptance case
```

`distributed/manifest.json` must contain the checkpoint format version,
experiment-config digest, model ID/revision, policy version/update index,
trainer world size, observed collective backend, local rank-to-module/device
mapping, equal contiguous match-sharding rule, source rollout-manifest digest,
per-rank match indices and artifact digests, and SHA-256 plus byte count for
each RNG file. Bind the metadata to the rank-0 adapter and optimizer files
with hashes too. Paths in the manifest must be relative to the checkpoint
root and validated against path traversal. Do not rely on timestamps or
Python object hashes for identity.

Each rank RNG record contains its rank and module ID, CPU Torch RNG state,
and that rank's HPU RNG state. Record the HPU API/representation used. On
resume, reject a missing state, duplicate rank, wrong device binding, hash
mismatch, or a state whose recorded rank differs from its filename. Do not
silently substitute rank-0 RNG for other ranks. Save the established format-2
files unchanged for existing single-HPU tests; select format 3 only when
distributed metadata is supplied.

## Save protocol

The D3 update's existing barrier and `verify_optimizer_replicas` checks are
the precondition. After the optimizer step, each rank writes its RNG record
to an output staging area using a temporary filename followed by rename.
Synchronize, then have rank 0 validate all expected rank records and hashes.
Rank 0 writes adapter, optimizer, metadata, and copied rank RNG files inside
the existing temporary checkpoint directory; publish by one final directory
rename. A barrier follows publication, and every rank checks the published
manifest. Any failure leaves no published checkpoint and no partial policy
version. Keep failed output directories for diagnosis; retries use a new
output directory.

Avoid object collectives for Python RNG payloads until their HCCL behavior is
measured. Rank-local staging files plus barriers use the shared artifact
filesystem already required by the rollout/checkpoint workflow. Validate
that filesystem visibility explicitly in the hardware gate.

## Resume protocol

Every trainer rank reads the same manifest and validates, before mutation:
configuration/revision, source-rollout identity, checkpoint version,
expected world size, rank/module/device mapping, sharding rule, policy
version, file paths, hashes, and byte counts. A mismatch is fatal; no
automatic resharding or changed world size is part of D4. Each rank loads
the identical adapter and optimizer state, then verifies exact cross-rank
parameter and Adam-state equality. Restore rank-specific RNG state *after*
model and optimizer loading, because loading may consume random numbers.
The validation CLI may explicitly opt into the D3 validation-only source
checkpoint; production resume must refuse validation-only checkpoints.

## Deterministic continuation gate

Use the saved D2 manifest and the D3 `policy-000001` checkpoint as immutable
inputs. The D3 checkpoint lacks rank RNG records, so the D4 validation gate
must initialize a documented deterministic per-rank RNG state after loading
it. This is a bridge for the gate, not a claim that D3's original rank RNG
streams can be reconstructed. Once a format-3 checkpoint exists, future
resumes must use its recorded streams.

On each rank select one fixed, complete recorded action from that rank's
assigned match shard. Use the exact recorded batched KV-cache replay shape
and scoped Qwen3 attention checkpointing proven by D3. Assign a fixed
nonzero synthetic advantage only in this validation command. Since the
checkpoint is already one update beyond the behavior policy, do **not**
apply the pre-update behavior-replay tolerance to continuation steps;
instead require finite losses/gradients and compare the two branches
exactly. This is an optimizer-resume test, not a fresh GRPO correctness or
on-policy replay gate.

Run one synchronized controlled step, save a format-3 checkpoint, then:

1. Capture per-rank CPU and HPU random draws and run the next synchronized
   step without reloading; snapshot scalar metrics, trainable parameters,
   and every optimizer tensor/counter.
2. Reload the saved format-3 checkpoint on all ranks, repeat the draws and
   identical step, and compare against the uninterrupted branch.
3. Require exact equality for RNG draws, scalar metrics, parameters, first
   and second Adam moments, step counters, and cross-rank replicas. Report
   all comparisons and checkpoint hashes, not only `status: ok`.

The gate is validation-only and must not publish its adapter as a training
policy. A two-rank run is a low-cost smoke of the protocol. Four ranks are
the D4 acceptance gate because the final trainer side has four HPUs. Use
fresh output directories for both; do not overwrite the successful D3
artifact. No long-running or eight-HPU job is authorized by this document.

## Model-free tests and fail-closed cases

- Format-2 single-process save/load remains unchanged; format-3 metadata
  round-trips canonically.
- Reject missing/duplicate rank state, altered RNG bytes, adapter or
  optimizer hash mismatch, malformed/traversing relative path, incomplete
  staging, and checkpoint destination collision.
- Reject world-size, rank-to-device, model revision, config, policy version,
  source manifest, and shard-index drift before loading weights.
- Refuse validation-only resume unless the caller explicitly enables it;
  never permit that override in the production launcher.
- Assert no checkpoint is published if any rank report or file is missing.
- Compare optimizer state by stable parameter order and state key, with
  `torch.equal` on tensors and exact scalar equality; do not infer equality
  from a checksum alone.

## Exit criteria and handoff to D5

D4 passes only when model-free tests, two-rank smoke, and four-rank
continuation all pass; rank-specific RNG draws, uninterrupted/resumed
metrics, parameters, optimizer tensors, and cross-rank replicas must match
exactly. Publish a version-3 checkpoint with a complete manifest and no
frozen base-model weights. Record elapsed time, peak HPU memory, bytes on
disk, file hashes, and every rejected-drift test.

Only then begin D5: the distinct four-rollout/four-trainer process groups,
policy transfer at a version boundary, one integrated collect-update-save
smoke, and clean teardown on eight HPUs. D6 remains the multi-update
resume/evaluation gate before a full GRPO run.
