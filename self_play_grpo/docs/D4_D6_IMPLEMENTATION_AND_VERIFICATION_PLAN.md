**Quoridor Self-Play GRPO — Next Implementation and Verification Plan**

Prepared 27 September 2026 · Qwen3-4B / shared LoRA / Intel Gaudi

**Purpose and evidence boundary**

Turn the proven four-HPU rollout and four-HPU update components into a resumable eight-HPU training loop, then measure whether outcome-only training improves gameplay. This is a plan, not an implementation-completion report. No repository changes or hardware jobs were executed while preparing it.

The starting evidence is the supplied progress report and `D4_DISTRIBUTED_CHECKPOINT_RESUME_IMPLEMENTATION.md`. The live repository and source code were not available for inspection here. Existing symbols explicitly named by that document are `SynchronousTrainer.save_checkpoint` and `verify_optimizer_replicas`. Other module names, CLI commands, test filenames, report fields, and helper interfaces below are proposed contracts to implement or map onto existing code.

The repository root reported by the user is `/scratch/svijay46/verl-gaudi-support/self_play_grpo`. Preserve the attached D4 specification as the checkpoint-specific contract; this document adds integration, restart, evaluation, and verification details. Do not overwrite the existing D3 artifacts.

**1. Starting state and completion claims**

| Area | Evidence supplied | What remains unproven |
|---|---|---|
| D0–D1 | Four-trainer runtime and optimizer math passed | Integration must preserve the validated normalization and collectives |
| D2 | Four rollout HPUs; 16 games; zero illegal substitutions; replay error 0.0 | Rollout refresh after an integrated update |
| D3 | Four trainers; 16 games; 892 turns; 3,658 owned tokens; one synchronized optimizer step; reported parameter/Adam differences 0.0 | Distributed continuation and repeated fresh-data updates |
| Checkpoint | Single-process continuation passed; D3 checkpoint is validation-only | Format-3 distributed save/load and rank RNG restoration |
| Tests | Latest reported non-model suite: 94 passed | New distributed contracts and recovery tests |
| Evaluation | Bot tournament and analysis tools exist | Frozen baseline and measured learning curves |

The D3 peak reported allocation was 70,382,665,216 bytes. Treat that as a measured planning input for the tested execution shape, not a guarantee that other sequence shapes or integrated residency will fit.

Completion levels must remain separate: D4 proves continuation on controlled inputs; D5 proves integration and policy transfer; D6 proves repeated operation and a fresh-process restart; a learning experiment measures playing strength. A passing infrastructure gate does not imply a stronger policy.

**2. Scope and invariants to preserve**

Keep one Qwen3-4B actor with a shared LoRA adapter across four seats, player-relative observations, constrained legal action labels, the existing outcome transformation and within-match advantages, the validated probability replay path, and the existing loss normalization. Continue with no KL penalty for the first baseline, as configured.

Use four rollout HPUs and four trainer HPUs. Two concurrent games per rollout HPU means eight active matches at a time; collect 16 complete matches per update. Every match uses a single policy version. D5/D6 use synchronized collection and update phases; asynchronous stale-policy collection is deferred.

Preserve the pinned OpenSpiel patch, attention backend, batched KV-cache replay shape, scoped attention checkpointing, dtype, tokenizer, action grammar, and generation parameters that passed D2/D3. Changes to these require targeted probability/gradient checks before acceptance.

The trainer group contains four ranks. Rollout workers must never join the trainer gradient collective. Compute four-seat advantages on complete games before flattening examples or distributing them. Preserve the D0–D3 global normalization when local shards have different numbers of turns or tokens.

Validation commands may use synthetic advantages and explicitly permitted validation-only source artifacts. The production launcher must reject both. A D4/D5/D6 validation result must not silently become the starting policy for a learning claim. A learning run starts from a recorded initial policy or a genuine production checkpoint.

**3. Implementation sequence**

| Work package | Changes | Verification before moving on |
|---|---|---|
| P0: Ground the current interfaces | Identify existing CLI, checkpoint, rollout, evaluation, and HPU launch code; record revisions; correct stale D3 status | Match D2/D3 manifests to their reports; preserve successful artifacts; rerun the existing non-model suite once after the first changes |
| P1: D4 checkpoint schema | Format-3 manifest, rank-state records, canonical validation, format-2 compatibility | Model-free round-trip, tampering, metadata mismatch, and validation-only rejection tests |
| P2: D4 save/load and gate | Coordinated staging/publish, strict load, per-rank RNG restore, continuation CLI | Two-rank protocol smoke, then four-rank exact continuation |
| P3: D5 coordinator | Separate worker roles, complete-batch handoff, optimizer step, checkpoint, adapter refresh and acknowledgements | Fake-worker state-machine tests, then one eight-HPU cycle with a post-refresh inference probe |
| P4: D6 recovery and multi-update loop | Update-boundary restart, committed-batch bookkeeping, deterministic game identity/seed handling | Fresh-process fixed-input continuation; three updates, stop, resume for two more; bounded failure recovery |
| P5: Evaluation and learning pilot | Immutable opponent suite, baseline results, 50–100-update pilot, uncertainty reports | Candidate identity checks, seat-balanced results, matched budgets, honest learning conclusion |

Evaluation specification and the initial-policy baseline can be prepared during P1/P2 because they do not depend on the new launcher. Hardware execution should follow the existing cluster allocation/run procedure. The attached D4 scope permits its bounded gates; it does not itself authorize an eight-HPU or long-running job. This document specifies future acceptance tests and does not launch them.

**4. D4 implementation contract**

**4.1 Checkpoint contents.** Extend the existing checkpoint representation, rather than introducing a second incompatible trainer state implementation. Retain adapter files, optimizer state, `trainer_state.json`, and the format-2 compatibility RNG file. Add `distributed/manifest.json` and exactly one rank-specific RNG file for every trainer rank.

| State | Required representation |
|---|---|
| Identity | Format version, run kind, run ID, policy version, update index |
| Model | Base-model ID/revision, adapter configuration and parameter schema, tokenizer/template/grammar identity |
| Experiment | Canonical configuration digest and the pinned code/runtime/backend identities |
| Topology | Trainer world size, observed backend, rank/module/device binding, sharding rule |
| Provenance | Source rollout-manifest digest; per-rank match IDs/indices; artifact hashes |
| Trainable state | LoRA tensors; complete optimizer tensors, groups, hyperparameters and step counters |
| Random state | Each rank's Torch CPU and HPU RNG; Python/NumPy or dedicated generators if actually used in the continuation path |
| Other state | Scheduler/scaler/sampler state if those components exist; otherwise explicitly mark them absent |
| Integrity | Relative paths, byte counts, cryptographic hashes, expected file inventory |

Manifest paths must resolve within the checkpoint root, including symlink resolution where applicable. Hash referenced content, not timestamps or process-specific object hashes. Compare tensor contents and stable parameter identities directly when checking numerical equality; a file hash alone is not an optimizer-equivalence test.

Record training-semantic settings separately from operational output paths. The D4 strict fixture preserves the existing configuration-identity requirement. If output directories are excluded from a semantic digest, define that exclusion in the schema and test it; do not broadly ignore configuration drift to make resume succeed.

Source rollout identity records the data that produced the saved update. In D4 it also identifies the controlled fixture. In the production loop it is provenance for an already consumed batch, not an instruction to train that same batch again.

**4.2 Coordinated save.**

1. Finish the optimizer step and synchronize pending device work using the project's tested Gaudi execution path.
2. Confirm exact cross-rank adapter and optimizer equality using the existing verifier.
3. Capture each rank's state at the same logical boundary. Avoid random operations between that capture and the point from which uninterrupted continuation is measured.
4. Write rank records to a unique staging directory using temporary files and rename. Use the shared filesystem approach specified in D4; do not introduce unvalidated HCCL object collectives.
5. Rank 0 validates the complete rank set, metadata, hashes, and sizes; writes adapter, optimizer, and manifest into the unpublished checkpoint directory.
6. Publish by a final directory rename within the supported filesystem. Establish and test the filesystem's visibility assumptions; do not claim power-loss durability merely because rename is atomic.
7. Every trainer rank validates the published manifest and reports success. Bound synchronization waits; any rank failure fails the gate and triggers coordinated teardown.

Distinguish failure before and after publication. Before publication there must be no committed checkpoint. If publication succeeds but a worker dies before acknowledging it, the complete checkpoint can exist while the gate fails. Recovery must recognize that committed artifact; it must not assume a failed acknowledgement means publication never happened.

**4.3 Strict load.**

Validate the entire manifest and referenced file inventory before mutating model or optimizer state. Reject incompatible format, topology, binding, model/config identity, source fixture, missing rank records, corruption, destination collision, and validation-only artifacts in production mode. All ranks must agree on the checkpoint identity.

Load identical adapter and optimizer contents on all trainers, verify equality, complete required runtime initialization, and restore each rank's own RNG state last. Any subsequent warm-up that consumes RNG must be accounted for explicitly. Restore the current training mode and dropout configuration exactly. D4 supports the same four-trainer mapping; changed world size or automatic remapping is outside scope.

**4.4 Controlled continuation test.**

Use the immutable D2 manifest and D3 checkpoint. D3 lacks the original per-rank RNG, so initialize documented rank-specific RNG streams after loading it. This bridge is validation-only. Select the fixed complete action per rank and recorded replay shape described by D4; use the fixed synthetic advantage only inside this gate.

Run a synchronized controlled step, save checkpoint C, then fork the comparison:

| Branch | Operations |
|---|---|
| Uninterrupted | Draw CPU/HPU random values; execute the next controlled step; copy parameters, optimizer contents, and numerical metrics |
| Reloaded | Restore C on all ranks; repeat the draws and the identical step; compare with the uninterrupted snapshots |

Snapshots must be independent copies, not live references into the current optimizer/model. Compare random draws and each deterministic numerical metric against the same rank in the other branch. Different ranks can legitimately have different local losses because they process different examples. Cross-rank equality is required for synchronized model and optimizer replicas.

Require exact equality of random draws, trainable tensors, Adam first/second moments, optimizer counters, and deterministic scalar metrics. Timing and peak-memory measurements are recorded, not expected to match exactly. Exercise a real update path: verify a nonzero trainable gradient and parameter change in the controlled fixture; if clipping removes all signal, diagnose/select the documented fixture rather than declaring a no-op continuation sufficient.

The D3 source policy has already changed relative to its behavior data. Do not require its continuation log probabilities to equal the original rollout probabilities. The oracle is equality between continuation branches, plus finite calculations. Do not relax exact equality silently; report the failing tensor/rank and diagnose the operation. Exact reproducibility is scoped to the pinned runtime, topology, and execution shape. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

**4.5 D4 verification matrix.** Proposed test filenames can be consolidated into the repository's existing test organization.

| Test ID | Proposed test | Expected result |
|---|---|---|
| D4-U01 | Format-2 legacy round-trip and format-3 canonical round-trip | Legacy behavior preserved; new metadata round-trips without loss |
| D4-U02 | Missing/duplicate rank; filename/state-rank mismatch | Rejected before mutation |
| D4-U03 | Corrupted RNG, adapter, optimizer; changed sizes | Hash/size failure identifies the file |
| D4-U04 | Absolute, parent-traversing, or escaping symlink path | Rejected before file contents are trusted |
| D4-U05 | World size, rank binding, model/config, policy, source-manifest, shard drift | Rejected with the incompatible field identified |
| D4-U06 | Incomplete staging or destination collision | No new partially published checkpoint |
| D4-U07 | Production load of validation-only checkpoint | Rejected; no production override flag |
| D4-U08 | Rank fails before publication | Nonzero gate result, bounded teardown, no new committed checkpoint |
| D4-U09 | Rank fails after publication but before acknowledgement | Gate fails; complete published checkpoint remains recognizable |
| D4-H02 | Two trainers, controlled continuation | Exact branch comparisons and rank equality pass |
| D4-H04 | Four trainers, original final topology | Same exact comparisons pass; this is D4 acceptance |

Run model-free tests before HPU gates. Stop after sufficient verification; repeat the expensive four-rank gate only when a change affects its state or execution contract.

**5. D5 implementation contract: one integrated cycle**

**5.1 Coordinator and worker boundaries.** Create one coordinator and explicit rollout/trainer worker roles. Reuse the proven D2/D3 launch mechanisms and use distinct role groups or rendezvous domains; do not let a default eight-process group determine trainer gradient synchronization. Record the actual mapping and reject overlapping physical-device assignments.

Proposed interfaces are `collect_batch(policy_descriptor)`, `verify_rollout_manifest`, `update_from_manifest`, `save_distributed_checkpoint`, `refresh_rollout_policy`, and `verify_policy_acknowledgements`. A policy descriptor contains a version, immutable adapter identity, base-model revision, tokenizer/grammar identity, and run kind. Transport can use the already-tested shared artifact filesystem; direct tensor transfer is not required for this first gate.

The coordinator advances only when all required workers acknowledge the same transition. Rollout workers never generate a new-policy match while another worker is still serving the previous policy. D5 does not hot-swap weights in an active match.

**5.2 Lifecycle.**

| State | Action and exit condition |
|---|---|
| READY(k) | All relevant workers acknowledge policy k and compatible config |
| COLLECTING(k) | Rollout workers produce 16 uniquely identified complete matches; all four seats in each match use k |
| BATCH_COMMITTED(k) | Immutable manifest records every shard, match ID, seed, owner count, version, and digest |
| VERIFIED(k) | Trainers verify integrity, group completeness, ownership, and behavior-probability replay before mutation |
| UPDATED(k+1) | Exactly one synchronized optimizer step consumes the complete intended batch; trainer replicas agree |
| CHECKPOINT_COMMITTED(k+1) | Format-3 checkpoint records the consumed batch and complete continuation state |
| REFRESHING(k+1) | All rollout workers load the exact committed adapter, invalidate model-dependent inference caches, and acknowledge it |
| READY(k+1) | All acknowledgements and post-refresh probes pass; new collection may begin |

For D5 run this lifecycle once plus the post-refresh probe. Start from a designated initial policy, not the D4 synthetic-update artifact. Store the D5 gate output as validation-only. The production path later uses the same implementation with production inputs and real outcome advantages, but no validation override.

**5.3 Probability and gradient verification.** Before the update, replay the complete D5 batch using the unchanged behavior policy and exact recorded generation/replay contract. The current D2/D3 evidence supports a zero-error fixed-path target; retain that gate unless a separately justified execution change is tested. Record maximum error and its game/turn/token, not only the rounded aggregate.

After updating, old behavior probabilities should generally differ from new probabilities. Those differences are policy movement, not replay failure. Keep separate fields for behavior replay error and post-update probability change. Never regenerate the denominator from the updated model.

Track complete game/turn/token coverage and check that every intended owned token is accounted for exactly once in the update. Exclude observations, opponents' tokens, history copies, and padding exactly as D3 does. All-draw batches can have zero outcome signal; log that condition and its prescribed update/skip behavior rather than silently changing rewards. Preserve the existing optimizer semantics for such batches, including any momentum/weight-decay effects.

Reuse the validated D0–D3 reduction. Add an inexpensive uneven-shard fixture to check that distributing different turn/token counts across four ranks reproduces the global objective. Do not independently average each rank by its local token count and then assume rank averaging recovers the intended global weighting. [PyTorch distributed gradient synchronization reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

**5.4 Policy transfer oracle.** Every rollout worker must report the expected policy version and the content identity of all loaded trainable tensors. Checksums should use a canonical named-tensor representation, not incidental checkpoint container bytes. Base model, adapter configuration, tokenizer, and action grammar identities must also match.

Run a fixed public-observation probe set through the refreshed inference path. Compare with a canonical forward/replay of the same updated adapter under the same dtype, masking, and batch shape. Exact numerical comparison is valid only for the same validated execution contract; if different backends/shapes are deliberately introduced, establish and document the justified probe tolerance before running the gate. Do not alter the full-batch pre-update replay criterion opportunistically.

Verify a nonzero adapter change and changed action probabilities on a suitable probe from a batch with learning signal. Do not require sampled or greedy actions to change; a real policy update can preserve its preferred move. The identity check is essential even if probes happen to be insensitive to the update.

**5.5 D5 acceptance.**

| Test ID | Verification | Pass condition |
|---|---|---|
| D5-U01 | Fake-worker lifecycle | No transition before all required valid acknowledgements |
| D5-U02 | Stale policy/shard, duplicate game, missing seat/turn, modified manifest | Batch rejected before optimizer mutation |
| D5-U03 | Partial policy refresh | No new match starts; failure or bounded recovery is reported |
| D5-U04 | Uneven-shard loss fixture | Matches the established global-gradient reference |
| D5-H08 | Eight distinct HPUs in 4+4 roles | Correct role membership; only four trainers synchronize gradients |
| D5-H16 | Integrated collection | 16 complete matches; zero illegal substitutions; exact coverage |
| D5-HP | Probability replay | Meets the inherited D3 numerical gate before updating |
| D5-HU | Update and checkpoint | One synchronized step; finite state; exact trainer replica equality; complete format-3 checkpoint |
| D5-HR | Adapter refresh | Four correct acknowledgements and passing numerical probes |
| D5-HX | Teardown | All worker processes exit within the declared deadline; no orphan job processes |

**6. Restart semantics and recovery**

Use complete-update checkpoints as the recovery boundary. Do not initially checkpoint partial matches or partially accumulated gradients. The committed checkpoint is authoritative for optimizer step, policy version, last consumed rollout manifest, and next logical collection index. A convenience 'latest' pointer is secondary; a stale pointer must not cause a valid committed update to be applied again.

Record run ID, update index, batch ID, consumed manifest digest, next collection index, game ID allocation, seed derivation version, and evaluation progress. Deterministic game seeds can derive from a stable digest of `(experiment seed, collection index, game ID, purpose)`; do not use Python's randomized `hash()`. Give evaluation a distinct RNG namespace.

To make stochastic rollout continuation reproducible, use per-game generators or capture the relevant worker generator/scheduler state. With concurrent games, one global worker RNG consumed in timing-dependent order can produce different games after restart. Document whether exact gameplay reproducibility is supported; seed equality alone does not prove it.

| Interruption point | Recovery rule |
|---|---|
| During collection | Ignore unpublished partial shards; restore last committed policy; recollect deterministically or restore the documented collection state |
| After a complete batch is published, before update | Reuse it only after validating its version, identity, and not-yet-consumed status |
| During update or before new checkpoint publication | Restore the last committed optimizer/policy; re-execute the intended batch from its recorded state; never continue a half-applied update |
| After checkpoint publication, before worker refresh | Treat that update as committed; restore it and finish refreshing workers; do not apply its source batch again |
| During a partial refresh | Restore one authoritative checkpoint on all rollout workers before allowing collection |
| During evaluation | Resume or rerun deterministic evaluation game IDs, deduplicating results; do not affect training RNG |

Test these transitions with fake workers and tiny fixtures first. A bounded real-worker termination test then verifies that the actual coordinator tears down its workers and recovers at the declared boundary. Full power-loss recovery or mid-match resume is outside this phase.

**7. D6 implementation and verification**

**D6-A: Fresh-process continuation on identical data.** D4's in-process reload is necessary but does not test a completely rebuilt runtime. Compare a full-batch update in an uninterrupted trainer against the same update after terminating and relaunching the trainer processes with identical hardware mapping, checkpoint, RNG, recorded rollout batch, replay shape, and configuration. Require the same exact tensor/optimizer comparisons as D4. This can reuse a committed checkpoint and next-version rollout batch generated during D6-B; avoid collecting duplicate hardware fixtures unnecessarily.

**D6-B: Three updates, stop, two more.** Run three integrated updates with fresh 16-game batches. Save, shut down every worker and coordinator process, and relaunch from the committed checkpoint. Complete two additional updates, with refreshed inference workers and newly collected data. That gives five logical updates and 80 training games. Branching a controlled continuation comparison can execute an extra physical optimizer step; report it separately from the logical training history.

Verify monotonic logical versions, preserved optimizer counters, unique consumed batch IDs, fresh collection policy identity, no repeated token application, complete checkpoints, and successful refresh at every boundary. A fresh restart with different newly sampled games is not expected to produce identical final weights; exact continuation must compare identical data and RNG inputs.

**D6-C: Evaluation isolation.** Evaluate selected checkpoints through the existing tournament tool. Assert it loads the requested adapter and performs no optimizer step. Compare training state and RNG before and after an in-process evaluation hook, or use a separate evaluator process. Every result records candidate, opponent versions, seat, game configuration, seed/opening ID, and outcome. Validation-run gameplay results are diagnostics, not the final learning claim.

**D6-D: Resource behavior and failure.** Record phase times and allocated/reserved memory separately. Check for orphan processes, unreleased model/optimizer references, and unexplained growth on a repeated same-shape fixture after warm-up. End-to-end dynamic shapes can allocate additional caches; distinguish that from leaks before attributing a trend. Test one bounded worker failure at a checkpoint/update boundary and confirm a nonzero result, coordinated teardown, and valid recovery under the table above.

| D6 acceptance item | Required evidence |
|---|---|
| Cold-start trainer continuation | Exact comparisons on the same checkpoint and full recorded input batch |
| Repeated online training | Five logical updates; 80 fresh games with complete provenance |
| Actual restart | Old process IDs terminated; new processes load the recorded checkpoint |
| Optimizer continuity | Counters/moments persist; no fresh optimizer initialization disguised as resume |
| No duplicated training | Each committed batch appears once in the logical consumption ledger |
| Policy freshness | Each collection batch has the policy required by its update |
| Evaluation | Candidate identity verified; seat/opponent breakdown; no training state mutation |
| Resource stability | Phase measurements and bounded shutdown; no unexplained retained allocations on the fixed-shape probe |

**8. Proposed CLI and test surface**

The actual existing entrypoint was not inspected. Use the repository's CLI infrastructure and map the following contracts onto it. `sp-grpo` below is a proposed command alias, not a command known to exist. Flags and filenames are also implementation targets. Replace paths with resolved artifacts from the current repository; do not copy validation-only options into the production command.

| Proposed command | Contract |
|---|---|
| `sp-grpo validate-distributed-checkpoint-resume` | D4; 2 or 4 trainers; controlled fixture; optional validation-source bridge restricted to this gate |
| `sp-grpo validate-eight-hpu-cycle` | D5; exactly 4 rollout + 4 trainer HPUs; one real-outcome cycle and policy-refresh probe |
| `sp-grpo validate-multi-update-resume` | D6; manages bounded workers, stop/restart, and continuation comparisons |
| `sp-grpo evaluate-checkpoint` | Read-only policy evaluation with immutable opponent manifest |
| `sp-grpo train-distributed` | Production synchronized loop; bounded max updates; rejects validation-only checkpoints and synthetic advantages |

Illustrative D4 invocation, once that interface is implemented and an appropriate allocation exists:

```bash
sp-grpo validate-distributed-checkpoint-resume \
  --trainer-hpus 2 \
  --source-rollout-manifest "$D2_MANIFEST" \
  --source-checkpoint "$D3_CHECKPOINT" \
  --allow-validation-source \
  --output "$D4_TWO_RANK_OUTPUT"
```

Run the same gate with four trainers and a fresh output directory after the two-rank result passes. `$D2_MANIFEST`, `$D3_CHECKPOINT`, and the output variables must be set explicitly to verified paths. Do not invent a manifest filename from the artifact-directory name.

Illustrative D5 and D6 contracts:

```bash
sp-grpo validate-eight-hpu-cycle \
  --config "$OUTCOME_CONFIG" \
  --rollout-hpus 4 --trainer-hpus 4 \
  --games-per-update 16 --post-refresh-probe \
  --output "$D5_OUTPUT"

sp-grpo validate-multi-update-resume \
  --config "$OUTCOME_CONFIG" \
  --rollout-hpus 4 --trainer-hpus 4 \
  --updates-before-restart 3 --updates-after-restart 2 \
  --verify-cold-start-continuation \
  --opponent-manifest "$EVAL_MANIFEST" \
  --output "$D6_OUTPUT"
```

Proposed focused test files are `test_distributed_checkpoint_contract.py`, `test_distributed_checkpoint_failures.py`, `test_training_coordinator.py`, `test_policy_refresh.py`, `test_resume_ledger.py`, and `test_evaluation_isolation.py`. Integrate them into the current test tree. Keep HPU gates opt-in and separate from ordinary unit tests. Do not assert that the existing count of 94 tests will remain the final count.

Each validator must write a machine-readable report, return zero only when all required checks pass, and distinguish `failed`, `incomplete`, and `skipped` from `passed`. A timeout or skipped hardware check cannot be promoted to a pass.

**9. Evidence bundle and numerical acceptance rules**

Every hardware gate should produce its resolved configuration, code/runtime/model identities, launch topology, input manifests/digests, rank logs, checkpoint identities, phase timings, peak memory, and individual comparison results. Save these under a unique run directory and add the outcome to the existing `IMPLEMENTATION_AND_TEST_LOG.md` and `RUNBOOK.md`.

| Check | Required report detail | Acceptance rule |
|---|---|---|
| RNG continuation | Rank, generator type, equality, differing index if any | Exact equality for controlled branches |
| Adapter/Adam continuation | Parameter name, state key, shape/dtype, max difference, equality | `torch.equal` for tensors and exact counters in the fixed contract |
| Cross-rank replicas | Compared ranks and named state entries | Exact synchronized adapter/optimizer equality |
| Numerical metrics | Per-rank branch loss/gradient-related values and reductions | Same deterministic metric equals its corresponding branch value |
| Behavior replay | Maximum unrounded log-probability error and offending token | Inherited zero-error fixed execution path; any changed tolerance requires a separate justified validation |
| Update activity | Trainable gradient norm, adapter delta, optimizer step increment | Finite values; nonzero update in the designated nonzero-signal fixture |
| Data coverage | Expected/observed games, turns, owned tokens, IDs | Complete, unique logical consumption |
| Policy refresh | Expected/loaded tensor identity, version, probe result per worker | Every rollout worker matches before collection resumes |
| Performance | Collection/update/save/refresh/evaluation time; peak allocated/reserved memory | Measured and within the allocated budget; no invented throughput target |

Loss magnitude, replay accuracy, and optimizer activity are different checks. In particular, a near-zero aggregate policy loss can coexist with a nonzero gradient because positive and negative terms cancel. Do not use a nonzero scalar loss alone as the training-validity criterion.

A gate report must name the evidence files and comparison results rather than only printing `status: ok`. Preserve failure artifacts for diagnosis. After a passing gate, rerun only the tests implicated by subsequent changes plus the required next-stage checks.

**10. Freeze the evaluation baseline**

Create an immutable opponent manifest containing random-legal, shortest-path, wall-aware, and initial-Qwen policies where supported by the current bot interface. Include implementation/config digests and fixed mixed lineups. Evaluate one candidate seat at a time and rotate it through all four seats. Keep the game variant, legal-action rules, observation representation, token budget, and sampling protocol fixed.

The ordinary initial Quoridor board is deterministic. If evaluating generalization to starting positions, use a held-out bank of legal opening prefixes and group related prefixes when splitting. Different sampling seeds alone are not different board tasks.

Record wins, losses, draws, fractional result, seat breakdown, opponent lineup, game length, and wall usage. Track candidate identity on every game. Self-play's average four-seat result is fixed by the reward design and cannot establish stronger play.

Use a small roughly 200-game evaluation to verify the evaluator pipeline; it is only a coarse screen. Expand a final candidate comparison to approximately 1,000 games initially and inspect uncertainty. The needed sample size depends on the effect size and opponent/seat design. Bootstrap complete games or matched opening/seed blocks as appropriate, not individual turns or the four dependent player perspectives. Report training-seed variation separately from within-run evaluation uncertainty.

**11. First learning pilot and subsequent reward experiment**

After D6, start a production outcome-only run from the recorded initial policy. A proposed diagnostic budget is 50–100 updates: 800–1,600 games at the current batch size. Evaluate at update 0 and selected checkpoints such as 25, 50, and 100. Keep hyperparameters fixed for the first pilot; the budget does not imply convergence.

Save checkpoint, consumed rollout manifest, policy versions, seed identities, and metrics at each committed update. Use validation opponents/openings for checkpoint selection and keep a test partition untouched until selection finishes.

| Pilot observation | Next action |
|---|---|
| Clear external improvement | Confirm with larger evaluation and additional training seeds |
| Finite updates but little policy movement | Inspect effective update scale and optimizer behavior |
| Large policy movement with poorer results | Inspect clipping, learning rate, entropy collapse, and repeated actions |
| High draw fraction / many zero-advantage batches | Inspect the horizon and behavior before changing rewards |
| Gains only against one reference | Examine opponent overfitting and mixed lineups |
| Flat outcome-only performance with otherwise valid training | Diagnose exploration and credit assignment, then design a targeted reward comparison |

Once the outcome baseline is reproducible and measured, compare continuous process signals using the same initial policy, training budget, evaluator suite, and multiple seeds. Start with progress logging, then a potential-shaping control, then a learned value/turn-advantage extension if justified. Document that potential shaping with complete returns may telescope into a state baseline, while a learned bootstrapped evaluator changes the estimator and introduces approximation error. Written plans, additional games, and asynchronous training remain separate experimental variables.

**12. Work completion checklist**

- [ ] Current CLI and source locations mapped; stale D3 status corrected with evidence links.
- [ ] Format-3 manifest and compatibility/failure tests pass.
- [ ] D4 coordinated save/load implemented; two-rank smoke passes.
- [ ] D4 four-rank exact continuation passes and evidence is recorded.
- [ ] D5 coordinator and fake-worker lifecycle/failure tests pass.
- [ ] D5 eight-HPU collection/update/checkpoint/refresh probe passes.
- [ ] Update-boundary recovery and consumed-batch ledger tests pass.
- [ ] D6 fresh-process fixed-input continuation passes.
- [ ] D6 three-update/stop/two-update run and evaluation isolation pass.
- [ ] Frozen initial-policy evaluation is complete.
- [ ] Bounded production learning pilot is complete and externally evaluated.
- [ ] Learning claims distinguish improvement, no detected improvement, and inconclusive evidence.

**First implementation slice:** map the existing checkpoint interfaces, add the format-3 schema and model-free failure tests, then implement coordinated save/load. The next hardware gate is the bounded two-rank D4 smoke, followed by the four-rank continuation gate.
