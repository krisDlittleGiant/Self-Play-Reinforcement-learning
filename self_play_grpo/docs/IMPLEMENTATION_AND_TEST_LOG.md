# Implementation and test log

Status date: 2026-09-26

This document records what was implemented, what was actually executed, the
observed results, recovered failures, and work that is still pending. It is an
evidence log, not a claim that a training experiment has completed.

## Current status

| Area | Status | Evidence |
|---|---|---|
| Project package and configuration | Passed | Editable install succeeded; configuration tests pass |
| OpenSpiel engine build | Passed | OpenSpiel 2.0.2 compiled from the verified source archive with the Quoridor `d0606878` backport |
| Four-player engine contract | Passed | Original seven engine tests plus the new targeted complete-replay test pass; `validate-engine` passes |
| Pure math, schema, engine, grammar, and distributed contract tests | Passed | Latest aggregate suite reports 92 passed, including player-relative coordinate/action bijections, batch-shape dispatch, finite-probability gates, engine contracts, rank sharding, synthetic distributed-gradient math, trainer-manifest contracts, and per-action graph release |
| Bot evaluation fixture | Passed | Eight deterministic games completed and were written to `artifacts/fixture.jsonl` |
| Model files and tokenizer | Passed | Pinned Qwen3-4B revision, three shards, chat template, and all 209 real action labels covered |
| Model and LoRA loading on HPU | Passed | Model and adapter load on `hpu:0`; 33,030,144 trainable parameters |
| One-action constrained forward/replay | Passed in absolute and player-relative modes | Player-relative recheck selected legal `MOVE_B5`; two stochastic and two forced token positions; exact replay ratios of 1 |
| Single-HPU batched action inference | Passed through batch size 16 | Right-padded batches 2, 4, 8, and 16 were finite and replayed exactly; batch 16 peaked at 46,906,020,992 allocated bytes |
| Batched complete-match collector | Passed on one HPU | Two concurrent complete games (23 and 31 turns) replayed with exact `0.0` batch-shape error, one policy version, natural outcomes, and zero substitutions |
| Full LLM match collection and saved-artifact replay | Passed | Corrected artifact gate: 100 turns, 416 tokens, cached replay error `0.0`; 82.018 seconds |
| Single-action backward pass | Passed | Exact gradient-enabled replay; finite gradient norm `1.5376510660497742`; 504 parameter tensors with gradients; 37.220 seconds |
| Four-action bounded-memory backward | Passed | Exact replay, 20 owned tokens, gradient norm `1.9488064050674438`, peak HPU allocation 25,289,977,728 bytes; 41.893 seconds |
| Complete 100-action bounded-memory backward | Passed | Exact replay, 416 owned tokens, gradient norm `5.128081321716309`, peak HPU allocation 25,337,058,048 bytes; 187.173 seconds |
| Optimizer step | **Not run** | Backward probe reported zero optimizer steps and cleared gradients |
| Initialization checkpoint/resume | Passed | Exact adapter, behavior probabilities, CPU RNG, and HPU RNG; adapter-only checkpoint is 132,211,226 bytes |
| Non-empty optimizer checkpoint/resume | Passed | Exact continuation for all trainable parameters, 504 Adam entries, 1,512 optimizer tensors, and scalar metrics; 41.134 seconds |
| Original absolute-coordinate pilot | Archived at 63/100 | Exact replay and zero substitutions, but wins `[7,0,42,13]` exposed action-label bias; the player-relative semantic fix requires a fresh pilot directory |
| Distributed runtime contract | **Passed on four HPUs** | Four ranks bound to physical modules `0,1,2,3`; rank and device membership were one-to-one; both HCCL reductions returned the expected sum `10.0`; no model was loaded; 30.509 seconds |
| Four-HPU distributed gameplay | **Passed** | 16/16 natural games; four games per rank with two concurrent; replay error `0.0`; zero substitutions; one policy version; 3,658 owned tokens; 892 turns; wins `[4,4,4,4]`; 636.230 seconds |
| Four-trainer synthetic optimizer math | **Passed** | Exact averaged gradient and zero cross-rank differences for clipped norm, parameters, Adam moments, and step counter; one AdamW step; 32.024 seconds |
| Four-trainer real-artifact update | Implemented; HPU gate pending | Four intact matches per rank from the passed D2 artifact; action-bounded backward; one post-backward gradient synchronization phase; exact replica checks; one validation-only rank-0 checkpoint |
| Eight-HPU distributed trainer | Required, not implemented | Final target is one node with 8 Gaudi HPUs; replicated match-sharded data parallel design and staged acceptance gates are documented |
| Process evaluator experiment | Implemented as library code, not fitted | No evaluator dataset or trained evaluator exists yet |

## Environment decision and verified pins

The project uses the repository's existing `.runtime/venv` through
`bash env/shell.sh`. A second environment was deliberately not created because
it could duplicate or replace the Gaudi-specific Torch stack. Project
installation always uses `--no-deps`.

| Component | Verified value |
|---|---|
| Torch | `2.7.1+hpu.1.22.2.32.git6dbe0a4` |
| Transformers | `5.12.1` |
| PEFT | `0.20.0` |
| Accelerate | `1.14.0` |
| Hugging Face Hub | `1.28.0` |
| HPU | Available; model loaded on `hpu:0` |
| OpenSpiel package metadata | `2.0.2` |
| OpenSpiel behavioral source contract | commit `d0606878b957274cc67a918ed173b36e9fe0fed6` backported onto the 2.0.2 source archive |
| Actor | `Qwen/Qwen3-4B` |
| Actor revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Actor local path | `/scratch/svijay46/models/Qwen3-4B` |

OpenSpiel intentionally retains upstream distribution version `2.0.2`; the
forced-pass regression is the behavioral proof that the backport is present.

## Implemented components

| Path | Responsibility |
|---|---|
| `configs/quoridor_outcome.yaml` | Standard 9x9 outcome-only experiment configuration |
| `configs/quoridor_fixture.yaml` | Small 5x5 validation configuration with two-game per-HPU batching |
| `envs/quoridor.py` | One authoritative OpenSpiel state, seat mapping, horizon handling, legal transitions, terminal conversion, clone, and exact replay serialization |
| `envs/observations.py` | Structured public state, stable action labels, descriptions, and deterministic prompts |
| `policies/bots.py` | Seeded random-legal, shortest-path, and wall-aware fixed policies |
| `policies/llm.py` | Qwen3/LoRA loading, legal-token trie, constrained sampling, exact behavior log probabilities, and replay |
| `rollouts/collector.py` | Sequential and dynamically compacted batched complete-match collection with independent per-game RNG streams and match-level credit assignment |
| `rollouts/analysis.py` | Accelerator-free per-seat outcome, movement-direction, opening-action, and action-frequency diagnostics |
| `rollouts/pilot.py` | Frozen adapter identity, resumable pilot manifest, artifact hashes, deterministic seeds, and atomic per-game commits |
| `rollouts/distributed.py` | Deterministic rank-to-match sharding, per-rank reports, and globally contiguous report aggregation for distributed gameplay |
| `rollouts/replay.py` | State-by-state action replay with terminal result, reason, and byte-identical final-environment checks |
| `rollouts/schema.py` | Versioned match/turn/policy records, ownership checks, JSON/JSONL serialization, and policy-version checks |
| `rewards/outcome.py` | Fractional result conversion and four-player outcome advantages |
| `rewards/progress.py` | Wall-respecting path proxy, potential shaping, own-turn GAE, and evaluator features |
| `training/loss.py` | Owned-token clipped surrogate, constrained ratios, optional sampled KL, and fixed horizon normalization |
| `training/value.py` | Optional four-output outcome evaluator, match-level split, fit loop, and Brier score |
| `training/loop.py` | Explicit collect-update-checkpoint loop with replay gate and resume support |
| `evaluation/tournament.py` | Fixed-bot seat rotation, whole-match bootstrap, summary, and atomic JSONL output |
| `cli.py` | Guarded engine, bot, model asset/load, policy-forward, collect-only match, and training entry points |
| `scripts/install_open_spiel.sh` | Verified, repeatable OpenSpiel source build that preserves the Gaudi environment |

## Changes made during bring-up

1. Created `self_play_grpo/` as a separate editable package inside the existing
   repository and reused `.runtime/venv`.
2. Implemented the complete adapter, observation/action representation, bot
   policies, match collector, schema, outcome and process rewards, constrained
   actor, trainer, evaluator, tournament runner, CLI, configurations, and tests.
3. Added a source-based OpenSpiel installer. It downloads the 2.0.2 archive by
   exact URL, verifies SHA-256
   `6f520dfe499b9e5e1e3a31cf77bf071abb02b98104789e7698021922a064e856`,
   applies the Quoridor forced-pass fix, caps compiler parallelism, and builds
   without dependency resolution or build isolation.
4. Added recovery for interrupted/partially applied OpenSpiel patches using a
   completion stamp and pristine re-extraction from the verified archive.
5. Switched the actor from unavailable Qwen3-1.7B to the requested local
   Qwen3-4B checkpoint and retained both the remote identity/revision and local
   load path in configuration.
6. Added `validate-model-assets` to verify local revision metadata, shards,
   tokenizer round trips, chat templating, and action-token limits without
   loading weights.
7. Added `validate-model-load` to load the model and LoRA adapter without a
   forward pass.
8. Made training imports lazy so engine, bot, and tokenizer-only commands do not
   import the trainer unnecessarily.
9. Added a scoped PEFT compatibility workaround: dense BF16 models bypass the
   Intel Neural Compressor LoRA dispatcher and use PEFT's generic dense Linear
   backend. No environment packages were upgraded or downgraded.
10. Added `validate-policy-forward`, which samples one legal action and replays
    its constrained behavior probabilities. The gate now passes exactly.
11. Corrected the exhaustive label enumerator after source inspection confirmed
    that a forced pass is encoded as a move to the pawn's current coordinate,
    not a separate `PASS` notation.
12. Added `collect-policy-match`, a no-optimizer gate that collects one complete
    model match, writes full `MatchRecord` JSONL, reads it back exactly, replays
    every environment state, and replays every action's behavior probabilities.
13. Added `diagnose-policy-replay` to compare recorded behavior probabilities
    against both the KV-cached sampling path and full-sequence training path.
14. Diagnosed the complete match and made incremental KV-cache decoding the
    single probability contract for sampling, replay, policy loss, and
    reference-policy replay. The full-sequence function remains diagnostic
    only because it is not numerically equivalent on the pinned HPU stack.
15. Added `validate-policy-artifact` so a preserved match can undergo exact
    environment and authoritative probability replay without recollection or
    any optimizer action.
16. Removed the trainer's obsolete `config.use_cache = false` assignment; the
    differentiable loss path now explicitly and consistently requests the same
    cache behavior as collection.
17. Replaced the trainer's retained full-batch autograd graph with exact
    action-by-action gradient accumulation. Every action is divided by the same
    fixed batch denominator before backward, preserving the summed objective
    while bounding live graph memory to one action.
18. Made AdamW `weight_decay: 0.0` explicit in both configurations and skip the
    optimizer step when the complete gradient norm is zero. A uniform outcome
    draw therefore cannot move adapter weights through decay or stored Adam
    momentum.
19. Added `validate-policy-batch-backward`, which runs the production streaming
    loss over a configurable prefix of the saved match with synthetic +1
    advantages, checks replay and gradients, clears them, and never constructs
    or steps an optimizer.
20. Added `show-action-history`, a model-free, engine-free reader for compact
    action timelines or detailed JSON containing exact generated tokens,
    behavior probabilities, grammar branching, prompts, and legal menus.
21. Replaced redundant full-model checkpoint serialization with atomic PEFT
    adapter-only format 2. Resume requires the exact recorded experiment config
    and rejects missing or unexpected adapter keys.
22. Added `validate-checkpoint-roundtrip`, which saves an initialization
    checkpoint, perturbs an adapter parameter and RNG states, reloads, and
    requires exact adapter, probability, and RNG continuation without an
    optimizer step.
23. Added `validate-optimizer-checkpoint` for a non-empty Adam state. It saves
    after one controlled synthetic step, computes the next step uninterrupted,
    reloads, repeats that step, and requires exact parameter, optimizer tensor,
    and scalar-metric equality.
24. Removed manual blanket movement of loaded optimizer state to HPU and rely on
    PyTorch's per-state placement policy, which preserves CPU Adam step counters
    when the optimizer is non-capturable while placing moments with parameters.
25. Added `collect-policy-pilot`, a resumable no-optimizer M1 collector. It
    snapshots and hashes the initial PEFT adapter, pins the exact configuration,
    uses a deterministic seed per game, writes one match atomically, replays its
    environment and cached probabilities before registering it, and refuses
    policy/config/seed/tolerance drift or target shrinkage. `--max-new-games`
    bounds the amount of work performed by one invocation.
26. Extended pilot summaries with fractional results and win counts by seat and
    mean game length. Loading remains backward-compatible with manifests written
    before these additive summary fields existed; the next successful commit
    upgrades the saved summary.
27. Recorded eight-HPU training as a final acceptance requirement. The chosen
    design is one replicated policy process per HPU, complete-match sharding,
    one gradient reduction per update, rank-0 atomic checkpointing, and staged
    two-HPU then eight-HPU validation. Current code remains single-HPU.
28. Added `analyze-policy-pilot`, a model-free and engine-free pilot reader that
    verifies every registered artifact hash and identity, then reports outcomes,
    goal-forward/backward/lateral movement, net progress, wall use, openings,
    and top action labels by canonical seat.
29. Added a player-relative action representation. Each acting seat's board,
    player numbering, recent moves, and legal-action coordinates rotate so its
    own pawn starts at the bottom and its goal is always the top. Canonical
    logged states and engine action IDs are unchanged, and absolute mode remains
    available for replaying the original 63-game pilot.
30. Added `validate-policy-batched-forward`, the first single-HPU scaling gate.
    It performs padded multi-environment KV-cache sampling with independent RNG
    streams and legal tries, then requires the resulting behavior probabilities
    to replay through the established single-row path within tolerance before
    production batching may be enabled.
31. Made probability replay fail closed on every NaN or infinity. Diagnostics
    now report JSON-safe per-row behavior and replay values plus the exact
    non-finite positions. The padded batch prefill was changed from left to
    right padding so padded query positions retain real prefix keys on the HPU.
32. Validated the corrected action sampler at batch sizes 2, 4, 8, and 16 on
    one HPU. Every row was finite and reproduced by authoritative single-row
    KV-cache replay with maximum error `0.0`; batch 16 was selected as the
    one-HPU collection ceiling for the standard 16-game update.
33. Added `BatchedMatchCollector` and connected it to the resumable pilot.
    Active games share one model call per turn while retaining independent
    environments, seeds, RNG objects, legal tries, and artifact identities.
    Finished games are compacted and returned in original index order.
    `rollout.parallel_games_per_rank` is `2` in both standard configurations,
    matching two concurrently active games per rollout rank. In the final 4+4
    topology, each of four rollout ranks processes four games as two batches.
    `--parallel-games` remains a bounded operational override; batch 16 is the
    validated one-HPU inference ceiling rather than the training default.
34. The first complete two-game HPU smoke proved that one-action single-row
    equivalence does not generalize across variable prompt shapes. At joint
    step 14 (763 prompt tokens), batched behavior assigned `-2.5788879` to the
    `MOVE` token while both single-row cached and full replay assigned
    `-4.5110474`, an error of `1.9321594`. Added recorded batch-shape contract
    version 1: batch size, target row, right-padded prompt width, and pad token
    now travel with every batched sample. Replay and training reconstruct that
    tensor shape with duplicate non-target rows, which are mathematically
    independent and contribute no loss. Samples lacking this metadata fail
    closed instead of being treated as single-row samples.
35. Added the first distributed runtime contract. `validate-distributed-runtime`
    requires explicit `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and
    `LOCAL_WORLD_SIZE` from `torchrun`, rejects single-process, multi-node, or
    mismatched launches, initializes the pinned HCCL
    path, binds each process to its local HPU, and reduces rank/device one-hot
    vectors plus a known scalar sum. It loads no model and changes no weights.
36. Added `collect-distributed-policy-pilot`, an integrated single-node gameplay
    gate for the rollout half of the final 4+4 architecture. Rollout rank 0
    creates the only frozen adapter snapshot, broadcasts every trainable
    tensor, and four ranks collect four disjoint games each in two-game local
    batches. Each rank persists and replays its matches, then HCCL reduces the
    global maximum replay error. Rank 0 accepts four rank reports and publishes
    a normal 16-game pilot manifest only after every gate succeeds.
    The command contains no backward or optimizer path and refuses an existing
    output directory so a failed smoke cannot be mistaken for a resume.
37. Passed the standalone four-HPU distributed runtime gate. Habana bound the
    four processes through `HLS_MODULE_ID=0,1,2,3`; rank and logical-device
    membership were `[1,1,1,1]`; both collective probes reduced to `10.0`;
    tensor allocation succeeded on process-local `hpu`. The run loaded no model
    and completed in 30.509 seconds. With GPU migration enabled, PyTorch reports
    the process-group backend string as `nccl`; the actual Gaudi path is proven
    by `initialize_distributed_hpu`, HLS module binding, and successful HPU
    collectives.
38. Passed the integrated D2 four-rollout-HPU gameplay gate. Ranks owned
    disjoint match indices `0-3`, `4-7`, `8-11`, and `12-15`; each rank's
    replay maximum was `0.0`. All 16 games ended in natural wins, with 892
    turns, 3,658 owned tokens, no illegal substitutions, and wins
    `[4,4,4,4]`. Rank 0 published the complete manifest at
    `artifacts/distributed-gameplay-4rollout-hpu-qwen3-4b-seed-11/manifest.json`.
    Runtime was 636.230 seconds.
39. Implemented the bounded D1 four-trainer optimizer-math gate. It loads no
    model, performs no gameplay, and writes no artifact. Each rank constructs
    a different deterministic synthetic gradient, gradients are averaged at
    one synchronization boundary, the common gradient is clipped, and one
    AdamW step is executed. The gate requires exact averaged-gradient math,
    parameters, first and second Adam moments, step counters, and gradient norm
    across all ranks. The distributed parser/sharding/math suite now has 24
    passing model-free tests.
40. Passed D1 on four trainer HPUs in 32.024 seconds. The pre-clip norm was
    `96.69539642333984`, parameter change was `0.0010000020265579224`, and all
    averaged-gradient, norm, parameter, first-moment, second-moment, and step
    cross-rank differences were exactly `0.0`. Exactly one synchronization
    phase and one AdamW step executed; gradients were cleared.
41. Implemented D3 as `validate-distributed-policy-update`. The command rejects
    an incomplete/mixed/config-drifted rollout manifest and uneven trainer
    shards, then gives four ranks four complete matches each. It restores and
    broadcasts the exact behavior adapter, performs memory-bounded real replay
    and backward locally, reduces all 504 LoRA gradients only at the complete
    local-objective boundary, clips and steps once, and requires exact
    parameters, Adam moments, step counters, gradient norm, and changed-tensor
    count across ranks. Rank 0 alone writes a validation-only adapter/optimizer
    checkpoint plus a summary; every rank writes a disjoint report. The
    targeted distributed suite has 34 passing tests and the full model-free
    aggregate has 91 passing tests.
42. The first D3 hardware attempt failed safely before gradient synchronization
    or any optimizer step. All four ranks exhausted HPU memory while beginning
    another batched-shape action replay; the allocator could not satisfy a
    1,402,065,408-byte request. The run stopped in 113.238 seconds and published
    no rank report or checkpoint. The failed output directory is retained.
43. Root cause was differentiable action lifetime rather than the HPU caching
    allocator: native `torch.hpu` exposes no `empty_cache`, and Habana's mapped
    `torch.cuda.empty_cache()` explicitly warns that it is an inactive no-op.
    The streaming loop kept the prior action's output/graph Python references
    alive while Python evaluated the next forward. Each action is now isolated
    in its own function frame and returns CPU-only detached diagnostics after
    backward, forcing completion and making the old graph unreachable before
    the next variable-shape allocation. A regression asserts that the previous
    differentiable output is destroyed before the next replay. Per-rank memory
    progress is emitted every 25 actions. Focused loss/distributed tests report
    39 passed; the full model-free suite reports 92 passed.

## Executed validation record

### Package and deterministic tests

| Command or check | Observed result |
|---|---|
| `python -m pip install --no-deps -e ./self_play_grpo` through `env/shell.sh` | Editable wheel built; `self-play-grpo-0.1.0` installed |
| `pytest self_play_grpo/tests -m 'not engine and not model' -q` before OpenSpiel | 19 passed, 1 engine test module skipped, 3 Gaudi warnings; 8.62 s |
| Forced-pass test only | 1 passed; 2.56 s |
| `pytest self_play_grpo/tests/test_engine_contract.py -q` | 7 passed; 0.51 s |
| Full `pytest self_play_grpo/tests -m 'not model' -q` after engine install | 26 passed, 3 Gaudi warnings; 47.87 s |
| Updated aggregate after label/replay regressions | 28 passed, 3 expected Gaudi warnings; 14.14 s |
| Configuration tests after Qwen3-4B change | 2 passed; 0.18 s |
| Shell syntax for OpenSpiel installer | Passed |
| `python -m torch.distributed.run --nproc-per-node=4 --module self_play_grpo.cli validate-distributed-runtime --expected-world-size 4` | Passed on four HPUs: module IDs `0,1,2,3`, one-to-one memberships, reduction values `[10.0,10.0]`, status `ok`; 30.509 s |
| Four-rank `collect-distributed-policy-pilot`, 4 games/rank, 2 parallel/rank | Passed: 16 natural games, exact global replay, zero substitutions, wins `[4,4,4,4]`, complete manifest; 636.230 s |
| Distributed parser, rollout-sharding, and synthetic-gradient unit tests | 24 passed; 0.99 s |
| Four-rank `validate-distributed-optimizer-math` | Passed: one synchronization phase and one AdamW step; all gradient/parameter/state differences `0.0`; 32.024 s |
| Distributed runtime/rollout/trainer-sharding targeted suite after D3 | 34 passed; 0.59 s |
| Post-D3 model-free aggregate | 91 passed, 3 expected Habana warnings, no skips; 9.25 s |
| First four-rank D3 real update attempt | Failed safely on all ranks during batched-shape backward: HPU allocation request 1,402,065,408 bytes could not be satisfied; no synchronization, optimizer step, rank report, or checkpoint; 113.238 s |
| Post graph-lifetime-fix loss suite | 5 passed, 3 expected Habana warnings; 9.18 s |
| Post graph-lifetime-fix model-free aggregate | 92 passed, 3 expected Habana warnings, no skips; 10.23 s |
| Focused loss and distributed recheck with memory telemetry | 39 passed, 3 expected Habana warnings; 8.62 s |
| Python syntax checks for edited modules | Passed |
| New exhaustive-label and complete bot-match replay regressions | 2 passed; 1.05 s |
| Cached probability-dispatch regression (`test_token_trie.py`) | 4 tests in file passed; 0.49 s using the existing venv interpreter |
| Syntax compilation after cached-path correction | Passed for all source and test modules |
| CLI parser after cached-path correction | Passed; `validate-policy-artifact` is listed |
| Pure host-compatible modules after cached-path correction | 18 passed, 3 Torch-dependent tests skipped because the host cannot load the container HPU backend; 4.31 s |
| Updated aggregate after cached replay/backward changes | 29 passed, 3 expected Gaudi warnings; 8.85 s |
| Bounded-memory source/config/parser checks | Python compilation passed; 6 host-compatible tests passed; new command help passed |
| Updated aggregate including bounded-memory gradient equivalence | 30 passed, 3 expected Gaudi warnings; 11.69 s |
| Pilot collector syntax and CLI parser | Python compilation passed; `collect-policy-pilot --help` passed |
| Pilot manifest/configuration tests, first run | 7 passed, 1 failed in 1.25 s because the rejection message was less specific than the assertion; validation still rejected the bad record |
| Pilot manifest/configuration tests after clearer validation order | 8 passed; 0.37 s |
| Pilot syntax, parser, manifest/configuration recheck | Compilation and help passed; 8 tests passed in 0.51 s in the existing repository venv |
| Atomic pilot-root regression, first run | 8 passed, 1 failed in 0.90 s because the fake PEFT test double incorrectly required a nonexistent save directory |
| Final pilot manifest/configuration/atomic-root recheck | 9 passed in 0.44 s; syntax compilation passed |
| Updated aggregate including pilot collector regressions | 34 passed, 3 expected Gaudi warnings, no skips; 19.85 s |
| Pilot additive-summary compatibility checks | 9 targeted tests passed in 0.52 s; syntax compilation and CLI parsing passed |
| Aggregate after additive pilot summaries | 34 passed, 3 expected Gaudi warnings, no skips; 10.22 s |
| Pilot seat-analysis unit tests | 2 passed in 0.66 s using the repository interpreter |
| Live 63-game pilot diagnostic | All 63 artifact hashes/identities parsed in 4.5 s; reproduced wins `[7,0,42,13]` and the movement imbalance without loading OpenSpiel or the model |
| Updated aggregate including pilot analysis | 36 passed, 3 expected Gaudi warnings, no skips; 13.22 s |
| Relative-coordinate/configuration/analysis/token-trie targeted gate | 22 passed in 0.44 s; Python compilation and batched-forward parser check passed |
| Post-relative-representation aggregate | 50 passed, 3 expected Gaudi warnings, no skips; 9.84 s |
| First two-environment padded HPU gate | Failed safely as a scaling gate: selected `WALL_C1H` and `MOVE_B5`, but maximum replay error was non-finite (`NaN`); 34.832 s |
| Non-finite fail-closed and padding correction tests | 21 targeted tests passed in 0.47 s; compilation and diff checks passed |
| Corrected two-environment right-padded HPU gate | Passed in 33.434 s; both rows selected legal `MOVE_B5`, all behavior/replay values were finite, and maximum single-row replay error was `0.0` |
| Four-environment right-padded HPU gate | Passed in 34.126 s across prompt lengths 795, 805, 819, and 833; all four selected legal `MOVE_B5`, all values were finite, and replay error was `0.0` |
| Eight-environment right-padded HPU gate | Passed in 37.177 s; sampling-only time 1.6837 s, baseline allocation 8,183,381,632 bytes, peak 28,296,229,120 bytes, eight legal `MOVE_B5` actions, and exact single-row replay |
| Sixteen-environment right-padded HPU gate | Passed in 41.370 s; sampling-only time 3.1428 s, baseline allocation 8,182,873,728 bytes, peak 46,906,020,992 bytes, increase 38,723,147,264 bytes, all rows finite/legal, and replay error `0.0` |
| Batched complete-match collector CPU contract | 2 passed in 0.84 s; dynamic active batch sizes `[3,3,2,1]`, output ordering, seeds/results, and duplicate-ID rejection passed |
| Batched pilot config/parser gate | Compilation passed; 5 targeted collector/config tests passed in 0.56 s; `collect-policy-pilot --help` exposes `--parallel-games` |
| Host aggregate attempt after batched collector | Not an authoritative pass: engine collection stopped because container-built `pyspiel.so` requires GLIBC 2.38; excluding that module reported 43 passed/4 backend skips but the incompatible host Torch extension exited 139. Run the container aggregate below. |
| Container aggregate after batched collector | 56 passed, 3 expected Habana warnings, no skips; 17.15 s |
| First two-game multi-turn HPU smoke | Failed safely before registration or optimization: game 0 was preserved after a 23-turn natural win, but single-row cached replay diverged from its batched behavior probabilities by `1.93216`; 84.599 s. This proves one-action batch equivalence does not cover later variable prompt shapes. |
| Multi-turn mismatch diagnosis | 104 generated tokens checked in 48.077 s; worst error was joint step 14/token 0 (`MOVE`), prompt length 763, batch row 0 of 2. Behavior `-2.5788879`; cached and full single-row `-4.5110474`. |
| Recorded batch-shape contract unit gate | Compilation passed; 13 token-trie/dispatch/collector/config tests passed in 0.50 s; HPU equivalence remains pending. |
| Container aggregate after shape-contract v1 | 57 passed, 3 expected Habana warnings, no skips; 10.64 s |
| Container aggregate after setting the eight-HPU-aligned concurrency default | 57 passed, 3 expected Habana warnings, no skips; 9.37 s |
| Shape-contract v1 two-row HPU action gate | Passed in 33.577 s; sampling 0.6923 s, batch-shape replay error `0.0`, single-row diagnostic error `0.0`, all values finite, baseline 8,183,168,640 bytes, peak 13,153,430,016 bytes |
| Shape-contract v1 two-game multi-turn HPU retry | Passed in 91.568 s; 2 games/54 turns/244 owned tokens, lengths 23 and 31, two natural seat-2 wins, one frozen policy version, zero substitutions, exact environment round trips, and maximum probability error `0.0` |
| Shape-contract v1 differentiable replay at former worst state | Passed in 33.671 s at joint step 14 (`MOVE_A3`); replay error `0.0`, mean ratio `1.0`, loss `-0.04`, gradient norm `3.572176`, gradients on 504 trainable tensors, gradients cleared, and zero optimizer steps |
| Complete 23-action shape-contract backward | Passed in 64.675 s; 104 owned tokens, replay error `0.0`, mean ratio `0.99999994`, gradient norm `10.600649`, gradients on 504 tensors, peak 45,171,202,944 bytes (increase 36,988,378,368), gradients cleared, and zero optimizer steps |
| Shape-contract v1 optimizer checkpoint continuation | Passed in 38.919 s; 252 parameter tensors changed, 504 Adam entries and 1,512 optimizer tensors populated, parameters/optimizer/scalar metrics continued exactly after reload, gradients cleared, checkpoint 396,882,682 bytes |
| Fresh player-relative 9x9 pilot, first batched pair | Passed in 258.812 s; 2 natural games/108 turns/440 owned tokens, lengths 46 and 62, both won by seat 1, exact replay, zero substitutions, one frozen policy version, target remains 100 |
| Fresh two-game pilot behavior analysis | Model/engine-free analysis passed in 0.7 s; all seats had positive net goal progress, seat 1 made 16 forward/0 backward moves and won both games, and seats 0–2 shared relative opening `MOVE_D9` |
| Fresh 9x9 pilot batched cross-process resume | Passed in 249.488 s; adapter digest and policy version unchanged, games 2–3 appended with seeds 13–14, lengths 57 and 41, both natural seat-0 wins, exact replay, zero substitutions; cumulative 4 games/206 turns/844 tokens |
| Fresh four-game pilot behavior analysis | Model/engine-free analysis: 4/4 natural games, wins `[2,2,0,0]`; every seat retained positive net goal progress `[25,20,17,18]`, unlike the archived absolute-coordinate seat-1 failure |
| Fresh 9x9 pilot, next four games in two local batches | Passed in 772.826 s; seeds 15–18 produced four natural wins with lengths 84, 47, 56, and 55; exact replay, zero substitutions, unchanged adapter/policy identity; cumulative 8 games/448 turns/1,840 tokens and wins `[2,2,2,2]` |
| Fresh eight-game pilot behavior analysis | Model/engine-free aggregate: every seat has two wins and positive net goal progress; action/move/wall counts by seat are `115/109/6`, `113/107/6`, `111/106/5`, and `109/102/7`; forward/backward counts are `52/4`, `42/6`, `48/6`, and `50/9` |
| Distributed runtime parser unit gate | 13 passed; missing/noninteger variables, global/local world-size mismatch, invalid rank ranges, multi-node topology, and forbidden one-process validation all fail closed |
| Distributed rollout sharding/report gate | 4 CPU cases cover four-rank/four-game disjoint indices 0–15, globally ordered aggregation, invalid local-rank rejection, and atomic report round trip |
| Post-D1 model-free aggregate | 81 passed, 3 expected Habana warnings, no skips; 13.31 s |
| Distributed trainer sharding/report gate | 10 CPU cases cover equal intact-match shards, invalid topology/count rejection, complete-manifest requirements, configuration drift, and atomic rank reports |
| Post-D3 model-free aggregate | 91 passed, 3 expected Habana warnings, no skips; 9.25 s |

The coding sandbox cannot normally pass the container runtime's Unix socket and
its host cannot load the container-built `pyspiel.so`. Short container tests
therefore require an explicitly approved invocation, while the user runs all
HPU gates. The latest fully reported aggregate is the 91-case result above;
the targeted distributed parser/rollout/trainer-sharding subset contains 34 cases. Its three
warnings are the already documented Habana `pkg_resources`, `pyhlml`, and
unavailable-Apex warnings; no tests were skipped.

### Engine CLI gate

`validate-engine` returned `status: ok` with:

- package version `2.0.2`;
- expected revision `d0606878b957274cc67a918ed173b36e9fe0fed6`;
- canonical-to-engine seat map `[0, 2, 1, 3]`;
- observed first-round engine order `[0, 2, 1, 3]`;
- a natural winning fixture for every canonical seat;
- winning fixture lengths of 13, 14, 15, and 16 actions;
- exact environment serialize/replay equality.

### Bot tournament fixture

Eight games were run with two games per candidate seat, seed 11, and the
shortest-path candidate against random, shortest-path, and wall-aware opponents.

| Metric | Result |
|---|---|
| Games | 8 |
| Candidate fractional result | 0.25 |
| Win rate | 0.25 |
| Draw rate | 0.0 |
| Mean game length | 15.0 |
| Result by seat | seat 0: 0.0, seat 1: 0.0, seat 2: 1.0, seat 3: 0.0 |
| Whole-match bootstrap 95% interval | `[0.0, 0.625]` |

This fixture proves deterministic orchestration and artifact writing. With only
two games per seat it is not evidence of policy strength or a reliable seat
effect estimate.

### Model asset gate

`validate-model-assets` returned `status: ok`:

- 13 metadata files at the configured revision;
- three weight shards totaling 8,044,982,000 bytes;
- `Qwen2Tokenizer` selected for Qwen3-4B;
- the original run checked all 209 real 9x9 action labels plus one unnecessary
  generic `PASS` string; every real label round-tripped exactly;
- maximum action length is 6 tokens against a configured cap of 16;
- fixture chat prompt is 27 tokens.

The enumerator has since been corrected to check exactly 209 labels: 81 move
coordinates and 128 wall placements. OpenSpiel renders a forced pass as the
pawn's current coordinate, which is already in the move family.

### Model/LoRA load gate

The first load proved that checkpoint loading and HPU transfer worked, then
failed while PEFT probed an incompatible Neural Compressor dispatcher. After
the scoped dispatcher workaround:

- a tiny one-layer Qwen3 generic-LoRA injection passed with 2,048 trainable
  parameters;
- the real load returned `status: ok` in 33.43 seconds;
- all parameters were on `hpu:0`;
- model class was `PeftModelForCausalLM`;
- total parameters were 4,055,498,240;
- trainable LoRA parameters were 33,030,144 (about 0.814%);
- frozen base weights were BF16 and adapter parameters were FP32.

### One-action constrained forward/replay gate

`validate-policy-forward` returned `status: ok` in 33.372 seconds:

- selected legal action `MOVE_B5` (engine action 18);
- completion token IDs `[29116, 1668, 20, 198]` decode to exactly
  `"MOVE_B5\n"`;
- allowed-token counts were `[2, 3, 1, 1]`, giving two stochastic and two
  forced positions;
- a cloned environment advanced to exactly one joint action;
- behavior and replayed log probabilities matched bit-for-bit;
- maximum absolute log-probability and ratio errors were both `0.0`;
- every probability ratio was exactly `1.0`.

This validates the trainer's current `2e-4` replay threshold for this single
fixture action. A complete-match replay gate is still required before training.

After switching the standard configs to player-relative coordinates, the same
gate passed again in 53.315 seconds. It selected legal relative action
`MOVE_B5` with engine action 18 and token IDs `[29116,1668,20,198]`. Behavior
log probabilities were `[-0.0002288818359375,-0.011249542236328125,0.0,0.0]`;
single-row cached replay matched exactly, all ratios were `1.0`, and maximum
log-probability and ratio errors were `0.0`. This is an inference/replay gate,
not evidence about game quality or batching.

### Complete model-match gate: failed safely under the former replay path

`collect-policy-match` ran for 2 minutes 25.102 seconds and stopped before any
backward pass or optimizer construction because maximum behavior replay error
was `0.157623`, above the required `0.0002`. The tolerance was not changed.

The preserved artifact is
`artifacts/policy-match-gate-seed-11.jsonl` and has:

- one record, 921,026 bytes;
- 100 turns and 25 turns per seat;
- one policy version, `gate:1cfa9a720891`;
- 416 owned completion tokens;
- prompt lengths from 518 to 873 tokens;
- completion lengths from 4 to 6 tokens;
- uniform final result `[0.25, 0.25, 0.25, 0.25]`;
- termination reason `engine_draw`.

The command writes/reads the JSONL and performs exact state-by-state environment
replay before checking model probabilities, so persistence and environment
replay passed. Only the former full-sequence probability gate failed. No
optimizer or backward pass ran.

### Cached-versus-full diagnostic: root cause isolated

`diagnose-policy-replay` checked all 416 owned tokens in the preserved match in
1 minute 42.564 seconds:

- maximum behavior-versus-cached error was exactly `0.0`;
- maximum behavior-versus-full-sequence error was `0.157623291015625`;
- maximum cached-versus-full-sequence error was the same
  `0.157623291015625`;
- the worst token was `_A` in `WALL_A1H` at joint step 3, where cached and
  behavior log probability were both `-0.9195594787597656` and the
  full-sequence value was `-1.0771827697753906`.

This rules out stochastic replay drift, record corruption, and grammar-mask
differences: the exact computation used during sampling reproduces every
recorded value. The discrepant operation is changing to a differently shaped
full-sequence forward on HPU BF16. The tolerance remains `2e-4`; instead, the
implementation now uses the sampling-aligned KV-cache path for differentiable
training probabilities. Validation of the corrected dispatcher against the
preserved artifact and a controlled backward pass were pending at this point;
both subsequently passed as recorded below.

### Corrected artifact gate: passed

The user ran `validate-policy-artifact --config
self_play_grpo/configs/quoridor_fixture.yaml --input
self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl --replay-tolerance 2e-4`
through `bash env/shell.sh python -m self_play_grpo.cli`.
It returned `status: ok`, one game, 100 turns, 416 owned tokens,
`probability_path: kv_cache`, and maximum absolute error `0.0`.
Wall time was 1 minute 22.018 seconds. Exact environment replay also passed.
This completes the corrected inference replay gate; gradient-enabled replay
and backward remain separate checks.

Added `validate-policy-backward`: one saved action (default joint step 3),
training mode with autograd enabled, original `2e-4` replay threshold,
synthetic advantage +1, actual clipped token objective and horizon normalizer,
finite nonzero gradient requirement, and unconditional gradient cleanup.
It constructs no optimizer and writes no artifact. This probe does not establish
full-batch memory feasibility, checkpoint correctness, or learning performance.
Its HPU execution subsequently passed as recorded below.

### Single-action HPU backward gate: passed

The user ran `validate-policy-backward` through `bash env/shell.sh python -m
self_play_grpo.cli` with `--config self_play_grpo/configs/quoridor_fixture.yaml`,
`--input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl`,
`--joint-step 3`, and `--replay-tolerance 2e-4`.

| Measurement | Observed result |
|---|---|
| Status/action | `ok`, `WALL_A1H` |
| Probability path | `kv_cache` |
| Gradient-enabled replay maximum error | `0.0` |
| Mean importance ratio | `0.9999999403953552` |
| Synthetic advantage | `1.0` |
| Loss | `-0.05999999865889549` |
| Gradient norm (unclipped) | `1.5376510660497742` |
| Parameter tensors with gradients | `504` |
| Gradients cleared | `true` |
| Optimizer steps | `0` |
| Wall/user/system time | `37.220 / 74.860 / 5.132` seconds |

This verifies autograd through cached decoding for this saved action in training
mode. The six-token synthetic objective normalized by the fixture horizon of
100 explains the loss near -0.06 at unit importance ratios. A gradient norm
above 1 is valid here: this probe reports the unclipped norm. Actual training
configures clipping at 1.0. This result does not establish full-batch memory
feasibility, checkpoint/resume correctness, or policy improvement.

### Four-action bounded-memory backward gate: passed

The user ran `validate-policy-batch-backward` with the fixture config, preserved
match, `--actions 4`, and replay tolerance `2e-4`. It returned `status: ok` in
41.893 seconds:

| Measurement | Observed result |
|---|---|
| Actions / owned tokens | `4 / 20` |
| Maximum replay error / mean ratio | `0.0 / 1.0` |
| Clip fraction | `0.0` |
| Synthetic loss / gradient norm | `-0.20000000298023224 / 1.9488064050674438` |
| Parameter tensors with gradients | `504` |
| Baseline HPU allocation | `8,183,185,024` bytes |
| Peak HPU allocation | `25,289,977,728` bytes |
| Peak increase | `17,106,792,704` bytes |
| Optimizer steps / gradients cleared | `0 / true` |

The result verifies sequential accumulation over multiple actions and leaves
substantial device-memory headroom. It does not yet prove that allocator usage
stays flat for all 100 actions; the complete-match form is the next memory gate.

### Complete 100-action bounded-memory backward gate: passed

The complete form processed all 100 turns and 416 owned tokens in 3 minutes
7.173 seconds. Replay error was `0.0`, mean ratio was
`0.9999999403953552`, clip fraction was `0.0`, synthetic loss was
`-4.159997463226318`, and gradient norm was `5.128081321716309`. All 504
trainable parameter tensors received gradients; gradients were cleared and
optimizer steps remained zero.

Baseline HPU allocation was `8,182,906,496` bytes and peak allocation was
`25,337,058,048` bytes, an increase of `17,154,151,552` bytes. Compared with
the four-action peak, the 100-action peak increased by only `47,080,320` bytes
(about 0.19%) while action count grew 25-fold. This validates bounded live
autograd-graph memory for one complete fixture match. It does not measure the
wall time of the 16-match main batch, though graph memory remains action-bound.

### Initialization checkpoint round-trip: passed

`validate-checkpoint-roundtrip` completed in 37.509 seconds and wrote format 2
to `artifacts/checkpoint-roundtrip-initial/checkpoints/policy-000000`.
The six files total 132,211,226 bytes: adapter README/config/safetensors,
optimizer state, Torch RNG state, and trainer state. No frozen base-model state
was saved. The adapter parameter restored exactly; behavior replay and
before/after continuation errors were `0.0`; CPU and HPU RNG continuations were
both exact. Optimizer state was intentionally empty and optimizer steps were
zero, so non-empty Adam continuation remains a separate gate.

### Non-empty optimizer checkpoint continuation: passed

`validate-optimizer-checkpoint` completed in 41.134 seconds and wrote a
396,882,614-byte validation-only checkpoint after the first controlled step.
That step changed 252 adapter tensors, had replay error `0.0`, mean ratio
`0.9999999403953552`, loss `-0.05999999865889549`, unclipped gradient norm
`1.561427116394043`, and zero clip fraction.

The command compared an uninterrupted second step with the same step after
reload. Every trainable parameter, all 504 optimizer entries and 1,512 Adam
state tensors, and all scalar metrics were exactly equal. The resumed step had
mean ratio `1.0573149919509888`, loss `-0.06173643097281456`, gradient norm
`0.3543495535850525`, and clip fraction `0.1666666567325592`. Its
behavior-policy distance `0.3149909973144531` is expected after the first
update and confirms that clipping was exercised. Gradients were cleared. The
checkpoint is marked validation-only and normal training resume rejects it.

### Resumable pilot collector smoke: passed

`collect-policy-pilot` completed one bounded 5x5 fixture game in 3 minutes
16.314 seconds. It atomically created
`artifacts/pilot-smoke-qwen3-4b-seed-11`, saved the frozen adapter, committed
`matches/game-000000.jsonl`, replayed the complete environment, and replayed
all 416 owned tokens through the authoritative KV-cache probability path.

The match reached OpenSpiel's 100-action engine draw with results
`[0.25, 0.25, 0.25, 0.25]`. Maximum probability error was `0.0`; illegal
action substitutions were zero; the manifest contained exactly one policy
version, `pilot:1cfa9a720891:c4be16bea2ac`, and reported `status: complete`.
The frozen adapter SHA-256 is
`c4be16bea2ac1006aa1c7a2633008d5efa7dfc76d58707318f6e6bb464600522`;
the match artifact SHA-256 is
`75b1c7aacb08fd81cfcc79923a7b3789b5602bbe8141c5be48e4bd4feec19f32`.
No backward pass, optimizer construction, or parameter update occurred.

This validates the one-game atomic workflow. It is not the plan's 100-game M1
gate, and the fixture draw is not evidence about the 9x9 natural-outcome rate.

### Main 9x9 pilot, first bounded invocation: passed

The first invocation for the declared 100-game 9x9 pilot completed one game in
1 minute 55.029 seconds and correctly reported `status: incomplete`. Seed 11
produced a natural seat-0 win with results `[1.0, 0.0, 0.0, 0.0]` after 45
turns and 180 owned tokens. Cached probability replay error was `0.0`, illegal
action substitutions were zero, and the manifest contained only
`pilot:1cfa9a720891:b3bab9010790`.

The frozen main-pilot adapter SHA-256 is
`b3bab90107908f094e5f236b7897c9c5ae350cdf2d83f6a2916e040ff33f93b9`;
game 0's artifact SHA-256 is
`082c595f3f0cb539a5efb9a1ae4e17b0a6595e72a59079cea35e30a4290ae7dd`.
The manifest target remains 100 games, with 1 completed and 99 pending. This
single natural result shows that nonzero outcome advantages are possible but
is far too small a sample for a natural-outcome-rate estimate.

### Main 9x9 pilot, cross-process resume: passed

The second bounded invocation reloaded and verified the saved adapter, retained
policy version `pilot:1cfa9a720891:b3bab9010790`, and advanced contiguously to
game index 1 and seed 12. It completed in 2 minutes 11.382 seconds. The game was
a natural seat-0 win after 57 turns and 246 owned tokens; probability replay
error was `0.0`, and substitutions remained zero. Its artifact SHA-256 is
`f226bd602bcb3034957ea9014a7a31541fe23b8d4705690ba0bf5ecc0150fa54`.

The manifest now records 2 of 100 games, 102 turns, 426 owned tokens, two
natural wins, no draws, one policy version, and maximum replay error `0.0`.
Both results favor seat 0, but two games are not evidence of a seat effect.

### Main 9x9 pilot, four-game same-process chunk: passed

One bounded invocation collected indices 2 through 5 with seeds 13 through 16
in 7 minutes 24.510 seconds. All four games ended naturally, all probability
replay errors were `0.0`, all substitution counts were zero, and the frozen
policy version remained `pilot:1cfa9a720891:b3bab9010790`.

| Index / seed | Winner | Turns | Owned tokens | Artifact SHA-256 |
|---|---:|---:|---:|---|
| 2 / 13 | seat 2 | 63 | 272 | `7d5d420cb5ff02a659a199a2ab9d14482646bd33211057b03d2e0c8cbcf7e5fc` |
| 3 / 14 | seat 0 | 105 | 448 | `db86eb9f505835cbd4f448aa9f3026072dec153078ca3914b6d9461000b80758` |
| 4 / 15 | seat 2 | 31 | 130 | `93efd303ea4fac98867351eea934dff610b202f663a9493b6477d79c57afb4a5` |
| 5 / 16 | seat 2 | 63 | 268 | `ecd7ba062a7d06d92fbef27cb631b7f0fc25faea9f584af1b50bfdbf993bd2bf` |

The manifest now has 6 of 100 games, 364 turns, 1,544 owned tokens, six
natural wins, no draws, and a 3/3 winner split between seats 0 and 2. Seats 1
and 3 have no wins yet. Six games are insufficient for a seat-effect claim.

### Main 9x9 pilot, additive-summary migration: passed

The next bounded invocation loaded the pre-extension manifest, accepted its
legacy summary, restored the same frozen adapter, and committed index 6/seed 17
in 1 minute 46.404 seconds. The game was a natural seat-2 win after 35 turns
and 146 owned tokens. Replay error was `0.0`, substitutions were zero, and the
artifact SHA-256 is
`281f54ebe1f18750750d31f00dd3db016b493c36fb31d3b79cfd89b80ee93c3f`.

The rewritten manifest now includes the additive summary fields. At 7 of 100
games it reports 399 turns, mean length `57.0`, 1,690 owned tokens, fractional
result sums `[3.0, 0.0, 4.0, 0.0]`, and win counts `[3, 0, 4, 0]`. All seven
games are natural wins and no draw has occurred. The observed restriction of
winners to seats 0 and 2 remains descriptive, not a conclusion.

### Main 9x9 pilot, eight-game chunk: passed

One bounded invocation collected indices 7 through 14 with seeds 18 through 25
in 12 minutes 30.631 seconds. All eight games terminated in natural wins, every
probability replay error was `0.0`, substitutions remained zero, and the policy
version and adapter digest were unchanged.

| Index / seed | Winner | Turns | Owned tokens | Artifact SHA-256 |
|---|---:|---:|---:|---|
| 7 / 18 | seat 3 | 80 | 336 | `59d3e370d5c06405cbc19073f2666a1058ba413839c414cb695968483935dfc6` |
| 8 / 19 | seat 0 | 65 | 286 | `7e7bf60ac18f4b9c14c88c49a735024e95bc78f975df1a58dd6148c4ea32c506` |
| 9 / 20 | seat 2 | 47 | 202 | `5adfbce0e70eff2ece895ed9254f6bcfffa7516183c507e8d81af02537ae9801` |
| 10 / 21 | seat 2 | 35 | 144 | `f04c619c5ca503d5d1ced22fa4ae543a29e8a1c5b3bfd934af826aaa4f76a585` |
| 11 / 22 | seat 2 | 31 | 128 | `e2bdb418f071d5ec65611bdd165d1e9f94531c9d130eda3af6612757d072ee87` |
| 12 / 23 | seat 3 | 72 | 314 | `ca329de5003fbd435d919d1f48b6145b600972439b919f899c725da1c590e98e` |
| 13 / 24 | seat 3 | 40 | 166 | `77ec677d4e5ae84026e1e4eb61f3c6f54a8411c74a6c7e93acbc5f2c26d2ab6f` |
| 14 / 25 | seat 2 | 63 | 268 | `ed0fa165feac03165a829c4bf3617779f4e6b65d25ff094ec633750d8a0774ca` |

At 15 of 100 games, the manifest records 832 turns, mean length
`55.46666666666667`, 3,534 owned tokens, no draws, wins `[4,0,8,3]`, one
policy version, zero substitutions, and maximum replay error `0.0`. Seat 1 has
not yet won, which should be monitored but is not a conclusion at 15 games.

### Main 9x9 pilot, second eight-game chunk: passed

Indices 15 through 22 (seeds 26 through 33) completed in 15 minutes 36.568
seconds under the unchanged adapter and policy version. Replay error was `0.0`
for every game and no illegal action substitution occurred. Seed 31 produced
the pilot's first 120-turn horizon draw; the other seven games ended naturally.

| Index / seed | Result | Turns | Owned tokens | Artifact SHA-256 |
|---|---|---:|---:|---|
| 15 / 26 | seat 3 win | 64 | 266 | `8bfa8fa29e29558ca7b586579c45b72c26613482ccf4527851b4b2944f13c504` |
| 16 / 27 | seat 3 win | 56 | 240 | `ff2e2b415621c2116dd49421218402896c8d9fb488be968e1fbff3e17539e02c` |
| 17 / 28 | seat 2 win | 47 | 196 | `b2a57400c8a3c4df1ce30deb8799b9ab960e8a918e80883a1133f936b5514da2` |
| 18 / 29 | seat 0 win | 37 | 152 | `d268929193d578768d1877a37043c335cca0bc7a701360fb76d16ae5dacee96b` |
| 19 / 30 | seat 2 win | 31 | 128 | `0fe8e19171a912ae6ef6e1e728e825958637f0450cfbd3600e763c928fef501f` |
| 20 / 31 | horizon draw | 120 | 520 | `aec27e2f306b569e9939d7cba47bfb77ea3496c9887ef06b21c55dfc8393cf1f` |
| 21 / 32 | seat 2 win | 51 | 212 | `92d7f67478842818f0ac1dcb9561059ee3042007b6cde838e7b2ef2958fce0e0` |
| 22 / 33 | seat 3 win | 84 | 366 | `0251a2a1e7551848c5c39ae95179f1d911dd14735629018a48f9db45f7835427` |

At 23 of 100 games, the manifest records 1,322 turns, mean length
`57.47826086956522`, 5,614 owned tokens, 22 natural wins, one draw, fractional
result sums `[5.25,0.25,11.25,6.25]`, and wins `[5,0,11,6]`. Seat 1 still has
no natural win; this remains a monitored pilot observation rather than a causal
seat-effect claim.

### Main 9x9 pilot, third eight-game chunk: passed

Indices 23 through 30 (seeds 34 through 41) completed in 11 minutes 31.510
seconds. All eight games ended naturally, replay errors were `0.0`, no illegal
substitution occurred, and the frozen adapter and policy version were
unchanged. Seven games were won by seat 2 and one by seat 3.

| Index / seed | Winner | Turns | Owned tokens | Artifact SHA-256 |
|---|---:|---:|---:|---|
| 23 / 34 | seat 2 | 35 | 142 | `d8fdc8dbd69d73e25c9d4b8415bdfd985d5713b90510af8c6cc9f99e0efe56e4` |
| 24 / 35 | seat 2 | 31 | 124 | `71e55d3fc16df1912d29d8d58b914f5759ad1a39b782f23372416c52e18e4710` |
| 25 / 36 | seat 2 | 47 | 192 | `247ccba3724a11be21f62b71d05bce06f5ab10bc2f1c85d3c1d64945e2727674` |
| 26 / 37 | seat 3 | 64 | 270 | `efdc0487bf6bdfe3a1f42201c7e39c9205f35b60ad7a9ba8e6c30c13000609de` |
| 27 / 38 | seat 2 | 39 | 160 | `157830fbfd95c4fd8d00776c5883271d0ac63dbc77749712c0f51a45d50469b1` |
| 28 / 39 | seat 2 | 103 | 442 | `bdd4d1b2a378c57670c8e7de13df2d617fa9465a99fce723d0cad1a1b4473101` |
| 29 / 40 | seat 2 | 35 | 148 | `088fa32080bf21d5e0e00c81a6572f60b27ed29df0d2e8c8ad877216fb0fa6a8` |
| 30 / 41 | seat 2 | 39 | 164 | `eb7ceddef15a527f68cafc01617d44a847046cb8885b5551edb9fd7f5d2be87e` |

At 31 of 100 games, cumulative totals are 1,715 turns, mean length
`55.32258064516129`, 7,256 owned tokens, 30 natural wins, one draw, fractional
results `[5.25,0.25,18.25,7.25]`, and wins `[5,0,18,7]`. Seat 1's zero-win
count is now a material pilot observation to report and investigate alongside
seat-rotated external evaluation; it does not indicate an engine impossibility,
because the deterministic engine contract includes a natural seat-1 win.

### Main 9x9 pilot, fourth eight-game chunk: passed

Indices 31 through 38 (seeds 42 through 49) completed in 12 minutes 0.598
seconds. All eight games ended naturally, replay errors were `0.0`, no illegal
substitution occurred, and the frozen adapter and policy version remained
unchanged. Four games were won by seat 2, three by seat 3, and one by seat 0.

| Index / seed | Winner | Turns | Owned tokens | Artifact SHA-256 |
|---|---:|---:|---:|---|
| 31 / 42 | seat 3 | 48 | 206 | `174ce83e5383019a140545e083881093cc477efdcb2514f4675c0b1ba1db739d` |
| 32 / 43 | seat 2 | 43 | 176 | `6b34283fb84534de94d11ceb068d6570d82c947abffdc5bb60198c79bb4a8594` |
| 33 / 44 | seat 0 | 53 | 224 | `c7445030cf65b5633c6606c78f1354dcb248a5200caf10281805cc7a27c2f736` |
| 34 / 45 | seat 3 | 68 | 290 | `8f0798927541fdf96a7419cb09a8a00fd014674e05d996fa473b9e4530a037c6` |
| 35 / 46 | seat 3 | 48 | 214 | `69482709b01e076162180dcdb1d202f524e5c1a18697e2de148cdad7a46a3d18` |
| 36 / 47 | seat 2 | 47 | 200 | `acaa134f88e551395f99a1a71f529f4859afd6a26c3bd5e3207c642cfda4f6e6` |
| 37 / 48 | seat 2 | 39 | 158 | `0f80ff97b28a005306a26a3d2def6de7e3cf365b717277ee1a4c156a46d5e67f` |
| 38 / 49 | seat 2 | 35 | 142 | `2b8b59249e8b808a0cddd7e620f0da880e20d02d4c86f04ed8ba102aa51af51d` |

At 39 of 100 games, cumulative totals are 2,096 turns, mean length
`53.743589743589745`, 8,866 owned tokens, 38 natural wins, one horizon draw,
fractional results `[6.25,0.25,22.25,10.25]`, and wins `[6,0,22,10]`.
Probability replay remains exact and the pilot still contains one policy
version. Seat 1 still has no natural win after 39 games, so the planned
seat-rotated evaluation remains a required diagnostic before interpreting the
imbalance as a policy or environment effect.

### Main 9x9 pilot, fifth eight-game chunk: passed

Indices 39 through 46 (seeds 50 through 57) completed in 10 minutes 39.932
seconds. All eight games ended naturally, replay errors were `0.0`, no illegal
substitution occurred, and the adapter digest and policy version remained
unchanged. Six games were won by seat 2, one by seat 3, and one by seat 0.

| Index / seed | Winner | Turns | Owned tokens | Artifact SHA-256 |
|---|---:|---:|---:|---|
| 39 / 50 | seat 2 | 39 | 168 | `4ce38e385d5d586e0e74b0d476f9bd5a515c605faf100a08b909c32ba3832729` |
| 40 / 51 | seat 3 | 60 | 254 | `7553473c948e0151ad1ad3d1b7f4165cf58f13d03ebe08652b4343a3a98a41b7` |
| 41 / 52 | seat 2 | 31 | 130 | `4737127acbd19ecb51cde44d8f0184e7fe821fa8bf4991f9ac050672908cada6` |
| 42 / 53 | seat 2 | 51 | 218 | `3f1c17fe556926c9e67792b18f6346a9f0d4ca91839e54688c33a782478e2564` |
| 43 / 54 | seat 0 | 49 | 202 | `5f150d666b0f3f1141a6025b3a1c3373c678829b35f00cd991c83d68d51dd179` |
| 44 / 55 | seat 2 | 39 | 166 | `bbe5a7df87c423bacb265992fa6ba0ac1c85265ad5c6ed1ced86d3509e4c011e` |
| 45 / 56 | seat 2 | 51 | 218 | `a219232a6e6bec0aae087f37f4f770be84d771642d1d27604b32a797526c4038` |
| 46 / 57 | seat 2 | 39 | 158 | `ce5f61f5272328f639219778e33fce0e41abb573debdd609662dcbd42bcf18dd` |

At 47 of 100 games, cumulative totals are 2,455 turns, mean length
`52.234042553191486`, 10,380 owned tokens, 46 natural wins, one horizon draw,
fractional results `[7.25,0.25,28.25,11.25]`, and wins `[7,0,28,11]`.
Probability replay remains exact and the pilot still contains one policy
version. Seat 1 still has no natural win after 47 games and remains a required
seat-rotated-evaluation diagnostic.

### Main 9x9 pilot, sixth eight-game chunk: passed

Indices 47 through 54 (seeds 58 through 65) completed in 10 minutes 7.717
seconds. All eight games ended naturally, replay errors were `0.0`, no illegal
substitution occurred, and the adapter digest and policy version remained
unchanged. Seven games were won by seat 2 and one by seat 3.

| Index / seed | Winner | Turns | Owned tokens | Artifact SHA-256 |
|---|---:|---:|---:|---|
| 47 / 58 | seat 2 | 55 | 232 | `076654b7bd2151fd3029835444cdf1b78d88983cddef14397de4aa5082d3a8c3` |
| 48 / 59 | seat 2 | 39 | 164 | `b65d9ccf24026b645cdc9b2626a700a3592d629a97eaea79611caac95537988b` |
| 49 / 60 | seat 3 | 44 | 184 | `08713f00ec4a0cacf10e17734cd85a6a1a0ba5ec504e63aa249efa7c21e31174` |
| 50 / 61 | seat 2 | 43 | 174 | `c4bb839dda7b6b88b66e45f7573f361be2f47368a255908918a672432f5a4c5e` |
| 51 / 62 | seat 2 | 51 | 216 | `34a598a4f665e6e79631b1c7dbe2bf8c039bb7e7e323f71545151c3ee36c047d` |
| 52 / 63 | seat 2 | 35 | 144 | `f701aebf2d3ac308a283fc7690e4f15e14802fbf539dfbf97be19a0a35075c38` |
| 53 / 64 | seat 2 | 35 | 144 | `54b41b6b9c63bced56b0f341cac20202865f21c564663c362aa28ccba5852b74` |
| 54 / 65 | seat 2 | 31 | 128 | `6e84a902d7da279e01ab5ff7e084bde384e80b2c7f80dbbf57a476c510fee239` |

At 55 of 100 games, cumulative totals are 2,788 turns, mean length
`50.69090909090909`, 11,766 owned tokens, 54 natural wins, one horizon draw,
fractional results `[7.25,0.25,35.25,12.25]`, and wins `[7,0,35,12]`.
Probability replay remains exact and the pilot still contains one policy
version. Seat 1 still has no natural win after 55 games and remains a required
seat-rotated-evaluation diagnostic.

### Main 9x9 pilot, seventh eight-game chunk: passed; pilot paused

Indices 55 through 62 (seeds 66 through 73) completed in 11 minutes 35.401
seconds. All eight games ended naturally, replay errors were `0.0`, no illegal
substitution occurred, and the adapter digest and policy version remained
unchanged. Seven games were won by seat 2 and one by seat 3.

| Index / seed | Winner | Turns | Owned tokens | Artifact SHA-256 |
|---|---:|---:|---:|---|
| 55 / 66 | seat 2 | 31 | 126 | `a614158f3ff8a9403d79c249536fc6fd87d8db69cce38a11b98f669692f70882` |
| 56 / 67 | seat 2 | 51 | 226 | `3b4874ad706f0f2a8810d58e49fb79ec08d56841eeeb77cfc68b670cb2716286` |
| 57 / 68 | seat 2 | 35 | 146 | `60a662e153d9d7d02a40080450614607fb7c2aad23eee8ea03a2a1fdf569f4e8` |
| 58 / 69 | seat 2 | 43 | 176 | `fe35f8d8716e13a4ac48a5a1c584d8547a062c82e330cb62b177dc8c1fb1d275` |
| 59 / 70 | seat 3 | 48 | 202 | `fc45ba59ae2f0f26efb392a5d1a0ca54cd7f38ac8c65699703818e4d9137ec67` |
| 60 / 71 | seat 2 | 47 | 194 | `654979f796aa4aaa6ff96a29ba06ba24654956e413fb8e0effb490691e21a2dc` |
| 61 / 72 | seat 2 | 47 | 200 | `3d9b25257fb095ff1ce840ab238cedd34a7961a1285081b02244853443791566` |
| 62 / 73 | seat 2 | 91 | 376 | `96529c3d526e6239514d8958e453890c09a228f0bffba6f6fbe1bb54f707785e` |

At 63 of 100 games, cumulative totals are 3,181 turns, mean length
`50.492063492063494`, 13,412 owned tokens, 62 natural wins, one horizon draw,
fractional results `[7.25,0.25,42.25,13.25]`, and wins `[7,0,42,13]`.

Collection is intentionally paused at this point. A read-only aggregate of
all recorded turns found the following goal-oriented movement counts:

| Seat | Actions | Moves | Walls | Goal-forward | Goal-backward | Lateral | Net goal progress |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 811 | 714 | 97 | 329 | 43 | 342 | 295 |
| 1 | 804 | 655 | 149 | 37 | 17 | 601 | 22 |
| 2 | 804 | 734 | 70 | 490 | 93 | 151 | 398 |
| 3 | 762 | 734 | 28 | 311 | 16 | 407 | 297 |

The prompt and canonical-to-engine mapping are internally consistent in the
inspected artifacts: seat 1 is told to reach the right edge and seat 2 to reach
the bottom edge. The observed failure is instead strongly associated with the
absolute action-label policy. On the first decision of each game, seat 1 chose
`MOVE_A4` 61 times, `MOVE_B5` once, and `WALL_A4V` once; only `MOVE_B5` makes
goal progress. Seat 2 made 490 forward moves and won 42 games. The old pilot
therefore remains valid diagnostic data, but it is permanently archived:
player-relative actions changed policy semantics, and the fresh pilot will use
the batched rollout path after its complete-match HPU smoke passes.

## Test catalog

The last executed aggregate non-model suite contains 92 passing cases,
including the bounded-memory, pilot-manifest, pilot-analysis, engine,
player-relative-coordinate, distributed runtime/sharding, synthetic-gradient,
finite-probability, and batched-collector contracts. It completed in 10.23
seconds with three expected Gaudi warnings and no skipped tests.

| File | Cases | Contract protected |
|---|---:|---|
| `test_config.py` | 3 | Engine/model revisions, local model path, horizons/concurrency are pinned, and nonpositive concurrency is rejected |
| `test_distributed.py` | 13 | Fail-closed torchrun variable parsing, expected global/local world sizes, rank/local-rank ranges, and rejection of one-process or multi-node distributed gates |
| `test_distributed_rollout.py` | 4 | Disjoint deterministic rank shards, global ordering, single-node local-rank identity, and rank-report persistence |
| `test_distributed_training.py` | 10 | Equal intact-match trainer shards, divisibility/rank validation, complete/config-identical rollout manifests, and atomic trainer-rank reports |
| `test_batched_collector.py` | 2 | Multi-game stepping, dynamic compaction, original ordering, independent seeds/results, and duplicate game-ID rejection |
| `test_engine_contract.py` | 8 | Seat order, legal label/ID round trip, complete match replay, forced-pass backport, and correct reward ownership for every seat |
| `test_loss_math.py` | 5 | Positive/negative advantage gradient direction, ratio one, owned-token masking, streaming-versus-summed gradient equality, and destruction of the previous action graph before the next replay |
| `test_outcome.py` | 3 | Closed-form four-seat advantages, zero draw advantages, and utility/result order |
| `test_progress.py` | 6 | Goal orientations, wall blocking, proxy values, shaping telescoping/cancellation, and terminal GAE boundary |
| `test_rollout_analysis.py` | 2 | Per-seat goal-direction accounting, opening-action summaries, winner accounting, and empty-input rejection |
| `test_relative_observations.py` | 14 | All four rotations share one starting perspective and forward label; move/wall rotations are bijective; engine IDs survive; prompts omit canonical-seat leakage |
| `test_schema.py` | 7 | Stable match JSON, owner and grammar validation, plus pilot manifest round trip, frozen-policy/contiguity rejection, monotonic target behavior, and atomic root initialization |
| `test_token_trie.py` | 8 | Exact legal prefix continuations, grammar rejection, complete labels, single/batched probability-path dispatch, and fail-closed finite behavior/replay diagnostics |

## Recovered failures and their resolution

| Symptom | Cause | Resolution |
|---|---|---|
| `rg: command not found` | Ripgrep is not installed on the host | Used targeted `find`, `grep`, and `sed` inspection |
| OpenSpiel source download appeared stuck at “Installing build dependencies” | `pip download` invoked PEP 517 build isolation just to fetch an sdist | Replaced it with direct `curl` plus a pinned SHA-256 |
| Git checkout build lacked `abseil-cpp` and `json` | The source checkout did not include packaged C++ dependencies | Build from the dependency-complete PyPI source archive |
| OpenSpiel patch was malformed | Incorrect unified-diff hunk counts | Corrected hunk counts and added interrupted-patch recovery |
| Installer rejected installed version `2.0.2` | `pyproject.toml` owns wheel metadata, not the patched `setup.py` field | Retained upstream metadata and verify the backport behaviorally |
| Qwen3-1.7B cache missing | The originally proposed checkpoint was not local | Adopted the requested, pinned Qwen3-4B checkpoint |
| PEFT LoRA injection imported incompatible INC code | PEFT probes INC whenever Neural Compressor is installed; the installed INC expects the old Transformers `Conv1D` location | Bypass INC only for the unquantized BF16 load and use generic dense LoRA |
| Complete-match probability replay error `0.157623` | The sampling KV-cache path reproduced all 416 tokens exactly; only the differently shaped full-sequence HPU BF16 path diverged | Made KV-cache decoding authoritative for sampling/replay/loss, retained full-sequence calculation for diagnostics, and kept the original tolerance |
| First padded batch reported `NaN` but returned success | IEEE comparisons with NaN make both `<=` and `>` false, so the status said mismatch while the old threshold branch did not raise | Treat every non-finite behavior or replay value as an unconditional failure, emit its row/token positions, and serialize it as JSON `null` rather than nonstandard NaN |
| New mixed-policy manifest test expected a different rejection message | The derived game-ID check rejected the record before the explicit policy-version check | Moved the policy-version check earlier; the same invalid record now fails with the direct mixed-version diagnostic |
| Atomic-root test failed while creating its fake adapter | The fake `save_pretrained` implementation did not accept the already-created temporary destination that real PEFT accepts | Made the test double use `exist_ok=True`; atomic directory initialization then passed |
| Four-HPU gameplay launch failed immediately: `~/.local/bin/torchrun: cannot execute` | The fallback host launcher has a shebang to `/packages/apps/mamba/2.0.8/bin/python3.12`, which is absent inside the container; the project venv has no `torchrun` console script | Use `python -m torch.distributed.run --module ...`, which invokes the pinned container Torch without the stale host wrapper; no HCCL or model work occurred in the failed attempt |
| First four-rank HCCL contract rejected device mapping after successful reductions | The pinned Habana eager/GPU-migration bridge rejects indexed tensor devices such as `hpu:1`, and native `torch.hpu.current_device()` stays at zero rather than reporting a physical module ID. Rank and scalar HCCL reductions had already succeeded. | The first correction switched tensors to unindexed `hpu` and tried the repository's `torch.cuda` migration shim for selection. The run stopped before model loading or artifact creation. |
| Second four-rank HCCL contract still reported current device zero on ranks 1–3 | The 34.113-second retry again passed rank/scalar reductions but showed that the installed `initialize_distributed_hpu` maps `LOCAL_RANK` to the physical module through `HLS_MODULE_ID`; each bound process then intentionally exposes its assigned module as logical device zero. Neither native nor migrated `current_device()` is a physical-module identity check. | Validate the exact Habana contract directly: initializer return values must match torchrun metadata, `HLS_MODULE_ID` must equal the local-rank mapping (or its `HABANA_VISIBLE_MODULES` entry), logical-rank one-hots must be unique, HCCL reductions must succeed, and tensors use unindexed `hpu`. No model or artifact was created. |

## Expected warnings

These have been observed and are not current test failures:

- OpenSpiel prints that Quoridor has known issues. The specific forced-pass bug
  relevant here is covered by the pinned regression test; the warning itself is
  still emitted upstream.
- Habana reports that `add_step_closure`, `mark_step`, and `iter_mark_step` have
  no effect in eager mode (`PT_HPU_LAZY_MODE=0`).
- Habana imports emit `pkg_resources` and `pyhlml` deprecation warnings.
- Habana GPU migration reports that Apex is not installed.
- With `PT_HPU_GPU_MIGRATION=1`, `dist.get_backend()` reports the migrated
  compatibility name `nccl` even though Habana initialization, physical
  `HLS_MODULE_ID` bindings, HPU tensors, and collectives use the Gaudi path.
- Hugging Face may suggest upgrading the CLI, installing an agent skill, or
  authenticating for higher download limits. None is required for the pinned
  local workflow, and the environment was not changed.

## Pending gates and limitations

The following must not be represented as completed:

1. Preserve the original absolute-coordinate pilot as diagnostic evidence;
   the player-relative semantic fix is implemented, so that manifest must not
   be resumed.
2. Preserve the fresh player-relative pilot at its current valid 8/100
   checkpoint. Its perfect eight-game seat balance is a screening result, not
   a statistical performance claim; completion to 100 remains resumable but is
   not required before the bounded distributed runtime gates.
3. Implement and run the bounded D3 four-trainer real-artifact update gate.
4. Implement and pass the remaining distributed checkpoint contracts
   in `EIGHT_HPU_TRAINING_DESIGN.md`.
5. Complete a short eight-HPU outcome-only training run after the pilot
   establishes a
   usable natural-outcome rate.
6. Freeze and record the initial-model external evaluation baseline.
7. Train and validate the optional evaluator only after outcome-only baseline
   data exists.
8. No empirical learning, convergence, or strength claim has been made.

When a new gate is run, append its exact command, configuration, result, timing,
and artifact path here. Do not overwrite failures; record the failure and its
resolution so future runs retain the diagnostic history.
