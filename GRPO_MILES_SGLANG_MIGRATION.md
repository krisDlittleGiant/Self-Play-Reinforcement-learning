# VERL with the Miles Gaudi SGLang patches

Prepared 2026-09-06. Hardware generation and distributed stability must be verified
with the staged commands below; the earlier successful optimizer step used the old runtime.

## Runtime and purpose

SGLang is pinned to `cb05a44f35a7c9e27e46d74112cc841ca674ef43` plus three patches applied
in order: `env/patches/sglang-miles-cb05a44-gaudi.patch`,
`env/patches/sglang-cb05a44-hpu-container-compat.patch`, then
`env/patches/sglang-cb05a44-verl-hpu-runtime.patch`.

The Miles patch is **regenerated from the live `/scratch/$USER/sglang-miles` checkout**,
not copied from Miles' own saved `patches/sglang-hpu-gaudi-full.patch`. That saved file is a
stale snapshot: the live checkout is substantially ahead of it. Using the snapshot would have
left out `hpu_graph_runner.py` at 934 lines instead of 421, the HPU paged allocator work in
`mem_cache/allocator/paged.py` (433 vs 292 lines), and three attention backends that did not
exist in it at all -- `hpu_attention_utils.py`, `hpu_blockwise_backend.py` and
`hpu_paged_v2_backend.py` -- plus newer `sampler.py`, `logprob_processor.py`,
`attention_registry.py`, `chunk_cache.py`, `server_args.py` and `common.py`. The regenerated
patch reproduces the live Miles tree byte-for-byte from the pinned commit. The third patch
contains the HPU scalar-control fixes found by the VERL GRPO tests. These repository files
are now the source of truth: the external Miles checkout is provenance only and is not
required at runtime. Reconstruction was verified by applying all three patches to a clean
pinned checkout and comparing the resulting files with the tested runtime byte-for-byte.

The patch contains HPU attention, decode graph execution, eager prefill routing, scheduler
tensor retention and execution boundaries, HPU-safe token IDs, RoPE, memory-pool/allocation
fixes, and HPU weight-update handling. These address the suspect paths behind
`ValidateSyncInputTensors tensor_data is empty`; a full GRPO run is still needed to establish
that the migration resolves that failure in VERL.

## Container compatibility (not needed by Miles)

Miles runs host-native on Synapse 1.24.1 with Torch 2.11.0a0 and never enables Habana's
gpu_migration layer. VERL runs inside the Synapse 1.22.2 container on Torch 2.7.1+hpu and
does enable migration for its FSDP actors. Three incompatibilities follow from that gap, and
each is fixed rather than worked around:

1. `pynccl_allocator.py` imports `_cuda_beginAllocateCurrentThreadToPool`, which exists only
   in Torch >= 2.8. The import is now guarded; the file already gated its two real uses behind
   `after_2_8_0`, and every use sits on CUDA symmetric-memory paths HPU never enters.
2. `is_cuda()` in `utils/common.py` returned true under gpu_migration, because migration makes
   `torch.cuda.is_available()` true and spoofs `torch.version.cuda` to "11.8". SGLang then
   imported CUDA-only kernels (`flashinfer.sampling`, `sgl_kernel`) that are not installed on
   Gaudi. It now also requires `not is_hpu()`; on a real CUDA host `torch.hpu` does not exist,
   so the change is a no-op there.
3. `cuda_graph_max_bs` was split upstream into `cuda_graph_max_bs_decode` / `_prefill`. The
   launcher and both verification scripts now pass the decode variant, matching Miles'
   `--sglang-cuda-graph-max-bs-decode`.

Setup builds the generated `.runtime/sglang-miles` checkout and `.runtime/venv` inside this
repository from the pinned upstream revision and repository-owned patches. Your existing
Miles checkout is preserved and untouched; `/scratch/$USER/sglang-miles`, `sglang-fork`,
and the old `/scratch/$USER/venvs/verl-gaudi` environment are not runtime dependencies.
Generated runtime directories are Git-ignored.

The dependency file is a pinned union of the local Miles and VERL environments, retaining
VERL's Ray 2.53.0 and moving Transformers to 5.12.1, which additionally needed a small
annotation fix where Habana's migration layer exposes `CUDAGraph` as a function. All installs
use `uv --no-deps`; Torch and torchdata are supplied by the Habana container. Installing
upstream `sglang[srt]` would instead request CUDA dependencies and an incompatible Torch.

The rollout processes also export Miles' decode-graph bucketing and warmup values
(`SGLANG_HPU_DECODE_BATCH_BUCKET_STEP=32`, `SGLANG_HPU_DENSE_DECODE_SEQ_BUCKET_STEP=128`,
warmup batches `1,16,32,64`, warmup sequence lengths 512) copied from the Miles run that
reached iteration 50, so the captured decode buckets match the validated configuration.

## Configuration

| Component | Setting |
|---|---|
| Model | `Qwen/Qwen3-4B-Base` |
| Algorithm / dataset | GRPO / GSM8K |
| Training | Four FSDP HPUs, FP32 parameters, BF16 compute, repaired Habana FusedSDPA |
| Rollout | Four dedicated HPUs, TP=1, `hpu_fused` attention, PyTorch sampling |
| Execution | Actor eager with GPU migration; SGLang children lazy with native HPU APIs |
| Graphs | Prefill disabled; decode full; overlap disabled on Synapse 1.22 |
| Batch | 32 prompts × 8 samples; at most 64 active requests/server |
| Lengths | 512 prompt / 1024 response; context rounded up to a 128-token boundary |
| Validation | Before-training and periodic validation disabled; training rewards enabled |
| Checkpoints | Every 10 steps for soak/full; unique experiment directory per invocation |

The VERL adapter preserves CPU-staged weight transfer for legacy FSDP1, CUDA, and as an
explicit fallback. HPU FSDP2 now uses the Miles disaggregated design instead: FSDP rank 0
and every SGLang rank form a separate HCCL group, and each materialized weight bucket is
broadcast directly HPU-to-HPU. This avoids copying and independently serializing the full
FP32 4B model from every actor rank, which stalled the initial update before rollout step 1.
As in Miles' FP32-master export path, floating parameters are converted to their outbound
rollout dtype after gathering and before broadcast; this workflow sends BF16 to the BF16
SGLang model instead of making its lazy worker convert the entire FP32 model. The SGLang
scheduler calls `mark_step()` after loading every bucket so lazy parameter copies are
submitted instead of accumulating through layer 26. Do not put a device-wide
`torch.hpu.synchronize()` at that boundary: live tests showed it hangs on bucket 0 with
lazy HCCL collectives. On the sender, HTTP runs on background event-loop threads while
HCCL is issued and waited on the actor's accelerator thread, matching Miles' RPC/broadcast
ordering; waiting for HCCL from a generic executor thread also stalls Habana.
The three control transitions around that stream (session opened, all buckets sent, and
session closed) use a separate CPU/Gloo actor group, exactly as Miles' FSDP updater does.
They must not use the default HCCL/FSDP group. In the failing run, all 146 Qwen3-4B weight
buckets completed before the four actor HPUs remained in a collective until Synapse killed
them with `[No progress error]`. `VERL_HPU_WEIGHT_SYNC_DEBUG=1` prints entry/completion for each
Gloo barrier and for `end_weight_update`, so a future failure identifies the exact stage.
Cache invalidation also follows Miles' ordering: after generation is paused, rank 0 flushes
all rollout replicas before `begin_weight_update`. There is no second cache flush after
`end_weight_update`; that redundant post-update call was observed to wedge all four SGLang
schedulers after the weight stream and Gloo barriers had already succeeded.
For the same distributed HPU path, the actor does not immediately force
`empty_cache(); synchronize()` after returning from weight sync. Miles has no such boundary,
and the actor owns its training HPU, so released full-tensor buffers may remain cached for
reuse; VERL's existing trainer-mode cleanup still runs before training resumes.
SGLang's non-Triton `compute_position_torch` fallback must also use the host-side prefix and
extend-length lists already carried by `ScheduleBatch`. Iterating their HPU tensor copies
passes device scalars as `torch.arange` bounds, which invokes `_local_scalar_dense`; all four
schedulers were observed there until the 300-second watchdog killed them immediately after
a successful 146-bucket weight update. The patched path keeps the output tensor on HPU but
uses Python integer bounds. A one-HPU regression with prefix lengths `[3, 10]` and extend
lengths `[2, 3]` returned positions `[3, 4, 10, 11, 12]` and start offsets `[0, 2]` with exit
status zero.
The same rule applies inside `TorchNativeAttnBackend`: query slice bounds, temporary tensor
shapes, KV-cache row indices, and decode lengths must come from `seq_lens_cpu`,
`req_pool_indices_cpu`, `extend_prefix_lens_cpu`, `extend_seq_lens_cpu`, and
`encoder_lens_cpu`. Otherwise the first prefill advances past position construction but
wedges at `query[:, start_q:end_q, :]` while Python extracts an HPU scalar for the slice.
`ForwardBatch` now preserves the request-pool CPU mirror, and both native extend and decode
loops prefer these host values for Python control flow. A one-HPU regression exercised the
actual SDPA extend and decode functions and returned finite outputs from both with exit zero.
After those scalar fixes, the next first-token watchdog exposed a different boundary:
`GenerationBatchResult.copy_to_cpu()` moved `next_token_ids` from the HPU forward stream to
SGLang's CUDA-oriented copy stream and called `.to("cpu", non_blocking=True)`. All four
schedulers remained in `copy_hpu_lazy_D2H` until the 300-second watchdog killed them. On
this HPU backend, that non-CUDA helper does not provide a useful asynchronous copy, and a
decode-graph result cannot safely use the CUDA cross-stream path. HPU results now remain on
their producing forward stream and use `non_blocking=False`. A follow-up run proved that
these two changes alone were insufficient: the stack moved to the new same-stream line but
waited in `HbExecutionContext::JoinPendingLaunchThread`, and the watchdog showed a prefill
batch of 63 requests with empty output IDs. Habana `mark_step()` defaults to `sync=False`,
so it had submitted the lazy prefill without completing it before the mandatory host read.
A test with `mark_step(sync=True)` still blocked at the host copy. The boundary now follows
the explicit pattern already used by Miles' HPU sampler and graph runner: `mark_step()`
followed by `htorch.hpu.synchronize()` before D2H. CUDA's pinned asynchronous copy path is
unchanged. The source preflight asserts all three parts of this contract.
Distributed collective HTTP calls are deliberately
single-attempt: retrying a timed-out receive without issuing a matching second broadcast
corrupts collective ordering and cannot recover the session.
Set `VERL_HPU_WEIGHT_SYNC_TRANSPORT=tensor` only to reproduce/diagnose the former path.
The GSM8K launcher now explicitly defaults to `distributed` for its full-model HPU
workflow, including FSDP1. The worker's automatic selection alone chooses distributed
transfer only for FSDP2. The September 8 10:17 run omitted the explicit setting and
selected FSDP1/tensor transfer; it ended with a Ray keepalive timeout without a completed
training step. This proves configuration drift, but the surviving log does not identify
the exact operation that blocked TaskRunner. Restore distributed transfer before drawing
conclusions about the post-forward synchronization patch. The launcher prints its selected
transport and defaults temporary files and Habana logs to per-user directories under
`/tmp`; `VERL_TMPDIR` and `VERL_HABANA_LOGS` remain available as explicit overrides and
are forwarded through the container wrapper.

The first distributed-transfer retry completed all 146 HCCL buckets and all Gloo/session
barriers, then all four SGLang schedulers stopped in the first prefill while copying
`next_token_logprobs`. Disabling rollout log-probabilities merely moved the same block to
the mandatory `next_token_ids` copy, proving that the general result D2H boundary is the
problem. Rollout log-probabilities therefore remain enabled by default.
An explicit `mark_step(); torch.hpu.synchronize()` moved the observed block earlier and
made the real boundary unambiguous: all four schedulers waited in
`HbExecutionContext::JoinPendingLaunchThread` at `scheduler.run_batch`, before result D2H.
The watchdog reported 63 requests in every first prefill batch, while SGLang still had more
than 338k free token slots. Habana DFA was triggered only by Ray's later SIGTERM cleanup;
its compute queues were complete and idle, so this was not an HPU fault or OOM. The run was
using `event_loop_overlap`. On the Synapse 1.22 lazy runtime, non-overlap is therefore the
safe default while standalone 1-scheduler and staged 1+1, 2+2, and 4+4 gates are rerun.
The adapter also handles changed server-launch return values and pins Ray workers to the
driver's Python environment. Runtime source checks precede training.

## Staged HPU verification

Do not use the full 4+4 GRPO job as the first test of a scheduler change. Run these gates
in order and stop at the first failure:

1. one standalone SGLang scheduler, one request, no weight update;
2. the same scheduler with 64 concurrent requests, matching the full per-server batch;
3. one FSDP trainer plus one rollout server (`N_GPUS_PER_NODE=1`);
4. two trainers plus two rollout servers;
5. four trainers plus four rollout servers with a small prompt batch;
6. the required four-plus-four, 32-prompt, eight-rollout configuration.

This separates a model-forward problem from concurrency, distributed weight transfer, and
multi-rank FSDP. A prior 4+4 Qwen3-4B run on the older rollout runtime already verified the
training/FusedSDPA path with finite `pg_loss=-0.02729` and `grad_norm=8.92047`; it does not
verify the newly migrated scheduler. `env/verify_sglang_generation.py` now disables overlap
by default and accepts `--overlap` only for a deliberate A/B comparison.

The first standalone gate completed on physical HPU 7 with concurrency 1,
`MAX_RESPONSE_LENGTH=2048`, and overlap disabled: SGLang returned one valid token sequence
after 1325.7 seconds without a scheduler watchdog or device error. The wrapper initially
reported exit 1 only because the diagnostic required that this single random Base-model
sample earn a positive GSM8K reward. That was not a scheduler failure; reward quality is
not statistically testable with one sample. Positive reward is now an opt-in diagnostic
gate (`--require-positive-reward`), and response lengths are printed for subsequent tests.
The next gate also passed on HPU 7 with concurrency 8: eight sequences completed in 93.3
seconds, response lengths ranged from 5 to 344 tokens, and mean GSM8K reward was 0.5. The
large time difference from the concurrency-1 test is therefore not an apples-to-apples
throughput comparison: generation lengths differed and the persistent recipe cache was warm.
The first 1+1 attempt then failed before weight synchronization because the server adapter
unconditionally injected Miles' graph-warmup seeds `1,16,32,64` while the staged test set
`cuda_graph_max_bs_decode=8`. `HPUDecodeGraphRunner` correctly rejected warmup sizes above
its maximum. The adapter now filters those seeds using the configured graph maximum; the
runner itself expands them to all reachable backend buckets, so max 8 warms `1,2,4,8` and
the production max 64 retains the original Miles warmup set.
Two immediate retries reproduced the same exception because they used a stale September 6
`/tmp/verl-local-*/verl_compat` mirror rather than the patched scratch source. The startup
message lacked the new `decode_graph_max=..., warmup_seeds=...` fields, and direct comparison
confirmed that the local file still contained the hard-coded warmup list. The GRPO preflight
now compares this critical adapter whenever a node-local mirror is selected and exits before
model loading with an explicit `sync_local.sh` instruction if the copies differ.
The first attempted refresh still transferred zero files: a prior `eval` had exported
`VERL_COMPAT`, `SGLANG_HPU_ROOT`, and `VENV_DIR` as `/tmp` paths, and `gaudi_env.sh`
correctly preserved those overrides, causing `sync_local.sh` to use each destination as its
own source. The sync script now derives immutable canonical sources from its repository
location and rejects any source/destination identity before invoking rsync.

## Commands

Run commands separately; proceed only after the preceding stage succeeds. All tests
print live output and an `exit=... elapsed=... log=...` footer. Run from the allocated
Gaudi node. Generation needs one free HPU; GRPO needs all eight free HPUs.

```bash
cd /scratch/$USER/verl-gaudi-support
unset VENV_DIR SGLANG_HPU_ROOT
bash env/setup_uv_env.sh
bash env/run_grpo_miles.sh preflight
```

Preflight does not start Ray or load the 4B weights. It checks the source/patch, Habana
Torch provenance, real VERL/SGLang imports, ServerArgs, CPU weight serialization, actual
launcher Hydra composition, tiny Qwen3 CPU forward/backward, and all 19 of Miles' HPU graph
regressions (eligibility, blockwise warmup shapes, backend graph safety, paged-v2 static
inputs, per-layer mark_step hooks). It requires the already-cached Qwen3 model configuration/tokenizer.

```bash
# Eight batches of 64 generations on physical HPU 0; choose a different free card if needed.
bash env/run_grpo_miles.sh generation --hpu 0

# Push the identical 4B checkpoint through VERL's HTTP tensor path and require
# token-for-token identical greedy outputs before and after the update.
bash env/run_grpo_miles.sh weights --hpu 0 --prompts 8 --max-response 256

# Existing masked-attention numerical regression on one free HPU.
bash env/shell.sh env VERL_HPU_FUSED_SDPA=1 PT_HPU_LAZY_MODE=0 python env/verify_sdpa_fix.py

# One FSDP2 optimizer/training step, then 30 steps to exceed the old step-22 failure.
# DEBUG prints one elapsed-time line per HCCL weight bucket.
VERL_HPU_WEIGHT_SYNC_DEBUG=1 bash env/run_grpo_miles.sh one \
  actor_rollout_ref.actor.strategy=fsdp2
VERL_HPU_WEIGHT_SYNC_DEBUG=1 bash env/run_grpo_miles.sh soak \
  actor_rollout_ref.actor.strategy=fsdp2

# Full epoch, no validation. WANDB=0 is the default; opt in if credentials are configured.
WANDB=1 bash env/run_grpo_miles.sh full actor_rollout_ref.actor.strategy=fsdp2
```

Keep the same settings across the three GRPO stages. After they pass, a longer response
can be requested with `MAX_RESPONSE_LENGTH=2048` before each command. It increases memory,
compilation work, and generation time; first-step ETA is not a reliable full-run estimate.

Check that `actor/grad_norm` and losses remain finite and that there is reward variation
and some nonzero advantages/updates. `pg_loss=0` alone is not a numerical failure: GRPO
can have zero loss from normalized advantages or an uninformative reward batch. An entire
run with zero rewards/advantages/gradients is not evidence of successful learning.

To share only errors and metrics (works without `rg`):

```bash
grep -nE 'Traceback|Error|ERROR|Exception|tensor_data is empty|grad_norm|pg_loss|reward/mean|clip_ratio|PASS|exit=' /path/from/log/footer.log | tail -n 80
```

If the migration fails, preserve that log and stop before the next stage. No process-kill
or recursive-delete command is required by the verification scripts themselves.

## Previous runtime reference

The previous environment remains on disk, but the current launcher now uses newer
SGLang-only arguments. Pointing it at `sglang-fork` is not a supported rollback by itself;
the old launcher/adapter must also be restored together from a saved revision.
