# GRPO on Gaudi — Pipeline Workflow & Failure Log

Living document. **Append a new `F##` entry every time a run fails**, even if the fix is
one line. The value of this file is the chronology: several failures below looked like new
bugs but were the same defect wearing a different costume, and only the sequence makes that
visible.

- Canonical runbook: `GRPO_GSM8K_QWEN3_GAUDI_RUNBOOK.md` (verified commands and triage)
- Companion doc: `GRPO_GSM8K_GAUDI_PLAN.md` (design decisions, config rationale)
- Hardware: 1 node, 8x Intel Gaudi HL-225, SynapseAI 1.22.2, Apptainer container
- Stack: verl 0.9.0.dev (`verl_compat/`) + sglang 0.4.9 HPU fork (`/scratch/$USER/sglang-fork`)
- Job history began with Qwen3-0.6B; current verified gate uses Qwen3-4B-Base

---

## 1. Pipeline workflow

### Phase A — startup (`trainer.init_workers()`), runs once

| # | stage | code | notes |
|---|-------|------|-------|
| 1 | host script re-execs into Apptainer, starts Ray inside that session | `env/run_grpo_gsm8k.sh` | Ray must live in the container session; see F15 |
| 2 | Hydra composes 51 overrides | `recipe/sppo/main_sppo.py:35` | `ray_kwargs.ray_init.address` needs a leading `+`, see F07 |
| 3 | `TaskRunner` actor: read parquet, build dataloader | `main_sppo.py:188` | 7473 train rows / 16 per step = **467 steps** |
| 4 | 4x `WorkerDict` on HPU 0-3: load Qwen3-0.6B (596.05M), FSDP wrap, optimizer | `verl/workers/fsdp_workers.py` | **eager** execution mode required |
| 5 | `LLMServerManager.create()` | `verl/trainer/ppo/ray_trainer.py:939` | |
| 6 | 4x `SGLangHttpServer` actors on HPU 4-7 | `verl/workers/rollout/sglang_rollout/async_sglang_server.py:958` | placement is manual, see F10 |
| 7 | each spawns a scheduler **subprocess**, loads weights, allocates ~48.7 GB KV | `sglang/srt/managers/scheduler.py:2797` | `mp.set_start_method("spawn")` — fresh interpreter, re-reads `os.environ` |
| 8 | `HPUGraphRunner` warmup: compile one graph per prefill/decode bucket | `sglang/srt/model_executor/hpu_graph_runner.py:474` | **lazy** execution mode required |
| 9 | servers ready, `init_workers()` returns | | |

### Phase B — training loop (`fit()`), per step

| # | stage | line in `recipe/sppo/sppo_ray_trainer.py` | what happens |
|---|-------|------|--------------|
| 1 | `_get_gen_batch(batch)` | 225 | 16 prompts, tensor keys stripped |
| 2 | `.repeat(rollout.n)` | 229 | 16 -> **128 sequences** |
| 3 | `generate_sequences` | 241 | async rollout across 4 sglang servers, 32 seq each, <=2048 new tokens |
| 4 | `batch.union(gen_batch_output)` | 274 | splice responses back |
| 5 | `compute_reward` | 297 | GSM8K answer match -> scalar per sequence |
| 6 | `compute_log_prob` | 301 | actor re-scores its own output (FSDP forward) |
| 7 | `compute_ref_log_prob` | 319 | **skipped** — `use_kl_loss=False` |
| 8 | `compute_advantage` | 351 | GRPO: normalize reward within each group of 8 |
| 9 | `update_actor` | 373 | 16 grad-accum micro-steps of 2 seq -> 1 optimizer step |
| 10 | metrics -> W&B | 438 | see F13 |
| 11 | `global_steps += 1` | 446 | see F13 |

GRPO core is 5->9: no critic, no value net. Advantage = "how much better than the other 7
samples of the same question".

### Batch geometry (current config)

```
16 prompts/step x n=8            = 128 sequences/step
mini-batch 16 prompts            -> 32 seq/rank, 1 optimizer step per rollout
micro-batch 2 seq/card           -> 16 grad-accum micro-steps
concurrency                      ~32 sequences per sglang server
467 steps x 1 epoch; save every 25; validate every 25 (1319 x n=2 = 2638 generations)
```

---

## 2. What a correct run looks like

Startup, roughly 8-12 min:

```
=== starting a new Ray cluster ===              <- NOT "reusing", see F15
=== topology: 4 training + 4 rollout = 8 HPUs (TP=1) ===
(WorkerDict) Qwen3ForCausalLM contains 596.05M parameters
(WorkerDict) Total steps: 467, num_warmup_steps: 15
(SGLangHttpServer) VERL HPU: sglang scheduler processes will spawn with PT_HPU_LAZY_MODE=1
(SGLangHttpServer) KV Cache is allocated...
   ... several silent minutes of graph capture, NO traceback ...
(SGLangHttpServer) The server is fired up and ready to roll!    <- never yet observed
step:1 - actor/pg_loss:... - critic/rewards/mean:...
  0%|   | 1/467 [..]
```

`hl-smi` should alternate: cards 4-7 busy during generation, cards 0-3 busy during
`update_actor`. Never both at once.

Step-1 triage:

| metric | good | bad -> action |
|--------|------|---------------|
| `actor/grad_norm` | finite, ~0.1-10 | NaN -> `VERL_HPU_FUSED_SDPA=0` |
| `critic/rewards/mean` | > 0 | exactly 0 -> reward parsing broken |
| `response_length/clip_ratio` | < 0.5 | ~1.0 -> 2048 too short for thinking mode |
| `timing_s/gen` vs `timing_s/update_actor` | gen dominates | — |

---

## 3. The central invariant (learned the hard way)

**Training and inference need OPPOSITE Habana execution modes, in the same job.**

| process group | `PT_HPU_LAZY_MODE` | why |
|---|---|---|
| Ray `WorkerDict` (FSDP actor) | `0` eager | lazy mode breaks FSDP flat-param sharding |
| sglang scheduler (rollout) | `1` lazy | `wrap_in_hpu_graph` is only called when `is_lazy` |

verl has no per-process env channel, so a single global value leaks into both. This one
mismatch caused F17-F20. Enforced in `async_sglang_server.py:183`, which sets lazy mode in
the `SGLangHttpServer` actor before it spawns schedulers.

Reference implementation: the miles repo splits these explicitly —
`extra_env_vars={"PT_HPU_LAZY_MODE": "0"}` for Ray vs
`--sglang-env-vars '{"PT_HPU_LAZY_MODE":"1", ...}'` for rollout
(`miles/scripts/run_llama3p2_3b_inst_fsdp_gaudi.py:357` and `:465`).

---

## 4. Failure log

Status: **FIXED** (verified by a later run getting further) / **UNVERIFIED** (fix applied,
not yet exercised) / **REVERTED** (wrong theory, backed out).

### Environment build

| id | symptom | root cause | fix | status |
|----|---------|-----------|-----|--------|
| F01 | miles sglang patches would not apply — 4x "can't find file", 17x failed hunks | miles targets sglang commit `cb05a44f` (modern main); ours is 0.4.9 with HPU vendored | do not apply; verified with `patch -p1 --dry-run` | FIXED |
| F02 | `ModuleNotFoundError: decord` | missing dep | `uv pip install decord` (`env/setup_uv_env.sh` §6b) | FIXED |
| F03 | `ImportError: cannot import name 'default_cache_dir' from 'triton.runtime.cache'` | triton 3.8.0 pulled transitively; API removed | pin `--no-deps triton==3.3.1` | FIXED |
| F04 | `ImportError: cannot import name 'find_bucket' from 'vllm_hpu_extension.bucketing.linear'` | deleted upstream in `ad558281`; PyPI 0.1 is a 12KB stub | pin `vllm-hpu-extension @ git+...@891db1d2` | FIXED |
| F05 | `TypeError: '>=' not supported between NoneType and int` at `fp8_utils.py:82` | GPU Migration spoofs `torch.version.cuda="11.8"`, so sglang's `is_cuda()` is truthy on Gaudi | patch `is_cuda()` to return False when `is_hpu()` (`env/patches/sglang-0.4.9-is_cuda-hpu.patch`) | FIXED |
| F06 | `ModuleNotFoundError: cachetools` | missing dep | installed | FIXED |

### Launch / wiring

| id | symptom | root cause | fix | status |
|----|---------|-----------|-----|--------|
| F07 | `Could not override 'ray_kwargs.ray_init.address'` | key absent from the config schema; Hydra needs an append | `+ray_kwargs.ray_init.address=...`; then validated all 51 overrides compose offline | FIXED |
| F08 | `AttributeError: 'SPPOActorRolloutRefWorker' object has no attribute '_qat_enabled'` | SPPO's `init_model()` is a stale copy of an older `ActorRolloutRefWorker.init_model()`, predating QAT support; `_build_model_optimizer` reads the flag unconditionally | add `self._init_qat_config()` in `recipe/sppo/sppo_worker.py:68`; AST-diffed against the parent — this was the only gap | FIXED |
| F09 | `ModuleNotFoundError: No module named 'torchao'` in all 4 schedulers | `from torchao.quantization import ...` sat ABOVE the `if torchao_config == "": return model` early-out | move the early-out above the import in `sglang/srt/layers/torchao_utils.py` (first attempt orphaned the `elif` and broke the parse) | FIXED |
| F10 | sglang servers landed on the training cards (`replica_rank=0 -> '1'`, `2 -> '0'`) | three defects: under-populated `train_cards`, every replica taking `free_cards[0]`, wrong `base_gpu_id` | rewrite the HPU placement block in `async_sglang_server.py`; verified by simulation -> cards 4,5,6,7. Note `hl-smi` AIP index != Habana module ID; bus IDs disambiguate | FIXED |
| F11 | `AttributeError: 'RayWorkerGroup' object has no attribute 'update_weights'` — after a full startup | SPPO extended `ActorRolloutRefWorker`; server-based rollout needs the `@register`'d `update_weights` that only the async subclass adds | `class SPPOActorRolloutRefWorker(AsyncActorRolloutRefWorker)` | FIXED |
| F12 | `AssertionError` at `protocol.py:741` in `gen_batch = batch.pop(...)` | SPPO hardcoded `["input_ids","attention_mask","position_ids"]` / `["raw_prompt_ids"]`; this dataset has only `dummy_tensor` as a tensor key | replace with `self._get_gen_batch(batch)` | FIXED |
| F13 | W&B received almost no metrics, and one point per epoch | two bugs: (a) SPPO never called `compute_data_metrics` / `compute_throughout_metrics` / `compute_timing_metrics`; (b) the whole metrics + `logger.log` + `global_steps += 1` block sat OUTSIDE the batch loop, so a "step" meant an epoch | import and call all three builders; re-indent lines 403-437 from 12 to 16 (verified against `.orig`) | FIXED |

### Runtime / infrastructure

| id | symptom | root cause | fix | status |
|----|---------|-----------|-----|--------|
| F14 | 30+ min hang, 0% AIP-Util on all 8 cards, log silent, no step completed | **self-inflicted.** `SGLANG_HPU_PREFILL_BUCKET_MAX=512` assumed a per-sequence cap, but `hpu_graph_runner.py:165` calls `get_prefill_seq_len_bucket(sum(prompt_lens))` — a whole-batch **token budget** (fork default 6144). ~32 prompts x ~75 tok = ~2400 -> `find_bucket` returned 2432, past the ceiling, no captured graph, silent hang. `SGLANG_HPU_SKIP_WARMUP=true` removed the pass that would have failed loudly | remove **every** `SGLANG_HPU_*` override; fork defaults stand | FIXED |
| F15 | `Some workers of the worker process(N) have not registered within the timeout` | script printed `=== reusing the Ray cluster already running ===`. The head node belonged to a previous container session; Apptainer tore down its squashfuse on exit, so new workers cannot exec the venv and die instantly | always run the kill block between runs (see §6). `pkill -f 'ray::'` is NOT enough — gcs_server, monitor, dashboard, log_monitor survive it | FIXED |
| F16 | `SGLANG_HPU_DECODE_BATCH_BUCKET_MAX: unbound variable` | F14 deleted the exports but left a reference in an `echo`, under `set -u` | rewrite the line (`run_grpo_gsm8k.sh:193`) | FIXED |

### Phase B — the training loop (reached for the first time at 09:45 on 2026-09-04)

F39 is retained as the original mask-only hypothesis. **F42 supersedes it** with the final
root cause, implementation, and HPU verification.

| id | symptom | root cause | fix | status |
|----|---------|-----------|-----|--------|
| F43 | A numerically healthy Qwen3-4B-Base SDPA run reached step 22, then one rollout server died at `tp_worker_overlap_thread.py:197`, inside the F41 `htorch.core.mark_step()`, with `ValidateSyncInputTensors tensor_data is empty` for a model input | F41's submission-only diagnosis was incomplete. The old SGLang overlap worker retains CUDA-style asynchronous tensor lifetimes across a background forward thread; an HPU lazy graph can therefore reach submission with a model input whose backing tensor data is already unavailable. Moving `mark_step()` merely changed the point where the invalid graph was detected. The Miles run used a substantially newer scheduler implementation with no `tp_worker_overlap_thread.py`, so its `disable_overlap_schedule=False` is not evidence that this old path is safe | default `+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True` in `env/run_grpo_gsm8k.sh`, selecting the synchronous `TpModelWorker`. This preserves FSDP and fused SDPA and changes only rollout scheduling. Re-enable overlap only after upgrading/porting the newer SGLang scheduler | **CONFIG FIX COMPOSED; NEEDS >22-STEP HPU SOAK** |
| F24 | `omegaconf.errors.ConfigAttributeError: Key 'launch_reward_fn_async' is not in struct` at `sppo_ray_trainer.py:294`, immediately after `Training Progress: 0%\| \| 0/467` | more SPPO staleness (cf. F08, F11, F12). `launch_reward_fn_async` was removed from verl's config schema; modern `ray_trainer.py` uses `RewardLoopManager` + `extract_reward()` instead. SPPO's `fit()` is an old copy that still reads the key as an attribute, which is fatal under OmegaConf struct mode | `self.config.reward_model.get("launch_reward_fn_async", False)` at both call sites (lines ~302 and ~340). False selects the synchronous path — correct here, since the GSM8K reward is a cheap rule-based function and a Ray task would only add scheduling latency | **UNVERIFIED** |
| F36 | `FATAL ERROR :: MODULE:PT_LAZY ... ValidateSyncInputTensors tensor_data is empty` inside the sglang scheduler at `logits_processor.py:139 extend_seq_lens_cpu.tolist()`. Kills the server, so `AgentLoopWorker` dies with it. First seen in `_validate()`, then **again in ordinary training generation** at batch 32 | **direct consequence of the F23 eager-prefill change, and my first diagnosis (validation-only) was wrong.** Despite the `_cpu` name, `create_hpu_forward_batch()` puts these tensors ON the HPU (`hpu_graph_runner.py:191-200`, assigned at `:282`), so `.tolist()` forces a device->host sync. Upstream that only ever ran **inside a captured graph**, where they are graph inputs backed by static buffers that always hold data. Running prefill eagerly makes them ordinary pending lazy tensors. Whether the sync succeeds depends on whether some earlier op already flushed the queue -- hence 6 clean steps at batch 16 and immediate death at batch 32. A latent race, not a threshold | `htorch.core.mark_step()` before the pair, plus `isinstance(..., torch.Tensor)` guards, in `logits_processor.py`. `.orig` backup kept | **UNVERIFIED** |
| F42 | `AssertionError: Expects the parameter to already be moved to device cuda:0 but got hpu:0` at `_fsdp_param.py:266 _init_sharded_param`, during `init_workers` with `actor.strategy=fsdp2`. 0 steps | `PlatformHPU.device_name` returns **"cuda" on purpose** (`platform_hpu.py:45-48`) so PyTorch's distributed APIs accept it under the GPU Migration Toolkit. FSDP1 never cross-checks, so it never noticed. **FSDP2 does**: it compares each parameter's device against the mesh's device type, and parameters honestly report `hpu:0` -- migration rewrites the `torch.cuda` API surface, not device identity | `_mesh_device_type()` in `fsdp_workers.py` returns `"hpu"` when `strategy == "fsdp2"` and vendor is intel. **Verified on this stack that `init_device_mesh("hpu", ...)` works** (torch 2.7.1 / SynapseAI 1.22.2), so the C++ type error the platform comment warns about no longer applies. Scoped to fsdp2 so the proven FSDP1 path is byte-for-byte unchanged; `VERL_FSDP_MESH_DEVICE` overrides either way. Note `reshard_after_forward` flows into `fully_shard` via the same config key (`fsdp_workers.py:726`), so the F38 speedup carries over to FSDP2 | **UNVERIFIED on HPU** |
| F41 | `ValidateSyncInputTensors tensor_data is empty` at `tp_worker_overlap_thread.py:183`, `self.future_token_ids_map[ct+1:ct+bs+1] = next_token_ids`. Killed a run at step 7 (F40's site never fired, so that fix held) | sglang's **overlap scheduler** hands out negative token ids as placeholders and later fills them from a shared buffer. The write splices `next_token_ids` -- an **unexecuted node in the accumulated lazy graph**, not data -- into that buffer, and the bridge needs the source tensor's storage to build the strided copy. Diagnostic says `ThreadPool m_tasks size: 0`: the queue is **empty**, not busy, i.e. the graph was never submitted. The Event machinery around this (`copy_done.record/synchronize`, `sync_event`) is present and correct -- **Events order work that has been SUBMITTED**, and on CUDA holding a tensor implies its kernel is already queued; under lazy mode it does not | `htorch.core.mark_step()` immediately after `forward_batch_generation`, before the splice. mark_step is the *submission* primitive that the Event mechanism was missing. **Corrections on record:** (a) I first proposed `--disable-overlap-schedule`, which deletes a legitimate optimisation to dodge a bug we understand -- an avoidance, not a fix; (b) I described this as a cross-thread race, but the read (`:170`) and write (`:183`) are **both inside `forward_thread_func_`** -- same thread, different iterations, so no cross-thread race exists and removing the thread was never the right lever | **UNVERIFIED on HPU** |
| F40 | `ValidateSyncInputTensors tensor_data is empty` at `logits_processor.py` `.tolist()` **returned** after the F36 mark_step fix, killing a full run at step 13 (also seen at step 6 and step 0 in earlier runs) | F36 identified the mechanism correctly but chose the wrong instrument: `mark_step()` flushes the pending graph but does not materialise these particular tensors, so the race survived. The real defect is that `create_hpu_forward_batch()` copied `extend_seq_lens_cpu` / `extend_logprob_start_lens_cpu` **to the device** and padded them there -- which only ever made sense for the captured-graph path, where every ForwardBatch field had to be a static device tensor. `ForwardBatch` declares both `Optional[List[int]]` (`forward_batch_info.py:201-202`) and **every consumer in the tree treats them as Python ints** (`max()` in lora_manager and flashattention, `enumerate()` in mm_utils, `torch.split()` in aiter, the `zip()` in logits_processor). No HPU attention backend reads them | keep them as **host lists**: `list(...) + [0] * pad` in the prefill path, `[1] * max_running_requests` in the decode dummy batch (`hpu_graph_runner.py`). Removes the device->host sync entirely -- there is nothing left to fail. `mark_step()` deleted from `logits_processor.py` (a graph flush on every logits call, now buying nothing); `isinstance` guards kept as cheap defence. Padding verified equivalent to `to_hpu_and_pad_1d(..., pad_value=0)` across empty/partial/full cases | **UNVERIFIED on HPU** |
| F42 | **Final correction to F39:** the mask-only repair was present but distributed SDPA still produced `actor/grad_norm: nan` | Two defects interacted. First, the import-based scheduler guard misclassified FSDP workers because importing the rollout client also imports `sglang.srt.managers.scheduler`; training therefore bypassed Habana FusedSDPA and used the native migration SDPA path, reproduced non-finite at Qwen layer 0. Second, fully masked padding rows genuinely produce NaNs in raw FusedSDPA additive masks. A broad exception fallback added a third failure by changing backend during non-reentrant checkpoint recomputation | replace module-presence detection with explicit `VERL_HPU_SGLANG_PROCESS=1` set only by the SGLang server; add non-mutating Boolean/`-inf`/dtype-min mask repair in `verl/utils/hpu_sdpa.py`; zero empty-row outputs; remove broad fallback; fail fast on non-finite actor tensors. Verified on a four-token kernel test, Qwen3-0.6B at length 3584, distributed 0.6B GRPO, and distributed Qwen3-4B-Base GRPO (`pg_loss=-0.02729`, `grad_norm=8.92047`) | **FIXED — ONE-STEP 4B VERIFIED** |
| F39 | **SUPERSEDED BY F42.** `actor/grad_norm: nan` on every rank whenever `attn_implementation=sdpa` (F29 worked around it by forcing eager) | **Original incomplete theory:** fully-masked query rows, guaranteed by verl's padding scheme. Prompts are LEFT-padded (`torch_functional.py:512`, `left_pad=True`), so a sequence is `[PAD x (512-prompt_len)][prompt][response][PAD]`. Under causal masking a leading-pad query sees only EARLIER positions, all padding -> the entire row is masked. At mean prompt 78 that is ~434 of 2560 rows, **~17%, by construction**. Eager materialises the score matrix and the 0/0 is contained; a flash-style kernel seeds its online softmax at -inf and computes `exp(-inf - -inf)` = NaN. The forward is masked afterwards so the loss looks finite (`training_log_ppl` read exactly 0.0) but the derivative is still NaN -> poisons the FSDP global norm on all ranks | `_verl_repair_empty_rows()` in `verl/__init__.py`: unmask column 0 for any fully-masked row before calling FusedSDPA. In place (mask is a non-differentiable buffer HF shares across all 28 layers, so layer 0 repairs it and the rest are no-ops), branchless (no `.any()` host sync), no clone. CPU-verified: degenerate rows 3->0, **real rows bit-identical**, idempotent | **SUPERSEDED BY F42** |
| F38 | **the real cause of the 0.10% MFU.** `compute_log_prob` ran 45+ minutes with all 8 cards at 0% AIP-Util, ~103% of one CPU core, and **zero new recipes** (so not compiling). py-spy: `_post_forward_reshard -> _free_unsharded_flat_param -> _free_storage -> torch/storage.py:1258 _resize_` | FSDP frees each layer's unsharded flat parameter immediately after that layer's forward (`reshard_after_forward: true`, the stock default). On HPU `storage.resize_(0)` is a **host-side synchronous** call that blocks on the device queue. 28 layers x every micro-batch x forward and backward means the host spends its life in that stall while the accelerator idles. This is not compilation, not attention, not batch size -- it is the FSDP reshard cycle | `reshard_after_forward=False` for actor and ref, exposed as `RESHARD_AFTER_FWD` in `env/run_grpo_gsm8k.sh`. Cost: ~1.2 GB per card to keep 0.6B bf16 params resident, against 98 GB (2.7 GB in use). Also removes the re-all-gather in backward. `forward_prefetch=true` is the next dial if more is needed | **UNVERIFIED** |
| F37 | Grafana showed nvme0n1p1 writes as a staircase and network climbing to 8 GB in 18 min | **the disk chart is not measuring this job.** `/scratch` is **BeeGFS** (network), `nvme0n1p1` is `/` (local XFS). Recipe cache, habana logs and wandb all live on /scratch, Ray's tmpdir is tmpfs -- so the job writes almost nothing to local disk. The staircase is background system daemons. The **network** chart is the job, and it is dominated by BeeGFS traffic. The real finding underneath: Habana emits **3,000-7,800 graph-dump files per minute** (5-20 MB/min), each one a BeeGFS file create against the metadata server, and the recipe cache adds ~34/min -- i.e. compilation never reaches steady state | `GRAPH_VISUALIZATION_DIR` moved to `/dev/shm` (tmpfs). Recipe cache deliberately stays on /scratch: persistence across runs is worth more than the writes cost. Add `rm -rf /dev/shm/graph_dumps_$USER` to the inter-run cleanup | **UNVERIFIED** |
| F32 | **0.10% MFU.** `update_actor` sustains 119 tok/s/card (0.425 TFLOP/s against ~432 TFLOP/s bf16 peak); `old_log_prob` 422 tok/s/card. A 1000x gap -- structural, not a batch-size tweak | **`PT_HPU_ENABLE_LAZY_COLLECTIVES` was never set on the training side.** FSDP is collective-bound: an all-gather of each layer's params on the way in, a reduce-scatter of its grads on the way out, ~28 layers x micro-batches x 2 per step. Without lazy collectives every one is a blocking, host-synchronised HCCL call, and eager mode (`PT_HPU_LAZY_MODE=0`, required for FSDP flat-param sharding) leaves no graph to hide the latency behind. miles sets it on **both** sides -- its training env is exactly `PT_HPU_LAZY_MODE=0` + `PT_HPU_ENABLE_LAZY_COLLECTIVES=1`; we had only ever set it inside the sglang actor (`async_sglang_server.py:185`) | `export PT_HPU_ENABLE_LAZY_COLLECTIVES=1` in `env/gaudi_env.sh`, added to the `env/shell.sh` forward list | **UNVERIFIED** |
| F33 | batch geometry wrong in two ways: `ppo_mini_batch_size` equalled `train_batch_size` (16:16), and the mini:micro ratio did not match the 4 training ranks | (a) 1:1 train:mini means **one optimizer step per rollout** -- 467 generation batches bought 467 weight updates. (b) The natural ratio for 4 data-parallel ranks is **mini:micro == world_size == 4:1**, which holds exactly when `seqs_per_rank == micro_per_gpu`, i.e. grad-accum == 1. Neither affects fwd/bwd volume (same tokens); (a) sets how many optimizer steps that volume is split into, (b) sets how many dispatch rounds each step costs -- and dispatch is the bottleneck at 0.10% MFU | `PPO_MINI_BATCH_SIZE` 16 -> 4 and `PPO_MICRO_BATCH_SIZE_PER_GPU` 2 -> 8 (miles' value). Geometry now: 128 seq/step; mini 4 prompts -> 32 global seqs -> 8 seq/rank; micro 8/card -> **grad-accum 1**, mini:micro = 32:8 = **4:1**; 4 optimizer steps per rollout | **UNVERIFIED** |
| F35 | (self-correction to F30) gradient checkpointing default flipped to `False` on the reasoning that 17.5 GB of 98 was in use | that measurement was taken **with FusedSDPA**, which never materialises a score matrix. Eager attention -- which F29 forces while grad_norm is NaN -- stores `[B, heads, L, L]` per layer for backward: at micro=8, L=2560 that is 1.68 GB x 28 = **47 GB held at once**, on top of 17.5 GB of weights/optimizer and against 94.6 GB already reserved. Would OOM | `GRAD_CKPT` default back to `True`. It is what makes micro=8 affordable: scores are recomputed per layer, 1.68 GB transient + 1.17 GB inputs. The extra compute is near-free at 0.10% MFU, where the bottleneck is dispatch and collective latency, not FLOPs | **UNVERIFIED** |
| F34 | `val-core/*` absent from W&B | **not a bug.** `_validate()` runs only when `global_steps % test_freq == 0` (`sppo_ray_trainer.py:397-400`) with `test_freq=25`, and `val_before_train=False`. At step 6 it has simply never run -- and at 17.5 min/step, step 25 was ~7 hours away | set `VAL_BEFORE_TRAIN=True` for a baseline point at step 0, and/or lower `TEST_FREQ`. Validation is generation-only (~8 min for 2638 sequences at the measured rollout rate), so it is cheap relative to a training step | NOT A BUG |
| F30 | `timing_s/update_actor: 689 s`, `timing_s/old_log_prob: 194 s`, `timing_s/gen: 21.8 s` -> 1049 s/step, 467 steps = ~136 h | rollout is fine (2% of the step); **98% is FSDP fwd/bwd**. Three compounding causes: (a) `ppo_micro_batch_size_per_gpu=2` -> 16 micro-batches/rank at ~12 s each for 2 sequences of a 0.6B model, i.e. per-op dispatch overhead dominating, not compute (`perf/max_memory_allocated_gb=17.5` of 98 -- 5x headroom unused; miles uses `MICRO_BATCH_SIZE=8`); (b) `enable_gradient_checkpointing=True` recomputes the whole forward inside backward for memory we do not need; (c) `use_remove_padding=False` pads every sequence to 512+2048=2560 while the real mean is 78+1004, so ~2.4x wasted token compute and ~5.6x wasted attention area | micro-batch 2 -> 4, new `GRAD_CKPT` knob defaulting to `False` (`env/run_grpo_gsm8k.sh:60-61,84`). (c) left alone: `use_remove_padding` needs the packed-attention path miles wrote (`TRAIN_ATTENTION_BACKEND=hpu_packed_fused_sdpa`) and is a bigger port | **UNVERIFIED** |
| F31 | "W&B not updating" | **not a bug.** `debug-internal.log` shows `filestream: request sent, status: 200 OK` every 15 s, and the metrics dict reaches `wandb.log()` via `tracking.py:183-186`. Two real explanations: the dashboard gets **one point per 17.5-minute step**, so it looks frozen; and each restart mints a new run id -- the live one is **a2zsaqrh** (11:04), not `of53wbap` (10:12) or `pvp7k5x6` (10:28) | none needed. Check the newest run id in the log line `wandb: Syncing run ...`. Metric coverage is at IOHER parity per F25 | NOT A BUG |
| F29 | `WARN: rank {0,1,2,3} grad_norm is not finite: nan` on every step; `actor/grad_norm: nan` | **no training is happening.** `dp_actor.py:419-421` skips the optimizer step and calls `zero_grad()` when grad_norm is non-finite, so 6 steps produced **zero weight updates**. Weights are NOT corrupted -- they are frozen. Corroborating evidence that the training forward is degenerate, not merely unlucky: `rollout_corr/training_log_ppl = 0.0` **exactly** (`rollout_corr_helper.py:945`), i.e. the actor assigns log-prob 0 (probability 1.0) to every generated token, while sglang's own `rollout_ppl` is a healthy 30.2. `chi2_token` is 1.0e10. Prime suspect is the training-side FusedSDPA monkeypatch (`verl/__init__.py`), whose **backward has never been validated on this hardware** -- flagged as a risk since it was introduced | test `VERL_HPU_FUSED_SDPA=0` (eager attention, correct but slower). Note the sglang side is unaffected: it has its own attention path and its rollouts are sane | **UNVERIFIED** |
| F28 | no crash — `update_actor` **hangs**. py-spy: `_engine_run_backward` (`torch/autograd/graph.py:824`) via `dp_actor.py:680`. All 8 cards 0% AIP-Util, but the worker burns ~107% of one core continuously (536 CPU ticks / 5 s). Last recipe written 10:32:14, still running at 10:49:27 -> **17+ minutes on one graph compile, zero recipes emitted** (4339 cached, 0 new in 10 min) | `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1`. It was set globally on the theory that RL's varying sequence lengths would otherwise force a recompile per shape; the measured effect is the opposite. Refinement makes the compiler search the shape space instead of compiling the shape it was handed. miles — known-good Gaudi FSDP training — **never sets this variable anywhere in the repo** (0 occurrences); its training env is only `PT_HPU_LAZY_MODE=0` | default flipped to `0` in `env/gaudi_env.sh:55`. Note this is the *training* side; the sglang side already unsets it (F21) | **UNVERIFIED** |
| F27 | `ConfigAttributeError: Key 'global_batch_info' is not in struct` (`full_key: actor_rollout_ref.actor.global_batch_info`) at `core_algos.py:1361`, reached via `update_actor` -> `dp_actor.py:628` -> `compute_policy_loss_vanilla`. Raised on all 4 ranks | every policy-loss fn splats `**config.global_batch_info` into `agg_loss`. The key is declared on the `ActorConfig` dataclass (`workers/config/actor.py:193`) but absent from the yaml SPPO composes, and OmegaConf struct mode makes that fatal | inject an empty dict before constructing the actor, exactly as `recipe/ioher/ioher_worker.py:91-94` does. **Empty is correct, not a patch-over**: `agg_loss` (`core_algos.py:1138-1146`) defaults `dp_size=1` and the other three to `None`, and the code that populates the dict (`workers/utils/losses.py:65-67`) belongs to the modern TensorDict engine path, which the legacy `DataParallelPPOActor` we run never enters | **UNVERIFIED** |
| F26 | `TypeError: 'NaiveRewardManager' object is not callable` at `reward.py:203`, called from `sppo_ray_trainer.py:306` | **there are TWO classes named `NaiveRewardManager`**: the old callable one (`verl/workers/reward_manager/naive.py:65`, has `__call__`) and the experimental one (`verl/experimental/reward_loop/reward_manager/naive.py:24`, has only `__init__` + `async run_single`). The config default `reward.reward_manager.source: register` (`verl/trainer/config/reward/reward.yaml:18-19`) resolves to the **experimental** one via `resolve_reward_manager_cls`, and it is meant to be driven by `RewardLoopManager`, never called directly. `compute_reward()` masks this: it tries `reward_fn(data, return_dict=True)` inside `try/except TypeError`, so the real error is swallowed and re-raised from the fallback line | replace the whole reward block with the modern API, copied verbatim from `recipe/ioher/ioher_ray_trainer.py:896-900`: `self._compute_reward_colocate(batch)` when `use_rm`, then `extract_reward(batch)`. `RayPPOTrainer.init_workers()` always builds `self.reward_loop_manager` (`ray_trainer.py:904`) and the `RewardLoopWorker` actors are already live in the log. Also fixed the dead REMAX baseline block (`self.rm_wg` no longer exists) to match `ioher:865-868` | **UNVERIFIED** |
| F25 | W&B looked empty of stability metrics | **not a metrics bug.** Verified against `recipe/ioher/ioher_ray_trainer.py:960-996`: IOHER logs exactly `actor_output_metrics` + `val_metrics` + `training/*` + the three builders (`compute_data_metrics`, `compute_timing_metrics`, `compute_throughout_metrics`). SPPO now emits the identical set after F13, so the two recipes are at parity. W&B is empty because **no training step has ever completed**, not because a metric is missing. `actor/grad_norm` arrives inside `actor_output_metrics` from `update_actor` -> `dp_actor.py:405-411` | no change needed. A `compute_variance_proxy_metrics` call was added and then **reverted** to keep strict GRPO/IOHER parity: it needs `sum_pi_squared` in the batch and self-skips without it | PARITY CONFIRMED |

**Milestone: `update_actor` entered (10:28 run).** The F27 fix worked — stage 9 is reached and the backward pass begins. Blocked on compile time (F28), not correctness.

**Milestone: Phase B stages 1-8 completed (10:12 run).** Generation, reward, `compute_log_prob` and `compute_advantage` all ran; the failure was in stage 9, `update_actor`. 7m25s of real training work before the error, and W&B opened a run. Stages proven end-to-end on Gaudi: sglang rollout actually returns tokens, and the FSDP forward pass works.

**Milestone: Phase A completed.** `Training Progress: 0%| | 0/467`, `LLMServerManager` listed 4 live servers, `AgentLoopWorker` / `RewardLoopWorker` spawned, `Training from scratch`. The F23 prefill fix worked — this was the first time `trainer.fit()` ever executed.

### The graph-compilation saga (F17-F21) — all one underlying defect

| id | symptom | root cause | fix | status |
|----|---------|-----------|-----|--------|
| F17 | `Graph compile failed. Recipe: .graph_dumps/hpu::sdpa_recomp_fwd_40_4550, synStatus 26 [Generic failure]` — all 4 schedulers, during `capture_prefill` | first theory: verl's global `F.scaled_dot_product_attention` -> FusedSDPA monkeypatch was leaking into sglang processes | added a runtime gate (`"sglang.srt.managers.scheduler" in sys.modules`) in `verl/__init__.py`. Recipe id changed 4550 -> 4525, so the gate took effect — but **this was not the cause**. sglang calls FusedSDPA directly via `vllm_hpu_extension`, not through `F.sdpa` | kept (correct in principle), did not fix |
| F18 | `RuntimeError: The size of tensor a (1024) must match the size of tensor b (1152) at non-singleton dimension 4` | second theory: switch prefill to `naive_impl`. But `_naive_prompt_attention` ignores `block_list`/`key_cache`, so its score matrix is sized for raw q/k/v while sglang passes a **paged** `attn_bias` | REVERTED to `fsdpa_impl`. Note: `impl` has only 3 options (`naive_impl`, `fsdpa_impl`, `flex_impl`) and `flex_impl` needs inductor codegen, unsupported here | REVERTED |
| F19 | (same as F17) | third theory: `recompute_mode = True` hardcoded in `_fsdpa_prompt_attention` selects the `sdpa_recomp_fwd` kernel named in the error | made it env-gated (`VLLM_HPU_FSDPA_RECOMPUTE`), then **reverted the default to upstream** once the real cause was found, to keep one variable moving at a time | REVERTED (knob retained) |
| F20 | **ROOT CAUSE of F17-F19** | `PT_HPU_LAZY_MODE=0` (global, required by FSDP) leaked into the sglang schedulers. `hpu_graph_runner.py:425` only calls `htorch.hpu.wrap_in_hpu_graph()` when `is_lazy`; under eager it skips the wrap and then runs `capture()` -> `model.forward()` -> `torch.hpu.synchronize()` anyway, compiling attention graphs with no graph context. Forensic proof: 1,200,938 graph dumps named `-eager-`, **zero** named `-lazy-`, and the sdpa dumps marked `PostGraphFailed` | set `PT_HPU_LAZY_MODE=1` + `PT_HPU_ENABLE_LAZY_COLLECTIVES=1` + `PT_HPU_AUTOLOAD=1` in the `SGLangHttpServer` actor before it spawns schedulers (`async_sglang_server.py:183`). Safe despite the actor having already imported habana in eager, because sglang uses `mp.set_start_method("spawn", force=True)` (`entrypoints/engine.py:696`) — each scheduler is a fresh interpreter | FIXED (mechanism confirmed) |
| F23 | same `HabanaFusedOpLazy_1_2 / synStatus 26`, but the traceback moved from `graphs.py:672 orig_fwd -> forward_hook -> mark_step` to **`graphs.py:673 graph.capture_end() -> _hpu_C.capture_end()`**. `SGLANG_CONFIG_HIDDEN_LAYERS=100` DID propagate (Apptainer passes the host env through; we do not use `--containall`) and DID remove the hooks — the forward now completes and the failure is the final whole-graph compile | **prefill cannot be graph-captured at all** on this stack. Not the hooks, not the attention kernel: `wrap_in_hpu_graph` over a whole prefill forward will not compile. On HPU `ForwardMode.is_cuda_graph()` returns True unconditionally (`forward_batch_info.py:119-122`), so prefill reaches the graph runner at runtime too — skipping warmup capture alone would only defer the same failure to the first request | implement miles' design in our fork: keep an **unwrapped** `HPUAdapter` (`self.model_eager`), route extend/prefill batches to it in `_forward()`, and skip the prefill capture loop. Decode still captured. Env: `SGLANG_HPU_GRAPH_PREFILL=1` restores old behaviour | **UNVERIFIED** |
| F22 | `Graph compile failed. Recipe: .graph_dumps/HabanaFusedOpLazy_1_2, synStatus 26` — bridge banner confirms `PT_HPU_LAZY_MODE = 1` and `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0`, so F21's fix was applied and **refuted** | current hypothesis: the per-layer `forward_hook -> htorch.core.mark_step()` installed by `modify_model_layers` (`hpu_graph_runner.py:678-699`) fires INSIDE the capture performed by `wrap_in_hpu_graph` — a graph boundary in the middle of a graph capture. Traceback terminates exactly there. miles avoids this by capturing **decode only**, on a separate `_graph_model`, running prefill through an `eager_runner` | `SGLANG_CONFIG_HIDDEN_LAYERS=100` (> 28 layers) makes `counter[0] % n == 0` never hold, so no hooks are installed | **REFUTED** — hooks confirmed removed (mark_step gone from the stack), failure persists at `capture_end`. See F23 |
| F21 | `Graph compile failed. Recipe: .graph_dumps/HabanaFusedOpLazy_1_6, synStatus 26` | **different, later failure.** Stack now contains `wrapped_hpugraph_forward`, `hpu_lazy_tensors.cpp`, `forward_hook -> htorch.core.mark_step()` — lazy mode and `wrap_in_hpu_graph` are active, i.e. F20 worked. Hypothesis: `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1` (set globally for RL's varying seq lengths) re-specializes shapes underneath a captured graph. sglang already pins every shape to a bucket, and **miles never sets this variable anywhere** | `os.environ.pop("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES")` in the same block | **REFUTED** — applied and confirmed in the bridge banner, still fails (see F22). Change retained: harmless, and shapes are bucket-pinned anyway |
| F43 | **ROOT CAUSE of the whole decode-graph blocker.** Capture aborts with `RuntimeError: cpu fallback is not supported during hpu graph capturing`, or (with `hpu_fused`) hangs in `JoinPendingLaunchThread` with no exception at all | **Four independent breakages, all required to be fixed together.** (1) `SGLANG_EXPERIMENTAL_HPU_PAGED_V2`, `PT_HPU_AUTOLOAD`, `GRAPH_VISUALIZATION_DIR` were missing from `env/shell.sh`'s forward list, so they never crossed into the container. (2) `SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH` was set only by `run_grpo_miles.sh`, never by `run_grpo_gsm8k.sh` — **the decode-graph runner was never constructed**, which is why `SGLANG_VERIFY_HPU_DECODE_GRAPH=1` printed nothing on two separate investigations. (3) the launcher hardcoded `attention_backend=hpu_fused`, whose metadata is built with a dynamic `index_select`; its shape changes per step, `synGraphInferShapes` cannot resolve it, and the failure **deadlocks rather than raises**. (4) in `hpu_paged_v2_backend.py`, `F.relu` is applied to an **int64** tensor; Habana's relu TPC kernel is float-only, so PyTorch silently falls back to CPU — illegal inside `wrap_in_hpu_graph` capture | (1) added the three vars at `env/shell.sh:44`. (2)+(3) `env/run_grpo_gsm8k.sh` now defaults `SGLANG_HPU_ATTENTION_BACKEND=hpu_paged_v2` and `SGLANG_DECODE_GRAPH_BACKEND=full`, and exports the two `SGLANG_EXPERIMENTAL_*` gates plus `PT_HPU_AUTOLOAD` itself, so no `++` override is load-bearing. (4) replaced `F.relu(groups)` with `torch.where(padding_groups, zeros, groups)` — exactly equivalent, since the only negatives are padding markers masked out on the next line. Shipped as a hunk in `env/patches/sglang-cb05a44-verl-hpu-runtime.patch` | **FIXED — verified over 5 GRPO steps** |
| F44 | long-held theory: "captured decode graphs return corrupted generations after an FSDP weight sync" | **REFUTED, twice over.** Weight-sync debug showed sender bytes == receiver-loaded bytes, identical. And per F43(2), no graph had ever been constructed in any of those runs — the corruption being attributed to graphs was `hpu_fused` running graphless. A 5-step run with real captured `hpu_paged_v2` graphs holds `rollout_corr/ppl_ratio` at 0.9998-1.0062 across five weight syncs | none needed. The retracted claim was removed from `env/run_grpo_gsm8k.sh`'s comment block, where it had been steering the defaults | **REFUTED** |
| F45 | "the node crashed" -- twice. Session dies, all processes gone, no logs written | **Neither time was a node crash.** Everything -- the editor, the Claude session and the training run -- shares ONE Slurm memory cgroup: the OnDemand **VSCode tunnel job**. Training exhausts it, Slurm kills the job, the session dies. `sacct`: job `62934641` ReqMem=256G MaxRSS=268429072K (=256GiB exactly) State=**OUT_OF_MEMORY**; job `62958214` ReqMem=400G, `script.sh "Killed"`. Meanwhile gaudi004 was **up 54 days**, never rebooted, 457 GB free, `/dev/shm` at 0%. Raising the tunnel 256G->400G did not help; the run went through that too | run training in its own Slurm job (`env/sbatch_grpo.sh`, `--mem=0` = the node's full 503 GB) so an OOM costs the run and not the session -- or accept the risk and keep a passive RSS sampler writing to `/scratch`, which survives the session dying | **DIAGNOSED** |
| F46 | theory: host RAM scales with batch size, so batch 32 caused the OOM | **REFUTED by measurement.** batch 8 @ 256 = 142 GB; batch 32 @ 256 = **168.6 GB**. 4x the tokens for **+19%** host RAM. Linear-in-batch would have predicted 568 GB. Host RAM here is near-fixed cost (4 FSDP workers + 4 SGLang processes), not per-token | batch is exonerated. The remaining suspects for the >400 GB blowup are **response length** and **decode-graph capture at long context** -- both crashes ran 2048 tokens with `hpu_paged_v2` graphs ON, the survivor ran 256 with graphs OFF | **REFUTED** |
| F47 | rollout load imbalance: `generate_sequences` min 72.4 s, mean 431.5 s, **max 908.9 s** | the step is bounded by the slowest of the 4 SGLang servers, so 913 s is spent where a balanced split would cost ~431 s -- **2.11x wasted**. `SGLANG_MAX_RUNNING_REQUESTS=32` with 256 requests over 4 servers (64 each) forces queuing, and the balancer is not evening it out | not yet fixed. Worth attacking only after decode graphs are back on, since graphs change the per-server service time by ~66x and may change the balance entirely | **OPEN** |

Evidence that F20 was real, not cosmetic:

| | before F20 | after F20 |
|---|---|---|
| failing recipe | `hpu::sdpa_recomp_fwd_40_4525` | `HabanaFusedOpLazy_1_6` |
| `wrap_in_hpu_graph` in stack | absent | `wrapped_hpugraph_forward` present |
| `mark_step` in stack | absent | present via `forward_hook` |
| graph dumps | 1.2M eager, 0 lazy | lazy tensors live |

---

## 4b. The recurring theme

F21, F23, F36 are all the same shape: **this fork's HPU code assumes prefill runs inside a
captured graph.** Once F20/F23 made prefill run eagerly instead -- necessary, because the
capture would not compile -- every place that silently relied on captured-graph semantics
(static buffers, guaranteed-materialised tensors, no mid-graph boundaries) becomes a latent
bug that surfaces one at a time, often intermittently and shape-dependently.

Expect more of these. The structural alternative is miles' design: a separate `_graph_model`
for decode with a genuine `eager_runner` for prefill, rather than reusing one code path for
both. That is the port described in section 5 item 2.

## 5. RESOLVED — decode graphs work (see F43/F44)

The blocker that spans F17-F23 and F36-F42 is closed. The leads formerly listed here are
all superseded: lead 1 (`index_select` in graphs) was correct in spirit but the fix was to
change backend, not to force contiguous PA; lead 2 (miles' bucket block) was unnecessary;
lead 3 (`SGLANG_HPU_SKIP_WARMUP`) and lead 4 (different SynapseAI) are not needed.

### Working configuration

```
SGLANG_HPU_ATTENTION_BACKEND=hpu_paged_v2      # now the script default
SGLANG_DECODE_GRAPH_BACKEND=full               # now the script default
SGLANG_EXPERIMENTAL_HPU_PAGED_V2=1             # exported by the script
SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH=1         # exported by the script
PT_HPU_AUTOLOAD=1                              # exported by the script
VERL_HPU_SGLANG_LAZY=1                         # decode graphs need lazy mode
```

### Measured, Qwen3-4B-Base, batch 8, 256 response tokens, 64 seqs/step

| metric | graphs off (`hpu_fused`) | captured `hpu_paged_v2` |
|---|---|---|
| `rollout_corr/ppl_ratio` | 15,000 - 126,000 | **0.9998 - 1.0062** |
| `timing_s/gen` | 718 - 953 s | **29.2 s capture step, ~12.7 s steady** |
| `perf/throughput` | 29 - 31 | **114 - 315** |
| decode | 9 tok/s/card | **~182 tok/s/card** |
| `actor/grad_norm` | 0.0 / non-finite | 23.8 -> 8.2 -> 5.7 -> 2.6 -> 3.2 |
| `critic/score/mean` | 0.0 | 0.19 - 0.58 |

Step 1 carries the capture cost once and it does not recur. **The bottleneck has moved to
`update_actor` (~17 s vs ~12.9 s for generation)** — generation is no longer the long pole.

### Still open

- **`hpu_fused` remains broken** under graph capture. It was routed around, not fixed. Its
  dynamic `index_select` shapes still fail `synGraphInferShapes`, and still fail by
  *deadlocking* — if a future run hangs silently in `JoinPendingLaunchThread`, check the
  backend first.
- **`_validate()` is untested** (`TEST_FREQ=-1`). It goes through the F36-era `.tolist()`
  path on `logits_processor.py`, which has its own history of `tensor_data is empty`.
- Scaling to 2048 / 4096 response length is unproven; 256 is what has been measured.
---

## 6. Operational notes

**Always run this between runs.** A stale Ray head is silently reused and fails as F15:

```bash
pkill -9 -u $(id -u) -f '/venvs/verl-gaudi/'   # matches every process in the venv
sleep 3
rm -rf /dev/shm/ray_svijay46 /dev/shm/graph_dumps_$(id -un)
pgrep -u $(id -u) -af 'ray|sglang|sppo'        # must print nothing
```

**Current launch command:** use the one-step gate and full-run blocks in
`GRPO_GSM8K_QWEN3_GAUDI_RUNBOOK.md`. The older Qwen3-0.6B thinking-mode command formerly
shown here is historical and produced zero-reward no-op runs.

**Debugging a hang.** `ray stack` needs sudo (unavailable). Use py-spy **from the host**,
not inside the container (`ptrace_scope=0` allows it; inside gives "Permission Denied"):

```bash
/scratch/$USER/venvs/verl-gaudi/bin/py-spy dump --pid <TaskRunner pid>
```

`ps aux` %CPU is **cumulative** (cpu-time / elapsed), not instantaneous — it will show a
finished thread at 98% forever. Use `ps -L` to see live per-thread state.

**Graph dumps grow without bound**: `/scratch/$USER/verl-cache/graph_dumps/.graph_dumps`
reached 1.2M files / 4 GB on BeeGFS. Safe to delete; it is debug output.

---

## 7. Patched files (all have `.orig` backups)

| file | change | entry |
|------|--------|-------|
| `verl_compat/recipe/sppo/sppo_worker.py` | `AsyncActorRolloutRefWorker` base; `_init_qat_config()` | F08, F11 |
| `verl_compat/recipe/sppo/sppo_ray_trainer.py` | `_get_gen_batch`; metric builders; loop re-indent | F12, F13 |
| `verl_compat/verl/workers/rollout/sglang_rollout/async_sglang_server.py` | HPU card placement; per-process lazy-mode env; explicit SGLang-process marker | F10, F20, F21, F42 |
| `verl_compat/verl/__init__.py` | training-only FusedSDPA routing through an explicit process marker | F17, F42 |
| `verl_compat/verl/utils/hpu_sdpa.py` | non-mutating empty-row mask repair; stable FusedSDPA/checkpoint adapter | F42 |
| `verl_compat/verl/workers/actor/dp_actor.py` | fail-fast checks for non-finite HPU actor tensors and gradient norm | F42 |
| `sglang-fork/python/sglang/srt/utils.py` | `is_cuda()` returns False on HPU | F05 |
| `sglang-fork/python/sglang/srt/layers/torchao_utils.py` | early-out above lazy import | F09 |
| `sglang-fork/python/sglang/srt/layers/attention/hpu_attn_backend.py` | `impl` env-switchable (default `fsdpa_impl`) | F18 |
| `sglang-fork/python/sglang/srt/layers/attention/hpu_paged_v2_backend.py` | `F.relu` -> `torch.where` on the int64 block groups (kills the CPU fallback that aborted capture) | F43 |
| `env/run_grpo_gsm8k.sh` | `hpu_paged_v2` + decode graphs are defaults; exports the `SGLANG_EXPERIMENTAL_*` / `PT_HPU_AUTOLOAD` gates | F43 |
| `env/shell.sh` | forward list extended so the paged-v2 / decode-graph / autoload vars reach the container | F43 |
| `venvs/verl-gaudi/.../vllm_hpu_extension/ops.py` | `recompute_mode` env-gated (default upstream) | F19 |

**`ops.py` lives in site-packages** — re-running `env/setup_uv_env.sh` will overwrite it.
If that patch ever becomes load-bearing, move it into `env/patches/`.

---

## 8. Template for new entries

```
| F## | <symptom, verbatim error line> | <root cause, with file:line> | <fix, with file:line> | FIXED / UNVERIFIED / REVERTED |
```

Record the *wrong* theories too, as F17-F19 do. Knowing what was ruled out, and by what
evidence, is what stops a fix being re-tried three weeks later.
