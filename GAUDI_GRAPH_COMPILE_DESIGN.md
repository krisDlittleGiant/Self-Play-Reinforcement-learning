# Design Doc — Why sglang HPU graph compilation keeps failing, and the plan to fix it

Status: **active investigation**. Written after F21's fix was refuted by the 09:08 run.
Companion docs: `GAUDI_FAILURE_LOG.md` (chronology), `GRPO_GSM8K_GAUDI_PLAN.md` (config).

---

## 1. Current state in one paragraph

Ray + FSDP training works. All four `WorkerDict` actors load Qwen3-0.6B, wrap it in FSDP,
build optimizers, and sit idle with zero errors. The blocker is entirely in the rollout
half: all four sglang schedulers die during `HPUGraphRunner.capture()` — specifically in
`capture_prefill` — with `synStatus 26 [Generic failure]`. The training loop (`fit()`) has
**never executed a single line** on this hardware.

---

## 2. The graph compilation workflow, end to end

### 2.1 Why graphs exist at all

Gaudi is not a latency-optimised kernel dispatcher. Its TPCs want large fused regions of
work uploaded ahead of time. SynapseAI therefore compiles an op DAG into a **recipe** — a
fused, shape-specialised binary. Recipes live in `PT_HPU_RECIPE_CACHE_CONFIG`
(`/scratch/$USER/verl-cache/habana_recipe`, currently 46 entries), keyed by a hash of
*graph structure + exact tensor shapes*.

Shape specialisation is the crux: a recipe compiled for `[2,16,512,512]` cannot run
`[2,16,640,640]`. Different hash, fresh compile, seconds each.

### 2.2 Bucketing

RL rollouts produce a different sequence length every step, so uncontrolled shapes mean
endless recompilation. sglang's HPU path rounds every shape to a **bucket**
(`sglang/srt/hpu_utils.py:277`):

```python
def get_prefill_seq_len_bucket(sum_seq_len):
    return find_bucket(sum_seq_len, (PREFILL_BUCKET_MIN, PREFILL_BUCKET_STEP, PREFILL_BUCKET_MAX))
```

Note `sum_seq_len` — the prefill bucket is a **whole-batch token budget**, not a
per-sequence cap. Misreading this caused F14.

### 2.3 Lazy vs eager, and where the graph is built

| mode | behaviour |
|---|---|
| lazy (`PT_HPU_LAZY_MODE=1`) | ops accumulate into a graph; `mark_step()` compiles + executes it |
| eager (`=0`) | each op dispatches immediately; graphs are single-op fragments |

`HPUGraphRunner.__init__` (`hpu_graph_runner.py:425-441`):

```python
self.is_lazy = 1 if htorch.utils.internal.is_lazy() else 0
if self.is_lazy:
    modify_model_layers(self.model_runner.model, <decoder layer suffixes>,
                        int(os.getenv("SGLANG_CONFIG_HIDDEN_LAYERS", "1")))   # (A)
    self.model = htorch.hpu.wrap_in_hpu_graph(                                 # (B)
        HPUAdapter(self.model_runner.model, self.model_runner.dtype),
        disable_tensor_cache=True,
    )
```

- **(A)** installs a `forward_hook` on every Nth decoder layer that calls
  `htorch.core.mark_step()` (`hpu_graph_runner.py:678-699`).
- **(B)** wraps the *whole* model so its forward is captured/replayed as one HPU graph.

### 2.4 Warmup

`capture()` (`hpu_graph_runner.py:528-556`) runs two independent loops:

```python
for prompt_len in prefill_seq_len_buckets:          # loop 1 — PREFILL
    for prefix_len in prefill_prefix_len_buckets:
        self.capture_prefill(prefix_len, prompt_len)

if self.model_runner.is_generation:                 # loop 2 — DECODE
    for batch_size, seq_len in all_buckets:
        self.capture_decode(batch_size, seq_len)
```

`capture_prefill` builds a synthetic batch via `create_hpu_dummy_batch_prefill(...)` and
calls `self.model.forward(...)` three times, bracketed by `torch.hpu.synchronize()`. Since
`self.model` is the **wrapped** model from (B), each of those calls is a graph capture.

**We die in loop 1, on the first bucket. Loop 2 has never run.**

---

## 3. Why it keeps failing — three distinct layers, found in order

| layer | what was wrong | evidence | status |
|---|---|---|---|
| L1 | `PT_HPU_LAZY_MODE=0` leaked from the training env into sglang, so `is_lazy` was false: no `wrap_in_hpu_graph`, yet `capture()` ran anyway and compiled attention graphs with no graph context | 1,200,938 dumps named `-eager-`, **0** named `-lazy-`; sdpa dumps marked `PostGraphFailed`; failing recipe `hpu::sdpa_recomp_fwd_40_*` | **FIXED** (F20) |
| L2 | `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1` suspected of re-specialising shapes under a captured graph | bridge banner now prints `= 0`, and it **still fails** | **REFUTED** (F21) |
| L3 | `mark_step()` from the per-layer hook (A) fires **inside** the graph capture from (B) | traceback terminates exactly at `forward_hook -> htorch.core.mark_step() -> htcore._mark_step()` | **CURRENT HYPOTHESIS** (F22) |

L3 in detail. `mark_step()` means "close the accumulated graph, compile it, run it." Issuing
that *inside* `wrapped_hpugraph_forward`, which is itself trying to record one contiguous
graph, is a contradiction: a graph boundary in the middle of a graph capture. The bridge
already warns that it inspects ops inside HPU graphs:

```
Warning: The following operations used in HPU graphs might result in accuracy issues :
index_select. (function WarnIfOpsIncompatibleWithHPUGraphs)
```

The recipe name is consistent: `HabanaFusedOpLazy_1_2` is a *lazy fused* graph — the
artifact `mark_step` produces — not an attention kernel.

---

## 4. The missing component (evidence from the miles repo)

Do the miles patches help? **Not by applying them** — they target sglang commit `cb05a44f`
(modern main, `model_executor/runner/...`, `arg_groups/`), ours is 0.4.9 with HPU vendored;
`patch --dry-run` gives 4x "can't find file" and 17 failed hunks (F01). **But their design
is the answer**, and two findings are decisive.

### 4.1 miles graph-captures DECODE ONLY

In `sglang-hpu-gaudi-full.patch`, HPU is added to the **decode** capture path:

```diff
-        if model_runner.device in ("cuda", "musa", "cpu", "npu", "xpu"):
+        if model_runner.device in ("cuda", "musa", "cpu", "npu", "xpu", "hpu"):
             decode = capture_decode_graph(model_runner=model_runner, eager_runner=eager_runner)
```

`capture_prefill_graph` is **not** given an HPU branch. Prefill runs through an
`eager_runner`. Only decode gets `wrap_in_hpu_graph`, into a *separate* `_graph_model`.

This is also the right call on performance grounds, and miles measured it (commit
`00cdae7`): decode is thousands of steps per request and is where the fixed per-step cost
dominates; prefill happens once per request. Graphing prefill buys little and costs the
combinatorial bucket matrix (`prompt_len x prefix_len`) we are currently failing inside.

### 4.2 miles hit L1 too, and added the guard our fork lacks

```python
if os.environ.get("SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH") != "1":
    raise RuntimeError("HPU decode graph capture requires SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH=1.")
if not htorch.utils.internal.is_lazy():
    raise RuntimeError("HPU decode graphs require PT_HPU_LAZY_MODE=1 and PT_HPU_AUTOLOAD=1")
```

Independent confirmation of F20. Our fork silently accepts eager mode and proceeds into
capture, which is exactly why L1 cost a day.

### 4.3 So the missing component is

> **Our fork captures graphs for prefill *and* decode, with layer-level `mark_step` hooks
> installed on the same module object it wraps. The known-good design captures decode only,
> on a separate wrapped model, and runs prefill eagerly.**

---

## 5. Candidate fixes, ranked

| # | change | cost | risk | why it might work | why it might not |
|---|--------|------|------|-------------------|------------------|
| **C1** | `SGLANG_CONFIG_HIDDEN_LAYERS=100` — larger than the 28 decoder layers, so `counter[0] % n == 0` never holds and **no hooks are installed** | 1 env var, zero code | very low | removes the `mark_step` calls that L3 blames, without touching capture | if L3 is wrong, nothing changes; lazy perf may drop without inter-layer boundaries |
| **C2** | skip the prefill capture loop (env-gated), keep decode capture — mirrors miles | ~5 lines in `capture()` | low | eliminates the failing loop entirely; matches proven design | `self.model` is still the wrapped model, so the first real prefill request may hit the same compile at runtime — deferring, not fixing |
| **C3** | C1 + C2 together | small | low | covers both mechanisms | harder to attribute which mattered |
| **C4** | port miles' architecture: separate `_graph_model` for decode, eager path for prefill | large (their patch is 4263 lines against a different tree) | high | this is the actually-proven design | days of work; not a today fix |
| **C5** | `SGLANG_HPU_SKIP_WARMUP=true` | 1 env var | medium | sidesteps capture completely | hides failure; compiles per-shape at request time; caused the F14 hang (though that had a separate cause, now fixed) |

---

## 6. Experiment plan — one variable at a time

Each step has an explicit pass/fail signal. Do not stack changes.

**E1 — test L3 directly (C1).**
```bash
SGLANG_CONFIG_HIDDEN_LAYERS=100 <normal launch>
```
- PASS: `Capture prefill time: N seconds` appears, capture proceeds to decode.
- FAIL: identical `HabanaFusedOpLazy_*` error -> L3 refuted, go to E2.
- This is the highest-information / lowest-cost experiment available. Run it first.

**E2 — skip prefill capture (C2).** Gate loop 1 of `capture()` behind
`SGLANG_HPU_CAPTURE_PREFILL=0`.
- PASS: `Capture decode with batch_size:...` lines appear. Decode capture is now the test.
- PARTIAL: warmup completes, then the first real request stalls or crashes -> confirms the
  deferred-failure concern; prefill genuinely cannot run through the wrapped model, and
  only C4 (separate `_graph_model`) fixes it properly.

**E3 — if decode capture also fails**, adopt miles' decode env block verbatim:
`SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH=1`, `SGLANG_HPU_BUCKETING_STRATEGY=pad`,
`SGLANG_HPU_BUCKET_PAD_PERCENT=25`, `SGLANG_HPU_DECODE_BATCH_BUCKETS=1,2,4,8,16,32,64`,
`SGLANG_HPU_GRAPH_WARMUP_COMPILE_ONLY=1`, `PT_HPU_AUTOLOAD=1`.
Known-good values rather than guesses.

**E4 — last resort**: `SGLANG_HPU_SKIP_WARMUP=true`, accepting slow first steps, purely to
get Phase B exercised and de-risk the ~11 untested stages of the training loop.

---

## 7. What we still do not know

1. **Whether decode capture works.** It has never been reached. Every conclusion so far is
   about prefill.
2. **Whether prefill can run at all through the wrapped model.** If not, C2 defers rather
   than fixes, and C4 becomes mandatory.
3. **Why `synStatus 26` specifically.** It is the compiler's catch-all: a C++ exception with
   no mapped error code, no shape, no op, no dimension. Only the recipe name carries signal.
   This opacity is why hypotheses must be tested one at a time against observable state
   changes (recipe name, stack shape) rather than reasoned about analytically.
4. **Whether Qwen3 matters.** miles' proven runs are Qwen2.5-3B and Llama-3.2-3B; their
   Qwen3-0.6B script exists but we have no evidence it was run to completion on this stack.

---

## 8. Decision record

- **Do not** apply the miles patches directly. Wrong sglang generation (F01). Mine them for
  design, which is what §4 did.
- **Do not** keep swapping attention implementations. F17-F19 were three variants of that
  and all were wrong; attention is where the error *surfaces*, not what is broken.
- **Do** keep the L1 fix (per-process lazy mode) regardless of what happens next — it is
  independently correct and confirmed by miles' own guard.
- **Do** prefer proven configuration over derived configuration wherever miles has one.
