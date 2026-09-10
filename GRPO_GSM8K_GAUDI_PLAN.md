# Plain GRPO on GSM8K, 4 FSDP + 4 SGLang, Intel Gaudi

Working notes and execution plan for running **plain GRPO** on GSM8K on this box, via a new
`recipe/grpo_hpu/` derived from the **SPPO** recipe (explicitly *not* IOHER), with
`uv`-managed Python, W&B logging, and all caches on `/scratch`.

> This is the original investigation document. For the current verified Qwen3-4B-Base
> configuration and runnable commands, use `GRPO_GSM8K_QWEN3_GAUDI_RUNBOOK.md`.

- Repo: `/scratch/$USER/verl-gaudi-support` @ `368391f` (branch `main`)
- Node: `gaudi005` — 8 × HL-225 (Gaudi2), 98 GB HBM each
- Driver/firmware: **SynapseAI 1.24.0**, `hl-1.24.0-fw-62.6.2.0`
- Written: 2026-09-03

---

## 0. TL;DR

| | |
|---|---|
| **Entry point** | **`recipe/grpo_hpu/`, built from the SPPO skeleton** (§9.5–9.6). There is no `recipe/grpo` — GRPO is `algorithm.adv_estimator=grpo`, not a recipe. SPPO is chosen for its legacy `DataProto` trainer + workers, which bypass the nested-tensor path entirely. |
| **Base script to copy** | `verl_compat/examples/grpo_trainer/run_qwen3_4b_dapo_fsdp.sh`, `hpu` branch |
| **Topology** | `trainer.n_gpus_per_node=4` + `rollout.tensor_model_parallel_size=1` ⇒ 4 FSDP ranks + 4 SGLang replicas = 8 cards |
| **Runtime** | Apptainer, `vault.habana.ai/gaudi-docker/1.22.2/ubuntu24.04/habanalabs/pytorch-installer-2.7.1:1.22.2-32` |
| **Why 1.22.2 and not 1.24.0** | The local Habana SGLang fork pins `torch==2.7.1`; 1.22.x ships exactly that. 1.24.0 ships torch 2.10.0. |
| **uv role** | `uv venv --system-site-packages` on top of the container's Habana torch, then `uv pip install` verl's deps. uv must **not** manage torch. |
| **Caches** | All redirected to `/scratch` — `/home` has 7.6 GB free and `.graph_dumps/` is already being written into the repo. See §6.5. |
| **Biggest risk** | SPPO never calls `checkpoint_manager.update_weights()`, so rollout servers would train against **frozen initial weights, silently**. Must be added. See §9.6-B. (Stock `main_ppo`'s step-4 nested-tensor risk, §2, is what we avoid by not using it.) |
| **Hard blockers found** | Every hardcoded path in the launch scripts is missing or unreadable on this box. See §3. |

---

## 1. What a plain-GRPO run is made of

Six components. Only the first three are contentious on Gaudi.

1. **Trainer path** — `recipe/grpo_hpu/main_grpo.py` → `RayGRPOTrainer(RayPPOTrainer)` →
   `fsdp_workers.ActorRolloutRefWorker` (legacy `DataProto` FSDP path), built from SPPO.
   *Not* `main_ppo` → `engine_workers` — see §2.
2. **Rollout** — SGLang HTTP servers, one Ray actor per replica, on their **own** cards.
3. **Platform layer** — `PlatformHPU` + the import-time patches in `verl/__init__.py`.
4. **Data** — GSM8K in verl's parquet schema (`data_source` / `prompt` / `reward_model`).
5. **Reward** — built-in rule-based scorer, dispatched on `data_source == "openai/gsm8k"`.
6. **Logging** — `verl.utils.tracking.Tracking`, `wandb.init()` inside the TaskRunner actor.

### Topology arithmetic

The rollout is **disaggregated** on Gaudi — cards are acquired exclusively, so the SGLang
servers do not share the trainer's cards (`async_sglang_server.py`, `_IS_HPU_HOST` block):

```
training cards = trainer.n_gpus_per_node                         = 4
rollout cards  = trainer.n_gpus_per_node / tensor_model_parallel = 4 / 1 = 4
TOTAL          = n_gpus_per_node × (1 + 1/TP)                    = 8
```

So `n_gpus_per_node=4`, `TP=1` gives exactly the 4 + 4 split asked for, and consumes the
whole node. Raising `n_gpus_per_node` to 8 does **not** give 8 trainers — it fails at
rollout launch with `synStatus=8 [Device not found]` because the actor group claims all 8
first and the rollout servers find nothing free.

---

## 2. ⚠ Why we are not using `main_ppo`

**Stock `main_ppo` GRPO on Gaudi is not a proven path**, and this is the reason the plan
routes around it via SPPO (§9.5). The commit log is explicit, and it went back and forth:

| Commit | What happened |
|---|---|
| `63c3f1b` | Added `trainer.use_legacy_worker_impl` to route `main_ppo` at the legacy dense-tensor FSDP workers, because the model-engine path uses nested tensors. |
| `97b9172` | **Reverted it.** "It cannot work." `RayPPOTrainer` *itself* calls `batch.to_tensordict()` → `left_right_2_no_padding(...)` → nested, at five sites, before every worker dispatch. Swapping in legacy workers just moves the failure to `'TensorDict' object has no attribute 'meta_info'`. Two protocols, not two workers. Concluded: *"Stock main_ppo therefore cannot run on Gaudi in this verl version."* |
| `090c065` | Attacked the real problem instead. Stock GRPO **"died ~4 steps in, reproducibly"** at `torch.nested.narrow(...)` → `RuntimeError: Graph duplication failed. synStatus=26`. Replaced with dense boolean-mask indexing at all four call sites. |
| `368391f` | Gated that to `vendor == intel` so CUDA keeps the original ops. ← **HEAD** |

Read that sequence carefully, because it cuts both ways:

- ✅ Stock `main_ppo` **does** start and train on Gaudi — it reached step 4. Nested
  `nested_tensor_from_jagged` and `to_padded_tensor` were explicitly left alone *because
  they work* ("the run reached step 4, so they work on Gaudi; only narrow was fatal").
- ⚠️ HEAD is the **first commit** where the step-4 crash should be gone, and **nobody has
  reported a run past it**. The pessimistic sentence in `97b9172` predates the fix and is
  superseded — but it has not been positively replaced by a successful long run.
- ❌ Do **not** pass `trainer.use_legacy_worker_impl=True`. The code still exists in
  `main_ppo_v0.py` but it is known-broken; `97b9172` removed it from the launch script and
  left the code behind.

**We avoid the question entirely.** The SPPO-derived trainer dispatches `DataProto` straight
to legacy workers and never enters `left_right_2_no_padding`, so none of the nested ops above
are on its path. `main_ppo` stays available as a cross-check once a GRPO run is working —
running both would finally answer whether HEAD cleared step 4.

---

## 3. ⚠ Everything hardcoded in the launch scripts is wrong on this box

Verified by direct check. None of these paths resolve for `$USER`:

| Path referenced in repo | Status here | Replace with |
|---|---|---|
| `/workspace/inoculation/verl-gaudi-support/verl_compat` | **missing** | `/scratch/$USER/verl-gaudi-support/verl_compat` |
| `/workspace/inoculation/hf_cache` | **missing** | `/scratch/$USER/hf_cache` (already 40 GB, populated) |
| `/workspace/inoculation/data/dapo_math/*.parquet` | **missing** | GSM8K, regenerated — see §5 |
| `/scratch/sgoli125/sglang-habana/python` | **Permission denied** (another user) | `/scratch/$USER/sglang-fork/python` — see below |
| `/workspace/inoculation/venv` | **missing** | the uv venv from §4 |

### The SGLang substitute is already on disk, and it is the right one

`/scratch/$USER/sglang-fork` is **sglang 0.4.9** with real HPU support
(`hpu_graph_runner.py`, `hpu_communicator.py`, `hpu_utils.py`, HPU-aware `custom_op.py`).
Every optional-import guard the fork wrote lines up with it exactly:

| Symbol | verl's guard expects | This tree |
|---|---|---|
| `sglang.srt.weight_sync.utils` | absent → use the hand-written fallback | **absent** ✅ |
| `ContinueGenerationReqInput` | absent → `resume_generation()` no-ops | absent ✅ |
| `PauseGenerationReqInput` | absent → `abort_all_requests()` no-ops | absent ✅ |
| `ServerStatus` | absent → skip the status assignment | absent ✅ |
| `add_prometheus_middleware` | absent → skip `/metrics` | absent ✅ |
| `UpdateWeightsFromTensorReqInput` | **required** by the fallback | present ✅ |
| `Release/ResumeMemoryOccupationReqInput` | present | present ✅ |

6 for 6. This is the tree the port was written against. It also pins `torch==2.7.1`, which
is what drives the container choice in §4.

Other SGLang trees on disk (`sglang-miles`, `gaudi-repo/sglang-miles`) are `0.0.0.dev0`
with `weight_sync/utils.py` **present** — they belong to the MILES project and would take
the *other* branch of verl's import guard. Do not use them here.

---

## 4. Environment: container + uv

### 4.1 Why not just uv on the bare host

Checked and ruled out:

- The login environment has `torch 2.9.0+cu128` (a **CUDA** build) in `~/.local` and a
  `habana_frameworks` namespace package in the mamba base with **no `.torch` submodule**.
  `import habana_frameworks.torch` fails ⇒ `PlatformHPU.is_available()` is False ⇒ the
  entire HPU path is dead. This is exactly why every script sets `PYTHONNOUSERSITE=1`.
- `https://vault.habana.ai/artifactory/api/pypi/gaudi-python/simple/` is reachable but no
  longer carries `habana-torch-plugin` — the index now holds only
  `hbn-horovod`, `hbn-lightning-plugins`, `hbn-pyhlml`, `hbn-tensorflow`, `pyhlml`.
  **Intel ships the PyTorch bridge in the container images now, not as a standalone wheel.**

So: the Habana torch stack comes from a container; uv manages everything on top of it.

### 4.2 Container choice

Available on disk, both unsuitable:

| Image | Contents | Verdict |
|---|---|---|
| `/scratch/$USER/apptainer/gaudi_pytorch.sif` | SynapseAI **1.16.0**, torch **2.2.2a0**, py3.10 | Too old. verl 0.9.0.dev needs `tensordict>=0.8`, FSDP2, `nn.Module.compile`, nested jagged tensors. |
| `/scratch/$USER/ioher/verl_sglang.sif` | torch **2.9.1+cu129**, sglang 0.5.9, transformers 4.57.1, ray 2.54.0, tensordict 0.10.0, flash-attn 2.8.3 | **CUDA** image, no `habana_frameworks`. This is the A100 reference environment — useful as a version reference, not runnable here. |

Pull instead:

```
vault.habana.ai/gaudi-docker/1.22.2/ubuntu24.04/habanalabs/pytorch-installer-2.7.1:1.22.2-32
```

Reasoning, in priority order:

1. **torch 2.7.1** — the exact pin in `sglang-fork/python/pyproject.toml`. This is the
   binding constraint; get it wrong and the rollout half does not run.
2. SynapseAI 1.22.2 userspace on a 1.24.0 driver is within Habana's backward-compat window
   (N−2). The 1.16 image already proved forward-tolerance by initializing all 8 cards.
3. Version map confirmed against the vault: `1.21.x → 2.6.0`, **`1.22.x → 2.7.1`**,
   `1.23.0 → 2.9.0`, `1.24.0 → 2.10.0`.

> If you would rather match the driver exactly (1.24.0 / torch 2.10.0), you must also
> re-pin or port SGLang. That is a separate project, not a config change.

### 4.3 uv layout

The one rule: **uv installs everything except torch.** The container's Habana torch is not
on PyPI and must not be shadowed.

```bash
# inside the container
uv venv /scratch/$USER/venvs/verl-gaudi \
    --python "$(command -v python3)" \
    --system-site-packages          # ← keeps habana_frameworks + Habana torch visible
source /scratch/$USER/venvs/verl-gaudi/bin/activate

# torch is excluded on purpose; --no-deps on the sglang install for the same reason
uv pip install -r /scratch/$USER/verl-gaudi-support/verl_compat/requirements.txt
uv pip install --no-deps -e /scratch/$USER/sglang-fork/python
```

Notes and gotchas:

- `--system-site-packages` is the whole trick. Without it the venv has no
  `habana_frameworks` and `VERL_PLATFORM=hpu` will fail at first device touch.
- Pin `ray[default]==2.53.0` per `requirements.txt`. The host has 2.54.1 in `~/.local`;
  `PYTHONNOUSERSITE=1` keeps it out, but do not rely on that alone.
- `requirements.txt` pulls `liger-kernel` and `flash-attn` transitively in some
  resolutions — both are CUDA-only. Install with them excluded; verl's Gaudi path never
  calls them (`attention_utils.py` routes Intel to the pure-torch pad/unpad helpers).
- Do **not** `pip install verl`. It is used from source via `PYTHONPATH`, and
  `verl/__init__.py` prints its own import path at startup precisely so you can confirm
  the right tree won.
- After the install, verify `uv pip list | grep -i torch` shows **nothing** — if uv pulled
  a PyPI torch in, the Habana bridge is shadowed and everything downstream is a lie.

### 4.4 Sanity gate before touching verl

```python
import torch, habana_frameworks.torch.hpu as hthpu
assert hthpu.is_available() and hthpu.device_count() == 8
print(torch.__version__)     # expect 2.7.1a0+...
```

Must be run **with `HABANA_LOGS` set** — see §6.

---

## 5. Data and reward

The existing `/scratch/$USER/datasets/gsm8k/*.parquet` is **MILES format**
(`question` / `answer` / `label` / `messages`, answers in `\boxed{}`). verl cannot read it
and its reward scorer will not match. Regenerate:

```bash
python examples/data_preprocess/gsm8k.py --local_save_dir /scratch/$USER/data/gsm8k_verl
```

Produces `train.parquet` (7473) and `test.parquet` (1319) with verl's schema:

| Column | Value |
|---|---|
| `data_source` | `openai/gsm8k` ← this is what routes the reward function |
| `prompt` | `[{"role": "user", "content": question + ' Let\'s think step by step and output the final answer after "####".'}]` |
| `ability` | `math` |
| `reward_model` | `{"style": "rule", "ground_truth": "<number>"}` |
| `extra_info` | `split`, `index`, `answer`, `question` |

**Reward** needs no configuration. `verl/utils/reward_score/__init__.py` dispatches on
`data_source == "openai/gsm8k"` to `gsm8k.compute_score`, which extracts with
`#### (-?[0-9.,]+)` in `strict` mode and returns 1.0 / 0.0. Binary reward ⇒ any
"is this rollout correct" threshold is 0.5.

**`data.prompt_key` must be `prompt`** (the default). The DAPO baseline script sets
`PROMPT_KEY=source_prompt`; that is wrong for GSM8K and must be overridden back.

**Model.** `/scratch/$USER/models/Qwen3-0.6B` and `Qwen2.5-3B-Instruct` are on disk, and
`/scratch/$USER/hf_cache` already has 40 GB. Start the smoke test on **Qwen3-0.6B**
(fast, local, no download), then move to Qwen2.5-3B-Instruct for a real GSM8K run. Do not
start on Qwen3-4B-Base — a 4-step failure is expensive at that size.

---

## 6. Environment variables

All load-bearing. Export in `start_ray.sh` **before `ray start`** so the raylet — and
therefore every actor — inherits them.

### Must be set

| Variable | Value | Why |
|---|---|---|
| `PT_HPU_GPU_MIGRATION` | `1` | `torch.cuda.*` → `torch.hpu.*`. `platform_hpu.py` is built on the CUDA namespace and does not work without it. Before torch import. |
| `VERL_PLATFORM` | `hpu` | Pins detection. Auto-detect caches once per process; one failed import in one actor makes that process think it is on NVIDIA forever. |
| `PT_HPU_LAZY_MODE` | `0` | Lazy mode crashes FSDP flat-param sharding. |
| `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES` | `1` | RL rollouts vary sequence length every step; without this, recipe count grows until compilation fails. |
| `HABANA_LOGS` | `/scratch/$USER/habana_logs` | **Verified failure here.** Without it the container aborts at import: `spdlog_ex: Failed opening file /var/log/habana_logs/synapse_utils_log.txt for writing: Read-only file system` — then core-dumps. Already forwarded by `constants_ppo.py`. |
| `RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES` | `1` | Otherwise Habana's eager distributed init asserts "not enough devices". Before `ray start`. |
| `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` | `python` | upb Descriptors are unpicklable; Ray fails serializing `WorkerDict` behind a bogus "async flag" error. |
| `HABANA_SYSTEM_FORK_UNSAFE_EXEC` | `1` | Fork safety for the Ray/SGLang process trees. |
| `PYTHONNOUSERSITE` | `1` | Keeps `~/.local`'s CUDA torch 2.9.0 and ray 2.54.1 out. |
| `RAY_TMPDIR` | `/dev/shm/ray_$USER` | tmpfs. 252 GB free here. Overlayfs/BeeGFS makes `ray start` crawl and times out the dashboard. |
| `HPU_CARDS_COUNT` | `8` | `hl-smi` reports all node cards, not your allocation. |
| `PYTHONPATH` | `<repo>/verl_compat:/scratch/$USER/sglang-fork/python` | ⚠ see §7 — on the `main_ppo` path this **only** arrives via the raylet. |
| `HF_HOME` | `/scratch/$USER/hf_cache` | ⚠ see §7. **Never leave this on `/home`** — 93% full, 7.6 GB left. |

### Performance (start off, enable after the smoke test passes)

| Variable | Value | Note |
|---|---|---|
| `VERL_HPU_TORCH_COMPILE` | `1` | Gaudi's documented FSDP path (eager + `compile(hpu_backend)`). |
| `VERL_HPU_FUSED_SDPA` | `1` | Inert unless the model also runs `attn_implementation=sdpa`. |

Turn both on only once a plain run survives >10 steps, so a failure has one cause.

---

## 6.5 Cache placement — everything on `/scratch`

`/home` is **93% full with 7.6 GB free**. Anything that lands there will fail a run
mid-flight. And caches are currently landing in the *repo*: `.graph_dumps/` already exists
at `verl_compat/.graph_dumps`, and the root `.gitignore` carries `verl_compat/.habana_cache/`
— both written into CWD because nothing redirects them.

Set all of these. `CACHE_ROOT=/scratch/$USER/verl-cache`.

| Variable | Value | What it catches |
|---|---|---|
| `HF_HOME` | `/scratch/$USER/hf_cache` | models + hub. Already 40 GB there — reuse it, don't make a second copy. |
| `HF_DATASETS_CACHE` | `$HF_HOME/datasets` | GSM8K download/prepare |
| `HF_HUB_CACHE` | `$HF_HOME/hub` | explicit; `automodel/transformer_impl.py` imports it from `huggingface_hub.constants` |
| `XDG_CACHE_HOME` | `$CACHE_ROOT/xdg` | **catch-all.** Anything well-behaved that you forgot lands here instead of `~/.cache`. |
| `TORCHINDUCTOR_CACHE_DIR` | `$CACHE_ROOT/inductor` | ⚠ **`VERL_HPU_TORCH_COMPILE=1` means torch.compile runs.** Nothing in the repo sets this. |
| `TRITON_CACHE_DIR` | `$CACHE_ROOT/triton` | inductor's codegen cache |
| `TORCH_HOME` / `TORCH_EXTENSIONS_DIR` | `$CACHE_ROOT/torch` | JIT extension builds |
| `PT_HPU_RECIPE_CACHE_CONFIG` | `$CACHE_ROOT/habana_recipe,false,20480` | ⚠ **Habana SynapseAI recipe cache.** Dir, `clear_on_start=false`, size MB. Persisting it across runs is what stops re-compiling every graph on every launch. |
| `HABANA_LOGS` | `/scratch/$USER/habana_logs` | load-bearing, §6 — the container aborts without it |
| `GRAPH_VISUALIZATION_DIR` *(or just `cd` elsewhere)* | `$CACHE_ROOT/graph_dumps` | stops `.graph_dumps/` appearing in the repo |
| `RAY_TMPDIR` | `/dev/shm/ray_$USER` | tmpfs, 252 GB free. Not scratch — BeeGFS is too slow for Ray's sockets. |
| `TMPDIR` | `$CACHE_ROOT/tmp` | everything else's scratch space |
| `WANDB_DIR` | `$CACHE_ROOT/wandb` | run dirs |
| `WANDB_CACHE_DIR` | `$CACHE_ROOT/wandb-cache` | artifact cache |
| `UV_CACHE_DIR` | `$CACHE_ROOT/uv` | ⚠ the wheel cache for the env build itself — set it **before** the first `uv pip install` |
| `APPTAINER_CACHEDIR` | `$CACHE_ROOT/apptainer` | ⚠ the 1.22.2 image is multi-GB and `apptainer pull` caches to `~/.apptainer` by default. Set this **before** pulling. |
| `trainer.default_local_dir` | `/scratch/$USER/checkpoints/...` | verl's own default is `checkpoints/<project>/<experiment>` — **relative to CWD**. |

Two ordering notes: `UV_CACHE_DIR` and `APPTAINER_CACHEDIR` matter during Phase 0, before
anything else exists — set them first or the pull and the install fill `/home`. And
`PT_HPU_RECIPE_CACHE_CONFIG` must be exported before `ray start`, like every other
`PT_HPU_*`, so the raylet passes it down (it is already covered by the `PT_HPU_*` family
forwarding in `constants_ppo.py`).

The sibling `miles-gaudi` project already runs this pattern —
`SGLANG_HPU_RECIPE_CACHE_CONFIG="${DIR},false,20480"` with a per-model directory under
`/scratch/<user>/habana_recipe_cache/` — worth copying, including the per-model split, since
recipes are shape- and model-specific and mixing them just thrashes the cache.

Verify after a run with `du -sh /home/$USER` — it should not have grown.

---

## 7. ⚠ Two forwarding gaps specific to the `main_ppo` path

IOHER's entry point forwards more than the stock one does. Read
`get_ppo_ray_runtime_env()` in `verl/trainer/constants_ppo.py`: it forwards `PT_HPU_*`,
`VERL_HPU_*`, `SGLANG_HPU_*`, `VERL_PLATFORM`, `HABANA_LOGS`,
`HABANA_SYSTEM_FORK_UNSAFE_EXEC`, `RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES`,
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` — and **nothing else**.

Missing versus `main_ioher.py`, which forwards them explicitly:

1. **`PYTHONPATH`** — not forwarded. Everything therefore depends on `start_ray.sh` having
   exported the correct `PYTHONPATH` before `ray start`, so actors inherit it from the
   raylet. This is the single most important reason to fix `start_ray.sh`'s hardcoded
   paths (§3) rather than working around them in the launch script.
2. **`HF_*`** — not forwarded. The baseline script exports `HF_HOME` in its *own* shell,
   which never reaches a Ray actor on this path. Tokenizer/processor loading happens
   inside the TaskRunner actor.

**`WANDB_*` is forwarded by neither.** See §8.

Two ways to close the gaps; belt and braces is fine:

```bash
# (a) in start_ray.sh, before `ray start` — inherited by every actor
export PYTHONPATH=... HF_HOME=... WANDB_API_KEY=...

# (b) on the launch command line — explicit, survives a raylet you did not start
+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=/scratch/$USER/hf_cache \
+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH \
+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=$WANDB_API_KEY
```

---

## 8. W&B

`Tracking.__init__` calls `wandb.init(project=trainer.project_name,
name=trainer.experiment_name, entity=os.environ["WANDB_ENTITY"], config=<full resolved
config>)`. That runs **inside the TaskRunner Ray actor**, not in your shell.

Config:

```
trainer.logger='["console","wandb"]'
trainer.project_name=verl_grpo_gsm8k_gaudi
trainer.experiment_name=qwen3_0p6b_gsm8k_grpo_hpu_smoke
```

Auth — pick one:

- `wandb login` on the node writes `~/.netrc`; Ray actors are local and share `$HOME`, so
  this works **provided** the container binds your real home. Under `--containall` it does
  not; bind it or use the env var.
- Forward `WANDB_API_KEY` through `runtime_env` per §7. More robust, and the only option
  in an isolated container.

Also available: `WANDB_ENTITY`, `trainer.wandb_proxy` (sets `wandb.Settings(https_proxy=)`),
`WANDB_MODE=offline` for an air-gapped run.

Metrics you will actually want to watch on this port:

| Metric | Watch for |
|---|---|
| `critic/rewards/mean` | should climb from ~0 on GSM8K |
| `actor/grad_norm` | **NaN ⇒ stop.** The known Gaudi SDPA failure mode. |
| `actor/entropy` | pinned at `ln(vocab)` ⇒ weight corruption (the `param_offload` signature) |
| `perf/hpu/graph_compilation/*` | growing every step ⇒ shape churn, check `REFINE_DYNAMIC_SHAPES` |
| `perf/hpu/cpu_fallback/*` | growing ⇒ ops silently running on host |
| `perf/hpu/memory_defragmentation/*` | non-zero ⇒ the eager-attention allocator storm |
| `perf/mfu/actor`, `timing_s/step` | the CUDA-comparison numbers |

The `perf/hpu/*` family comes from `_collect_hpu_runtime_metrics()` in `fsdp_workers.py`
and is exactly the instrumentation this port added for this purpose. Use it.

---

## 9. Execution plan

### Phase 0 — environment (once)

1. `apptainer pull` the 1.22.2 / torch 2.7.1 image to `/scratch/$USER/apptainer/`.
2. Start a shell in it with `-B /scratch/$USER` and `--env HABANA_LOGS=...`.
3. `uv venv --system-site-packages`, install `requirements.txt`, `uv pip install --no-deps -e /scratch/$USER/sglang-fork/python`.
4. **Gate:** §4.4 sanity check passes — 8 cards visible, torch 2.7.1a0.
5. **Gate:** `python -c "import sglang; print(sglang.__version__)"` → `0.4.9`.

### Phase 1 — repath the scripts

6. In `verl_compat/start_ray.sh`: replace the `PYTHONPATH` export with the two real paths;
   add `HABANA_LOGS`, `HF_HOME`, `PYTHONNOUSERSITE`; keep `HPU_CARDS_COUNT=8` overridable.
7. Copy `examples/grpo_trainer/run_qwen3_4b_dapo_fsdp.sh` →
   `run_gsm8k_grpo_gaudi.sh`. In the `hpu` branch: fix `SGLANG_HPU_ROOT`, `HF_CACHE_DIR`;
   set `PROMPT_KEY=prompt`, GSM8K files, `MAX_PROMPT_LENGTH=512`,
   `MAX_RESPONSE_LENGTH=512` (GSM8K is short — do not carry DAPO's 2048/4096).
   Start with `VERL_HPU_TORCH_COMPILE=0` and `VERL_HPU_FUSED_SDPA=0`.
8. Verify with `DRY_RUN=1` before running anything.

### Phase 2 — data

9. Generate the verl-schema GSM8K parquet (§5). Assert the columns.

### Phase 3 — smoke test (the real gate)

10. `HPU_CARDS_COUNT=8 bash start_ray.sh`, then confirm `ray status` shows `HPU: 8.0`.
11. Run with `TOTAL_TRAINING_STEPS=10`, `ROLLOUT_N=4`, `TRAIN_BATCH_SIZE=16`,
    Qwen3-0.6B, `trainer.val_before_train=False`, `logger='["console"]'`.
12. **Gate: does it pass step 4?** This is the whole question (§2).
    - Passes ⇒ continue.
    - Fails at `torch.nested.*` with `synStatus=26` ⇒ the fix is incomplete; capture the
      exact op and traceback and take it to §10.

### Phase 4 — real run

13. Re-enable `val_before_train`, wandb, `test_freq`. Move to Qwen2.5-3B-Instruct,
    `TRAIN_BATCH_SIZE=64`, `ROLLOUT_N=8`, `PPO_MINI_BATCH_SIZE=16`.
14. Enable `VERL_HPU_TORCH_COMPILE=1`, then separately `VERL_HPU_FUSED_SDPA=1` with
    `++actor_rollout_ref.model.override_config.attn_implementation=sdpa`.
    **One at a time**, watching `actor/grad_norm` for NaN after each.
15. Keep every offload **off**, including `ref` — §10.

### Fallback if Phase 3 fails at step 4

`RayIOHERTrainer` is the only trainer known to complete long runs on Gaudi: its `fit()` is
a full reimplementation that passes `DataProto` straight through and never calls
`to_tensordict()` / `left_right_2_no_padding()`. `97b9172` deliberately made
`ioh_sft_coef=0` skip all inoculation work — judge pass included — so:

```bash
bash recipe/ioher/run_ioher.sh actor_rollout_ref.actor.ioh_sft_coef=0 ...
```

is, per that commit, "a genuine GRPO run on the only trainer path that works on Gaudi".
Same algorithm, different plumbing. Not the goal, but it produces the number.

---

## 9.5 Entrypoint options — there is no `recipe/grpo`

**GRPO is not a recipe in verl. It is an advantage estimator** —
`algorithm.adv_estimator=grpo` on the stock PPO trainer. `examples/grpo_trainer/` holds
*launch scripts*, not a recipe; every one of them calls `python -m verl.trainer.main_ppo`.
There is no `recipe/grpo/` directory and no `main_grpo.py`.

So the choice is which **trainer + worker pair** to run GRPO on. That pairing is the whole
question on Gaudi, because there are two mutually incompatible data protocols in this
verl version:

| | Trainer speaks | Workers speak |
|---|---|---|
| **Modern path** | TensorDict + nested jagged (`to_tensordict` → `left_right_2_no_padding`) | `workers/engine_workers.py` |
| **Legacy path** | `DataProto`, dense padded | `workers/fsdp_workers.py` |

Mixing them is what `97b9172` discovered the hard way (`'TensorDict' object has no
attribute 'meta_info'`). A recipe is only useful to us if **both halves are legacy**.

### Which recipes actually bypass the nested path

A literal grep for `to_tensordict` in a recipe file is misleading — the conversions live in
five *inherited* `RayPPOTrainer` helpers (`_compute_old_log_prob`, `_compute_ref_log_prob`,
`_compute_values`, `_update_actor`, `_update_critic`). A recipe that calls those inherits
the nested path even though its own file is clean. Checking for calls to those five:

| Entrypoint | Calls nested helpers | Workers | Verdict for plain GRPO on Gaudi |
|---|---|---|---|
| `verl.trainer.main_ppo` | yes (11 sites in `ray_trainer.py`) | `engine_workers` | Canonical. Step-4 risk (§2). |
| `recipe/dapo/main_dapo.py` | **5 of 5** | inherits `main_ppo_v0` | ❌ and **broken at import** — see below |
| `recipe/entropy/main_entropy.py` | **0** — dispatches `actor_rollout_wg.compute_log_prob/generate_sequences/update_actor(batch)` directly | `fsdp_workers.{Async,}ActorRolloutRefWorker` ✅ | Matched legacy pair. Closest thing to a GRPO recipe with its own entrypoint. |
| `recipe/ioher/main_ioher.py` | **0** | `IOHERActorRolloutRefWorker(AsyncActorRolloutRefWorker)` ✅ | Matched legacy pair. **The only one proven on Gaudi.** |
| `sppo`, `spo`, `prime`, `rep_exp`, `spin` | 0 | legacy | Bypass the nested path, but each carries its own algorithm — not plain GRPO. |
| `atropos`, `fault_recover` | 3 / 5 | — | ❌ |

### ⚠ `recipe/dapo` is broken at import in this snapshot

`main_dapo.py:26` does:

```python
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
```

but `main_ppo.py` has **no module-level `TaskRunner`**. Its only module-level definitions
are `run_ppo`, `TaskRunnerV1`, and `main`; `TaskRunner` is imported *inside* `main()` at
line 164 (`from verl.trainer.main_ppo_v0 import TaskRunner`). So
`python -m recipe.dapo.main_dapo` dies with `ImportError` before anything else happens.

This is an upstream bug in this vendored snapshot, not a Gaudi issue — but it rules out
what would otherwise be the obvious GRPO-family entrypoint. (It would not have helped
anyway: `dapo_ray_trainer.py` calls all five nested helpers.)

### Decision: build `recipe/grpo_hpu/` from the SPPO skeleton

**No IOHER.** SPPO is the base. It has the property that matters — a `RayPPOTrainer`
subclass whose `fit()` dispatches `actor_rollout_wg.generate_sequences / compute_log_prob /
update_actor(batch)` **directly with `DataProto`**, paired with
`SPPOActorRolloutRefWorker(fsdp_workers.ActorRolloutRefWorker)`. Matched legacy pair, no
nested tensors, no step-4 exposure.

It is also the skeleton IOHER itself was cloned from (`REQUIRED_VERL.txt`: *"same rolling
verl pin as the SPPO recipe, since this recipe subclasses the same `DataParallelPPOActor` /
`ActorRolloutRefWorker` entry points"*), so this is the same route, taken deliberately and
without the inoculation machinery.

---

## 9.6 Turning SPPO into GRPO

Two separate jobs: **swap the algorithm**, and **repair SPPO's staleness**. Do not conflate
them — the second is where the danger is.

### A. Algorithm swap — three edits

SPPO is not GRPO. It replaces verl's advantage machinery wholesale, so three things come out
and their stock equivalents go in.

| # | SPPO has | Replace with |
|---|---|---|
| 1 | its own module-level `compute_advantage(data, beta)` in `sppo_ray_trainer.py:68` — `seq_level_rewards = rewards − softmean(rewards, β)`. Never touches `AdvantageEstimator`; this is why the config says `adv_estimator: null`. | verl's `compute_advantage(batch, adv_estimator=grpo, num_repeat=rollout.n, norm_adv_by_std_in_grpo=True, config=self.config.algorithm)` — writes `advantages` / `returns` |
| 2 | `compute_sppo_loss(...)` in `dp_actor.py:43` and `DataParallelSPPOActor`, keyed on `sppo_eta` | stock `DataParallelPPOActor` — its `update_policy` already dispatches through `get_policy_loss_fn(policy_loss.loss_mode)`, i.e. the GRPO clip loss. Point `sppo_worker.py:89,112` at it. |
| 3 | config: `adv_estimator: null`, `sppo_eta: 1.0`, `_target_: recipe.sppo.config.SPPOActorConfig` | `adv_estimator: grpo`, drop `sppo_eta`, drop `_target_` (stock `FSDPActorConfig` is fine — we add no fields) |

Delete after that: `config.py` (`SPPOActorConfig`), `dp_actor.py` entirely, and the
`softmean` / `compute_advantage` block in the trainer. What remains **is** a plain GRPO
trainer in DataProto form.

Also for HPU: `config/sppo_trainer.yaml` sets `rollout.tensor_model_parallel_size: 2` →
**must be 1** (everything on Gaudi is validated at TP=1, and TP=2 changes the card
arithmetic in §1).

### B. ⚠ SPPO is stale against this verl snapshot — four gaps

SPPO and IOHER pin the *same* upstream commit (`bcb638`), but IOHER was carried forward to
this tree and SPPO was not. Diffing the two `__init__` methods shows exactly what IOHER had
to add. All of it applies here.

| Gap | Consequence | Fix |
|---|---|---|
| **`checkpoint_manager.update_weights()` is never called.** SPPO's `fit()` has **0** calls; stock `ray_trainer.py` has 4, IOHER has 2. | 🔴 **Silent and severe.** That call is literally *"Update weights from actor worker group to rollout replicas"* (`checkpoint_engine/base.py:470`). Without it the SGLang servers keep generating from the **initial** weights for the whole run. Loss moves, `grad_norm` looks healthy, reward never improves — and nothing errors. | Add `self.checkpoint_manager.update_weights(self.global_steps)` after `_load_checkpoint()` and after each `update_actor`, as IOHER does at lines 803 and 963. |
| `__init__` never sets `self.checkpoint_manager`, `self.ref_in_actor`, `self.use_prefix_grouper`, `self.use_legacy_worker_impl` | `RayPPOTrainer.init_workers()` expects them → `AttributeError` at startup | copy IOHER `ioher_ray_trainer.py:102–109` verbatim |
| `__init__` never sets `self.use_teacher_policy` | same | `self.use_teacher_policy = need_teacher_policy(config)` |
| `__init__` never calls `self._init_dump_executor()` | `rollout_data_dir` / `validation_data_dir` dumps crash — `self._dump_executor` missing | add it after `_create_dataloader(...)` |

Checked and **not** a problem, despite looking like one: SPPO calls
`need_reference_policy(role_worker_mapping)` while IOHER passes `config`. Both work —
`verl/trainer/ppo/utils.py:79` branches on `isinstance(k, Role)` and accepts either. Leave
it alone.

Also worth aligning while in there: SPPO hand-rolls `batch.pop(...)` at line 181 where the
current trainer offers `self._get_gen_batch(batch)`, and uses `compute_reward(batch,
self.reward_fn)` where the current one uses `_compute_reward_colocate` / `extract_reward`.
Both old APIs still exist in this snapshot, so neither is urgent — but `_get_gen_batch` is
the one that keeps `raw_prompt` handling correct if you ever extend this.

### C. Gaudi wiring `main_sppo.py` lacks

`main_sppo.py:38-52` builds a bare `runtime_env` with three variables. Compare
`main_ioher.py`, which the fork extended. Port these across:

- forward `PT_HPU_*`, `VERL_HPU_*`, `VERL_PLATFORM` into `runtime_env.env_vars`
- forward `PYTHONPATH`, `HF_HOME`, `HF_DATASETS_CACHE` (§7 — the stock path does **not**)
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
- `include_dashboard=False`
- pin `ray_kwargs.ray_init.address` to `127.0.0.1:6381` in the yaml, as IOHER's config does

Alternatively, drop `main_sppo.py`'s hand-rolled `run_ppo` and call
`get_ppo_ray_runtime_env()` from `verl/trainer/constants_ppo.py`, which the fork already
patched for HPU — then only `PYTHONPATH` / `HF_*` / `WANDB_*` remain to add.

### D. Revised Phase 3

Replaces §9 Phase 3. The gate is no longer "does it pass step 4" — this path never had that
problem — it is **"do rewards actually move"**, because gap B-1 fails silently.

1. Build `recipe/grpo_hpu/` from `recipe/sppo/` with edits A + B + C.
2. Smoke run: Qwen3-0.6B, `TOTAL_TRAINING_STEPS=10`, `ROLLOUT_N=4`,
   `TRAIN_BATCH_SIZE=16`, `logger='["console"]'`, `val_before_train=False`.
3. **Gate 1** — it starts, 4 FSDP + 4 SGLang come up, a step completes.
4. **Gate 2** — `critic/rewards/mean` changes across steps. If it is pinned flat, suspect
   B-1 first: confirm `update_weights` is being called and the rollout servers are
   receiving new weights.
5. Only then: wandb on, Qwen2.5-3B-Instruct, `VERL_HPU_TORCH_COMPILE=1`, then
   `VERL_HPU_FUSED_SDPA=1` — one at a time, watching `actor/grad_norm` for NaN.

---

## 10. Known Gaudi failure modes to recognize

| Symptom | Cause | Action |
|---|---|---|
| entropy pinned at `ln(vocab)`, rewards → −1, `pg_loss` → exactly 0 | **FSDP `param_offload` silently corrupts weights on Gaudi.** No crash. | All offloads off, `ref` included (stock default is `True`). |
| `actor/grad_norm` NaN | stock torch SDPA lowering under GPU Migration | Use FusedSDPA (`VERL_HPU_FUSED_SDPA=1` + `attn_implementation=sdpa`) or eager. |
| `Graph duplication failed. synStatus=26` | a nested-tensor op SynapseAI cannot compile | Note the exact op. `narrow` is fixed at HEAD; others were left alone as working. |
| `synStatus=8 Device acquire failed` | something already holds the card | Three causes: rollout on a training card; a CPU-only process importing torchao; `n_gpus_per_node` too high. |
| `spdlog_ex ... /var/log/habana_logs ... Read-only` | `HABANA_LOGS` unset in a container | Set it. **Reproduced on this box.** |
| `"You set the async flag, but the actor does not have any coroutine functions"` | almost never about async — it is a masked pickling error | Check `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`; the real traceback is printed by the `verl/__init__.py` unmask patch. |
| `Total available GPUs 0 is less than total desired GPUs N` | no Ray cluster, or platform detected as nvidia | Start `start_ray.sh` first; check `VERL_PLATFORM=hpu` reached the actor. |
| step time far above CUDA | expected — no varlen attention kernel on Gaudi | `use_remove_padding=False` is forced; padded compute is the known dominant gap. |

---

## 11. Open questions

1. **Does `main_ppo` survive past step 4 at HEAD?** The single unknown that decides
   Phase 3 vs the fallback.
2. **1.22.2 userspace on a 1.24.0 driver** — expected to work (N−2, and 1.16 already
   initialized fine), but not verified for the full training path.
3. **sglang 0.4.9 + transformers.** Its pin is `transformers==4.53.0`; verl's
   `requirements.txt` says only `transformers!=5.6.0`. Let SGLang's pin win, and check the
   model's architecture is supported by 4.53.
4. **`use_remove_padding=False` + GSM8K's short sequences.** Padded compute hurts less at
   512 tokens than at 4096, so the CUDA gap should look better here than in the DAPO
   baseline. Worth measuring rather than assuming.
5. **Weight-sync bucket size.** The HPU path serializes through CPU;
   `update_weights_bucket_megabytes=128` was chosen for a 4B model. Probably fine for
   0.6–3B, but watch host RSS during the first sync.

---

## Appendix — verified facts about this box

```
node                gaudi005
accelerators        8 × HL-225, 98304 MiB each
driver              hl-1.24.0-fw-62.6.2.0   (SynapseAI 1.24.0)
apptainer           /usr/bin/apptainer, /usr/bin/singularity
uv                  0.12.5  (/home/$USER/.local/bin/uv)
system python       3.12.9  — torch 2.9.0+cu128 (CUDA!) in ~/.local, ray 2.54.1
habana_frameworks   present in mamba base but NO .torch submodule → unusable
/scratch            BeeGFS, 462 TB free
/home               93% full, 7.6 GB free  ← keep caches off it
/dev/shm            252 GB free            ← RAY_TMPDIR
network             pypi ✅  vault.habana.ai pip ✅  vault gaudi-docker ✅  huggingface ✅  wandb ✅
gaudi_pytorch.sif   SynapseAI 1.16.0, torch 2.2.2a0, py3.10          → too old
verl_sglang.sif     torch 2.9.1+cu129, sglang 0.5.9, tensordict 0.10 → CUDA reference image
sglang-fork         sglang 0.4.9 + HPU support, torch==2.7.1 pin     → USE THIS
datasets/gsm8k      MILES schema, not verl's                        → regenerate
models on disk      Qwen3-0.6B, Qwen2.5-3B-Instruct, Qwen3-32B, Qwen3-VL-*
hf_cache            /scratch/$USER/hf_cache, 40 GB populated
```

SynapseAI → torch map from the vault (`ubuntu24.04`):

| SynapseAI | torch | tag |
|---|---|---|
| 1.21.x | 2.6.0 | |
| **1.22.2** | **2.7.1** | `1.22.2-32` ← recommended |
| 1.23.0 | 2.9.0 | |
| 1.24.0 | 2.10.0 | `1.24.0-1007` (matches driver, but torch too new for sglang 0.4.9) |

---

## §12. miles-gaudi patches: checked, NOT applicable

Verdict: **do not apply any of `/scratch/$USER/miles-gaudi/patches/`.** They target a
different SGLang.

`miles-gaudi/setup_gaudi_venv.sh:17` pins its SGLang to upstream commit
`cb05a44f35a7c9e27e46d74112cc841ca674ef43` (a modern `main`) and then applies
`sglang-hpu-gaudi-full.patch` on top. Our rollout engine is `/scratch/$USER/sglang-fork`,
**SGLang 0.4.9 with HPU support already vendored in-tree** — nothing to patch on.

Measured, not assumed — `patch -p1 --dry-run` against `sglang-fork`: 4 × "can't find file to
patch", 17 × "Hunk FAILED", exit 1. The paths the patch edits do not exist in 0.4.9:

| miles patch expects | sglang-fork 0.4.9 has |
| --- | --- |
| `layers/rotary_embedding/base.py` | `layers/rotary_embedding.py` (module, not package) |
| `utils/common.py` | `utils.py` |
| `arg_groups/overrides.py` | — |
| `mem_cache/allocation.py`, `mem_cache/allocator/paged.py` | — |
| `layers/attention/attention_registry.py` | — (no backend registry) |
| `model_executor/runner/hpu_graph_runner.py` | `model_executor/hpu_graph_runner.py` |
| `layers/attention/hpu_{fused,blockwise,paged_v2}_backend.py` (added by patch) | `layers/attention/hpu_attn_backend.py` (already present) |

Consequence for the runner script: **the `SGLANG_HPU_*` names in
`miles-gaudi/run_grpo_qwen3_0p6b_gaudi.sh` are inert here.** miles sets
`SGLANG_ATTENTION_BACKEND=hpu_fused`, `SGLANG_HPU_DECODE_SEQ_BUCKET_STEP`,
`SGLANG_HPU_BLOCK_BUCKET_LIMIT`, `SGLANG_HPU_DECODE_GRAPH_WARMUP_BATCHES` … none of which
0.4.9 reads. The knobs our fork *does* read (`sglang/srt/hpu_utils.py:118-146`) are:

```
SGLANG_HPU_PREFILL_BUCKET_{MIN,STEP,MAX}          1024 / 1024 / 6144
SGLANG_HPU_PREFILL_PREFIX_BUCKET_{MIN,STEP,MAX}    128 /  128 / 2560
SGLANG_HPU_DECODE_BLOCK_BUCKET_{MIN,STEP,MAX}      128 /  128 / 4480
SGLANG_HPU_DECODE_BATCH_BUCKET_{MIN,STEP,MAX}        1 /   32 /  128
SGLANG_HPU_USE_CONTIGUOUS_PA                      true
SGLANG_HPU_SKIP_WARMUP                            false
```

`env/run_grpo_gsm8k.sh` sets the prefill/decode buckets to the GSM8K geometry (the stock
prefill MIN of 1024 would pad every 512-token prompt to 1024 and capture six unused graphs).

The one idea worth carrying over from miles is structural, not textual: miles' setup
*verifies its imports after install*. Walking the real import chain is what found §13.

## §13. Runtime gaps found by walking the import chain

`sglang[runtime_common]` does not cover the HPU path, so these all failed at
rollout-worker startup rather than at setup. Now installed by `env/setup_uv_env.sh` §6b/§6c.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: decord` | `sglang/srt/utils.py:77` imports it at module scope | `pip install decord` |
| `ImportError: cannot import name 'default_cache_dir' from 'triton.runtime.cache'` | triton 3.8.0 got pulled in transitively; sglang 0.4.9 needs the pre-3.4 API | `pip install --no-deps triton==3.3.1` |
| `ImportError: cannot import name 'find_bucket' from 'vllm_hpu_extension.bucketing.linear'` | `find_bucket` was deleted by upstream commit `ad558281` ("Bucketing refactoring #223"), so tag `v1.22.0` lacks it; PyPI `vllm_hpu_extension==0.1` is a 12 kB stub with no `bucketing` package at all | pin commit `891db1d2` (2025-07-02, the last one before that refactor) |
| `TypeError: '>=' not supported between instances of 'NoneType' and 'int'` at `fp8_utils.py:82` | see below | `env/patches/sglang-0.4.9-is_cuda-hpu.patch` |
| `ModuleNotFoundError: cachetools` | recipe.sppo trainer chain | `pip install cachetools` |

The fp8 one is the interesting one. GPU Migration spoofs `torch.cuda.is_available()=True`
**and `torch.version.cuda="11.8"`** (both measured on this box), and sglang's probe is
literally `def is_cuda(): return torch.cuda.is_available() and torch.version.cuda`. So
`is_cuda()` returns truthy on Gaudi and every `_is_cuda`-gated CUDA path switches on.
`cutlass_fp8_supported()` then calls `get_device_capability()`, which returns `(None, None)`
on HPU *by design* (there is a TODO in `sglang/srt/utils.py:1650` saying so), and the module
fails to import. The fork gates its HPU code on `is_hpu()`, so `is_hpu()` must win:

```python
def is_cuda():
    if is_hpu():
        return False
    return torch.cuda.is_available() and torch.version.cuda
```

This is the same *class* of fix the miles patch makes wholesale for the newer SGLang — which
is why the question was worth asking even though the patch itself does not apply.

## §14. Why Ray must start inside the training container session

`env/start_ray_gaudi.sh` refuses to run from the host, and `env/run_grpo_gsm8k.sh` starts
Ray itself inside the one container session it opens. Measured: a process daemonised inside
`apptainer exec` does survive the exec's exit (verified with `setsid sleep 45`), but apptainer
then tears down the SIF's squashfuse mount — `INFO: Terminating squashfuse_ll after timeout`
— so the surviving raylet/GCS keep running with a broken view of `/usr`. Ray therefore lives
either in an interactive `env/shell.sh`, or in the session `run_grpo_gsm8k.sh` opens for
itself (where it is torn down by an `EXIT` trap that preserves the trainer's exit code).

## §15. W&B wiring (supersedes §8)

Off by default (`LOGGER='["console"]'`). Turn on with `WANDB=1`.

Run identity comes from **hydra**, not env: `verl/utils/tracking.py:80` is
`wandb.init(project=project_name, name=experiment_name, entity=os.environ.get("WANDB_ENTITY"))`,
fed from `trainer.project_name` / `trainer.experiment_name`. `WANDB_PROJECT` and `WANDB_NAME`
are never read — set `PROJECT_NAME` / `EXPERIMENT_NAME` instead.

Auth on this box: `~/.netrc` already holds an `api.wandb.ai` entry and resolves inside the
container (verified: `wandb.Api().default_entity` → `$USER-arizona-state-university`).
That works because apptainer binds `$HOME` by default and `env/shell.sh` does not use
`--containall`. `WANDB_API_KEY` is the more robust route — it survives a bind-mount change
and reaches Ray workers through `main_sppo`'s `runtime_env`.

Three gaps closed while wiring this:

1. `env/shell.sh` forwarded `WANDB_API_KEY`/`WANDB_ENTITY` but **not** `WANDB_MODE`,
   `WANDB_RUN_GROUP`, `WANDB_TAGS`, or the proxy vars. `main_sppo.py:63` forwards
   `WANDB_MODE` into each worker's `runtime_env`, so it was being propagated to workers
   from a container that never received it. Added to the forward list.
2. `wandb.init()` runs in the driver actor *after* the model loads, so a bad credential cost
   a full startup and a Ray teardown to discover. The runner now preflights the credential
   before claiming any card, and names the exact fix in the error.
3. `WANDB=1` also sets `WANDB_RUN_GROUP=$EXPERIMENT_NAME` so repeat runs group in the UI.

Offline fallback for flaky egress: `WANDB=1 WANDB_MODE=offline`, then
`$VENV_DIR/bin/wandb sync $WANDB_DIR/wandb/offline-run-*`. The preflight skips the
credential check in offline mode.

## §16. SPPO recipe staleness against fsdp_workers.py

`SPPOActorRolloutRefWorker.init_model()` is a **copy of an older**
`ActorRolloutRefWorker.init_model()`, not a call to it. Every time upstream adds setup to the
parent's `init_model`, the recipe silently misses it — and the failure lands at worker init,
minutes into a run, on all four actors at once.

First instance hit: verl's QAT support. The parent calls `self._init_qat_config()` at
`fsdp_workers.py:1004` before `_build_model_optimizer`, which then reads `self._qat_enabled`
unconditionally at line 645 (and again at 934, on the weight-sync path). The recipe never
called it:

```
AttributeError: 'SPPOActorRolloutRefWorker' object has no attribute '_qat_enabled'
```

Fixed by adding the call in `recipe/sppo/sppo_worker.py`. Confirmed inert: the composed
config *does* carry `actor_rollout_ref.actor.qat` with `enable: False`, so
`_init_qat_config()` sets `_qat_enabled = False` and both call sites take the off branch.

**The whole class was swept, not just this instance.** Comparing the two `init_model` bodies
by AST for `self._*()` setup calls and `self.<attr>` assignments:

```
SETUP CALLS in parent but MISSING in sppo:   self._init_qat_config()
ATTRS assigned in parent but NOT in sppo:    self.rank      <- false positive
```

`self.rank` is a property on the base `Worker` (`single_controller/base/worker.py:334`); the
regex matched `self.rank ==` comparisons. So `_init_qat_config` was the only real gap, and
`init_model` is the *only* method SPPO overrides — nothing else in the worker can drift.

Worth re-running that AST comparison after any upstream sync of `fsdp_workers.py`.

## §17. `+` is required for ray_init.address

`ppo_trainer.yaml:461` declares `ray_kwargs.ray_init` with only `num_cpus`, and the config is
a struct, so a bare `ray_kwargs.ray_init.address=...` dies at composition:

```
Could not override 'ray_kwargs.ray_init.address'.  Key 'address' is not in struct
```

`main_sppo.py:79` splats the whole dict into `ray.init(**kwargs)`, so `+`-adding the key
arrives as `ray.init(address=...)` exactly as intended.

To avoid discovering such things one run at a time — hydra reports only the FIRST bad
override — validate the whole set offline before launching. `compose()` the recipe's config
with the runner's overrides, and on failure re-apply them one at a time to name every
offender. Current state: **all 51 overrides compose cleanly.**

## §18. torchao imported before the check that makes it unnecessary

All four sglang schedulers died with `ModuleNotFoundError: No module named 'torchao'` at
`model_runner.py:309 -> torchao_utils.py:53`. torchao is deliberately NOT installed (§5 of
setup: it is a CUDA/Triton library whose import runs a device probe that acquires a card in
processes that must stay CPU-only), and `torchao_config` is `None` here, so the function had
nothing to do. Upstream sglang 0.4.9 simply puts the "lazy import to suppress some warnings"
ABOVE the `if torchao_config == "" or torchao_config is None: return model` early-out.

Fixed in the fork by swapping the order (`torchao_utils.py.orig` is the backup). Note the
first `elif` in the chain below had to become an `if` — moving the guard up orphans it, and
the file will not parse otherwise.

Installing torchao would also "fix" this, and is the wrong fix: the card-grabbing probe is a
worse problem than the missing import.

## §19. HPU rollout card placement was landing on the training cards

`async_sglang_server.py`'s `_IS_HPU_HOST` block computed

```python
train_cards = {int(d) for d in node_cuda_visible_devices.split(",") if d.strip()}
free_cards  = [c for c in range(total_cards) if c not in train_cards]
node_cuda_visible_devices = ",".join(map(str, free_cards[: self.gpus_per_replica_node]))
```

`node_cuda_visible_devices` is a **per-replica slice** of `self.workers` (line 820), so with
TP=1 it names exactly ONE training card. The other three then looked free. Measured, with
4 trainers on cards 0-3:

```
replica_rank=0 ... cuda_visible_devices='1'
replica_rank=2 ... cuda_visible_devices='0'
```

— both straight on top of a trainer. Two independent defects:

1. **`train_cards` under-populated.** Now seeded with `set(range(self.gpus_per_node))` — under
   `RAY_EXPERIMENTAL_NOSET_<visible-devices>` each training rank takes the card matching its
   LOCAL_RANK, so the group occupies `[0, gpus_per_node)`, and `rollout.n_gpus_per_node`
   defaults to `trainer.n_gpus_per_node` (`rollout.yaml:14`). Unioned with the observed slice
   so an explicit-affinity setup still contributes.
2. **All replicas took `free_cards[0]`.** Every replica computes the same free list, so each
   must take a distinct slice: `free_cards[replica_rank * n : ...]`.

Third, latent: `base_gpu_id` was `replica_rank * world_size % gpus_per_node` = 1, 2, 3 for
replicas 1-3, but each server sees only its own card, renumbered from 0. Forced to 0 on HPU.

Verified by simulation before re-running: replicas 0-3 now pin cards 4, 5, 6, 7, base_gpu_id 0.

Watch for four of these in the log, with DISTINCT cards:

```
HPU rollout placement: 8 card(s) total, training holds [0, 1, 2, 3], free [4, 5, 6, 7], replica R takes [R:R+1]
HPU: pinning sglang rollout server to free card(s) <4|5|6|7>
```

Backup: `async_sglang_server.py.orig`.

## §20. SPPO must extend AsyncActorRolloutRefWorker

```
AttributeError: 'RayWorkerGroup' object has no attribute 'update_weights'
  checkpoint_engine/base.py:479 -> ray.get(self.actor_wg.update_weights(...))
```

`update_weights` is defined ONLY on `AsyncActorRolloutRefWorker` (`fsdp_workers.py:1879`), a
subclass of `ActorRolloutRefWorker` adding exactly one `@register`'d method:

```python
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps=None, mode="auto", *args, **kwargs):
        await self.rollout_mode()
        return True
```

`RayWorkerGroup` binds only `@register`'d methods it finds on the worker class, so with the
plain base the call does not exist. The rollout here is server-based (`SGLangHttpServer` +
`LLMServerManager`), and the trainer must push fresh actor weights to those servers each step
— hence `checkpoint_manager.update_weights()` in `fit()`.

Every other server-rollout recipe already does this: `recipe/ioher/ioher_worker.py:32`,
`recipe/entropy`, `recipe/dapo`, `recipe/atropos`. **SPPO is the outlier** — it predates the
sync/async worker split, which is the same staleness class as §16.

Fixed: `class SPPOActorRolloutRefWorker(AsyncActorRolloutRefWorker)`. Safe because the async
class overrides nothing else — in particular not `init_model`, which SPPO overrides — and
`rollout_mode()` does exist on the legacy worker (`fsdp_workers.py:850`).

Note `recipe/ioher` also patches `_qat_enabled` (§16), but with a `hasattr` guard in
`__init__` rather than calling `_init_qat_config()`. Ours reads the real config; both work.

### Startup milestones reached on this run
- rollout cards correct and distinct: `replica_rank=1 -> '5'`, `replica_rank=3 -> '7'` (§19 fix holds)
- all four sglang servers listening: `LLMServerManager: [':38395', ':46689', ':39815', ':43273']`
- W&B live: project `verl_grpo_gsm8k_gaudi`, run `Qwen3-0.6B_gsm8k_grpo_hpu`
- `Training from scratch` — checkpoint load path clean
- entered `fit()`; failed on the pre-step-1 weight push, before any rollout

## §21. SPPO's gen_batch key list assumed a pre-tokenized dataset

```
AssertionError  protocol.py:741  assert key in self.batch.keys()
  sppo_ray_trainer.py:213  gen_batch = batch.pop(...)
```

SPPO hardcoded:

```python
batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
```

Measured against the actual GSM8K parquet through `create_rl_dataset` with this config:

```
TENSOR keys     : ['dummy_tensor']
NON-TENSOR keys : ['ability','data_source','extra_info','index','interaction_kwargs',
                   'prompt','raw_prompt','reward_model','tools_kwargs']
```

**None of the four hardcoded keys exist.** In this snapshot's agent-loop path `RLHFDataset`
does not pre-tokenize — the agent loop tokenizes from `raw_prompt` — so there are no
`input_ids`/`attention_mask`/`position_ids` tensors, and the raw prompt is `raw_prompt`, not
`raw_prompt_ids`. Both halves of the list were wrong.

Fixed by using the builder SPPO already inherits, `RayPPOTrainer._get_gen_batch`
(`ray_trainer.py:572`), which pops NO tensor keys and every non-tensor key except the reward
keys, then re-attaches those so the agent loop can score. Generic over tokenized and
untokenized datasets alike. Also added `gen_batch.meta_info["global_steps"]`, which stock sets
at `ray_trainer.py:1443` and SPPO omitted.

Same staleness class as §16 and §20 — a hand-copied fragment of an older trainer.

### Rest of fit() audited, no further divergence
- `uid` is assigned BEFORE the x-n repeat, so each prompt's n samples share one uid — exactly
  what `compute_grpo_outcome_advantage` groups on.
- repeat/union block matches stock; `union_numpy_dict` (`protocol.py:188`) asserts
  `_deep_equal` on duplicate keys rather than rejecting them, so the shared
  data_source/reward_model/extra_info survive the union.
- `batch.batch["seq_level_rewards"] = token_level_scores` (line 341) is a dead leftover of the
  SPPO loss; nothing reads it under GRPO. Left in place — removing it is cosmetic and carries
  more risk than value right now.

## §22. Why W&B had almost no metrics: two separate bugs

### 22a. SPPO never built the metrics

SPPO's `fit()` logged only:

```python
metrics.update({"training/global_step": ..., "training/epoch": ...})
logger.log(data=metrics, step=self.global_steps)
```

plus whatever `actor/*` keys `update_actor` happened to return. It never called the three
builders that stock `ray_trainer` and every other recipe use
(`recipe/ioher/ioher_ray_trainer.py:992-995`). Added them. New keys:

| builder | keys |
| --- | --- |
| `compute_data_metrics` | `critic/rewards/{mean,max,min}`, `critic/score/*`, `critic/advantages/*`, `critic/returns/*`, `response_length/{mean,max,min,clip_ratio}`, `response_length_non_aborted/*`, `prompt_length/*`, `response/aborted_ratio`, `num_turns/*` |
| `compute_timing_metrics` | `timing_s/*`, `timing_per_token_ms/*` per phase (gen, old_log_prob, adv, update_actor) |
| `compute_throughout_metrics` | `perf/throughput`, `perf/total_num_tokens`, `perf/time_per_step`, `perf/mfu` |

`use_critic=False`, so `critic/values/*` and `critic/vf_explained_var` are skipped; everything
else is emitted regardless of the critic. **`response_length/clip_ratio` is the one to watch**
— near 1.0 means generations are hitting `max_response_length`, the `####` marker is being
truncated, and the GRPO advantage silently collapses to zero within every group.

### 22b. The logging block was OUTSIDE the batch loop

The more serious one, and **original to the recipe** (verified against
`sppo_ray_trainer.py.orig`, identical relative indentation):

```
for epoch in ...:                          # indent 8
    for batch_dict in self.train_dataloader:   # indent 12
        metrics = {}                            # indent 16   <- loop body
        ...
    # training metrics                          # indent 12   <- OUTSIDE the loop
    logger.log(data=metrics, ...)               # indent 12
    progress_bar.update(1)                      # indent 12
    self.global_steps += 1                      # indent 12
```

Consequences:
1. A "training step" meant a whole **epoch**. `total_training_steps=5` would have run 5 full
   epochs — ~2335 rollout+train iterations — not 5 batches.
2. Exactly **one W&B point per epoch**.
3. `global_steps` is constant across an epoch, so `global_steps % test_freq == 0` is either
   false for the entire epoch or true after **every batch** in it — validating and
   checkpointing on each one.

Re-indented 403-437 into the batch loop. `total_training_steps=5` now means 5 batches, and
W&B gets one point per step as intended.

## §23. Final config: thinking on, 2048 responses, FusedSDPA

| knob | was | now | why |
| --- | --- | --- | --- |
| attention | `eager` | `sdpa` + FusedSDPA | `VERL_HPU_FUSED_SDPA` was 0, so the script chose `eager`, which materializes the full `[B,H,S,S]` score matrix per layer. At 2560 padded tokens that is 1.26 GB/layer at micro_batch=2 plus an fp32 softmax copy. FusedSDPA tiles like flash-attention and never builds it. |
| `MAX_RESPONSE_LENGTH` | 512 | 2048 | Qwen3 thinking mode is left ON; the template does not pre-close `<think>`, so the model reasons before answering and blows past GSM8K's 292-token p99 reference solution. |
| `VERL_HPU_FUSED_SDPA` | 0 | 1 (gaudi_env.sh) | prerequisite for the above; confirmed live by the startup line `VERL HPU: F.scaled_dot_product_attention routed to Habana FusedSDPA`. |

Risk carried knowingly: stock torch SDPA lowering NaNs on Gaudi, which is why the patch
routes to FusedSDPA explicitly with a per-call fallback. **Watch `actor/grad_norm` on the
first steps** — NaN there means back out with `VERL_HPU_FUSED_SDPA=0` (and drop the response
length back, since eager at 2048 is ~3x the cost of eager at 1024).

## §24. The 30-minute hang: my sglang bucket "tuning" was wrong

Symptom: `hl-smi` showed all 8 cards at **0% AIP-Util** — training holding 289 MiB/card,
sglang holding 48.7 GB/card (= gpu_memory_utilization 0.5 x 98 GB, correct) — and the log
silent for 30+ minutes after the agent-loop workers came up.

Diagnosis path (py-spy from the HOST; `ray stack` insists on sudo, but ptrace_scope=0 makes
`py-spy dump --pid <pid>` work directly):

| process | state |
| --- | --- |
| `TaskRunner` (driver) | blocked in `generate_sequences`, `sppo_ray_trainer.py:236` |
| `WorkerDict` x4 | ~33 CPU-min burned in native threads that have since EXITED; now idle |
| `AgentLoopWorker` x8 | 13 s CPU each -- startup only, then nothing |
| `SGLangHttpServer` x4 | 13 s CPU each -- startup only, then nothing |

Note `ps aux` %CPU is CUMULATIVE (cpu-time / elapsed), so the "98.4%" on WorkerDict was
history, not a live spin. `ps -L -o tid,pcpu` showed every surviving thread near zero. The
33 minutes was model load + FSDP init + `update_weights` serializing weights to the four
servers -- all of which COMPLETED, since the driver is past it and inside generation.

**Root cause, self-inflicted.** `env/run_grpo_gsm8k.sh` set

```
SGLANG_HPU_PREFILL_BUCKET_MIN=128  STEP=128  MAX=${MAX_PROMPT_LENGTH}   # 512
```

on the assumption that the prefill bucket is a per-sequence length cap. It is not.
`hpu_graph_runner.py:165` calls `get_prefill_seq_len_bucket(sum(prompt_lens))` -- a token
budget for the ENTIRE prefill batch. With ~32 concurrent prompts of ~75 tokens the sum is
~2400, so `find_bucket(2400, (128,128,512))` returns 2432: past the ceiling, and a bucket no
graph was ever captured for. Generation hung with no error and no device activity.

The fork's default `PREFILL_BUCKET_MAX=6144` exists precisely because it is a batch budget.

Compounding it: `SGLANG_HPU_SKIP_WARMUP=true` (also mine, for "fast bring-up") removed the
startup pass that captures a graph per bucket -- i.e. the exact thing that would have failed
loudly at startup instead of hanging at the first request.

**Fix: all SGLANG_HPU_* overrides removed; the fork's defaults now stand.** Lesson recorded in
the script: these knobs are a batch budget, not a sequence length, and warmup is what makes a
bad bucket config visible. Change one at a time and confirm generation still completes.
