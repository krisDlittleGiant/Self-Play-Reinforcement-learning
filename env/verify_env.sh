#!/usr/bin/env bash
# Verify the Gaudi GRPO environment. Run from the host; re-execs into the container.
#
#   bash env/verify_env.sh
#
# Every check is independent and prints PASS / FAIL / WARN. Exit code 0 only if no FAIL.
# FAILs are ordered so the first one you hit is the one to fix.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/gaudi_env.sh"

if [ -z "${VERL_VERIFY_INNER:-}" ]; then
    exec env VERL_VERIFY_INNER=1 bash "$HERE/shell.sh" bash "$HERE/verify_env.sh"
fi

FAILS=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILS=$((FAILS+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$1"; }

PY="$VENV_DIR/bin/python"

hdr "1. Container and venv"
[ -f "$GAUDI_SIF" ] && pass "image $GAUDI_SIF" || fail "image missing: $GAUDI_SIF"
[ -x "$PY" ] && pass "venv python $($PY -V 2>&1)" || { fail "venv python missing: $PY"; echo; echo "Run: bash env/setup_uv_env.sh"; exit 1; }

hdr "2. torch is the Habana build, not a PyPI CUDA wheel"
TV="$($PY -c 'import torch;print(torch.__version__)' 2>&1)"
TF="$($PY -c 'import torch;print(torch.__file__)' 2>&1)"
case "$TV" in
    *"+cu"*|*"+rocm"*) fail "torch is $TV  <-- CUDA/ROCm wheel shadowing Habana torch" ;;
    *)                 pass "torch $TV" ;;
esac
case "$TF" in
    "$VENV_DIR"/*) fail "torch lives INSIDE the venv ($TF) -- a wheel was installed over the Habana build" ;;
    *)             pass "torch from container site-packages" ;;
esac
VSP="$(echo "$VENV_DIR"/lib/python*/site-packages)"
[ -d "$VSP/torch" ] && fail "$VSP/torch exists -- delete the venv and re-run setup" || pass "no torch directory in the venv"

hdr "3. HPUs visible"
$PY - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import habana_frameworks.torch.hpu as h
    ok, n = h.is_available(), h.device_count()
    print(("PASS  " if ok else "FAIL  ") + f"habana_frameworks: available={ok} device_count={n}")
    if n != 8: print(f"WARN  expected 8 cards, saw {n}")
except Exception as e:
    print(f"FAIL  habana_frameworks import: {e}")
PY
$PY -c 'import habana_frameworks.torch.hpu as h; raise SystemExit(0 if h.is_available() else 1)' 2>/dev/null || FAILS=$((FAILS+1))

hdr "4. hl-smi agrees"
if command -v hl-smi >/dev/null; then
    N=$(hl-smi -Q index -f csv,noheader 2>/dev/null | wc -l)
    [ "$N" -gt 0 ] && pass "hl-smi reports $N card(s)" || fail "hl-smi returned no cards"
    [ "$N" != "${HPU_CARDS_COUNT}" ] && warn "HPU_CARDS_COUNT=$HPU_CARDS_COUNT but hl-smi sees $N (hl-smi shows the whole node, set HPU_CARDS_COUNT to your allocation)"
else
    fail "hl-smi not found"
fi

hdr "5. A real tensor op on the HPU"
$PY - <<'PY' 2>&1 | sed 's/^/  /'
import torch
try:
    a = torch.randn(512, 512, device="hpu"); b = torch.randn(512, 512, device="hpu")
    c = (a @ b).sum().item()
    print(f"PASS  matmul on hpu ran, sum={c:.3f}")
except Exception as e:
    print(f"FAIL  hpu matmul: {type(e).__name__}: {e}")
PY

hdr "6. GPU Migration Toolkit active (torch.cuda -> hpu)"
$PY - <<'PY' 2>&1 | sed 's/^/  /'
import os, torch
mig = os.environ.get("PT_HPU_GPU_MIGRATION")
print(("PASS  " if mig == "1" else "FAIL  ") + f"PT_HPU_GPU_MIGRATION={mig}")
print(("PASS  " if torch.cuda.is_available() else "FAIL  ") + f"torch.cuda.is_available()={torch.cuda.is_available()} (True is CORRECT here - migration maps cuda->hpu)")
print(f"INFO  torch.cuda.device_count()={torch.cuda.device_count()}")
PY

hdr "7. verl imports from THIS tree and picks the HPU platform"
$PY - <<'PY' 2>&1 | sed 's/^/  /'
import os, sys
want = os.environ["VERL_COMPAT"]
import verl
got = os.path.dirname(os.path.dirname(verl.__file__))
print(("PASS  " if os.path.realpath(got) == os.path.realpath(want) else "FAIL  ") + f"verl from {verl.__file__}")
from verl.plugin.platform import get_platform
p = get_platform()
print(("PASS  " if p.vendor_name == "intel" else "FAIL  ") + f"platform vendor={p.vendor_name} device_name={p.device_name} backend={p.communication_backend_name()} ray_resource={p.ray_resource_name()}")
from verl.utils.device import get_vendor, get_device_name
print(f"INFO  get_vendor()={get_vendor()}  get_device_name()={get_device_name()}  (device_name 'cuda' is intentional)")
PY

hdr "8. Pinned Miles SGLang and VERL integration"
"$PY" "$HERE/verify_sglang_miles.py" || fail "Miles SGLang preflight"

hdr "9. sgl_kernel stub resolves (CUDA kernels have no Gaudi build)"
$PY - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import sgl_kernel, os
    where = getattr(sgl_kernel, "__file__", "?")
    is_stub = "verl_compat" in str(where)
    print(("PASS  " if is_stub else "WARN  ") + f"sgl_kernel -> {where}")
    f = sgl_kernel.some_missing_kernel          # must not raise on attribute access
    try:
        f()
        print("FAIL  stub did not raise when a kernel was actually called")
    except RuntimeError as e:
        print("PASS  calling a stubbed kernel raises a clear error")
except Exception as e:
    print(f"FAIL  sgl_kernel: {type(e).__name__}: {e}")
PY

hdr "10. Other required packages"
$PY - <<'PY' 2>&1 | sed 's/^/  /'
import importlib.metadata as md
want = {"ray":"2.53", "transformers":"5.12", "tensordict":None, "datasets":None,
        "hydra-core":None, "wandb":None, "accelerate":None, "peft":None,
        "pyarrow":None, "omegaconf":None, "codetiming":None, "torchdata":None}
for p, pin in want.items():
    try:
        v = md.version(p)
        bad = pin and not v.startswith(pin)
        print(("WARN  " if bad else "PASS  ") + f"{p} {v}" + (f"  (expected ~{pin})" if bad else ""))
    except Exception:
        print(f"FAIL  {p} MISSING")
# torchao must NOT be installed - it grabs a card via a Triton probe in CPU-only processes
try:
    md.version("torchao"); print("WARN  torchao is installed; verl will disable it, but cleaner to remove")
except Exception:
    print("PASS  torchao not installed")
PY

hdr "11. Caches all land on /scratch (not /home)"
for v in HF_HOME HF_DATASETS_CACHE XDG_CACHE_HOME TMPDIR TORCHINDUCTOR_CACHE_DIR \
         TRITON_CACHE_DIR TORCH_HOME WANDB_DIR UV_CACHE_DIR PIP_CACHE_DIR HABANA_LOGS; do
    val="${!v:-}"
    if [ -z "$val" ]; then fail "$v unset"
    elif [[ "$val" == /home/* || "$val" == "$HOME"/* ]]; then fail "$v -> $val  (on /home!)"
    elif [[ "$val" == /scratch/* ]]; then pass "$v -> $val"
    else warn "$v -> $val  (not under /scratch)"; fi
done
if [[ -v GRAPH_VISUALIZATION_DIR ]]; then
    fail "GRAPH_VISUALIZATION_DIR is set -> $GRAPH_VISUALIZATION_DIR (graph dumps enabled)"
else
    pass "GRAPH_VISUALIZATION_DIR unset (graph dumps disabled)"
fi
RC="${PT_HPU_RECIPE_CACHE_CONFIG%%,*}"
[[ "$RC" == /scratch/* ]] && pass "PT_HPU_RECIPE_CACHE_CONFIG -> $PT_HPU_RECIPE_CACHE_CONFIG" || fail "PT_HPU_RECIPE_CACHE_CONFIG -> ${PT_HPU_RECIPE_CACHE_CONFIG:-unset}"
[[ "$RAY_TMPDIR" == /dev/shm/* ]] && pass "RAY_TMPDIR -> $RAY_TMPDIR (tmpfs)" || warn "RAY_TMPDIR -> $RAY_TMPDIR (should be tmpfs, not BeeGFS)"
HOME_FREE=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
[ -n "$HOME_FREE" ] && { [ "$HOME_FREE" -lt 10 ] && warn "/home has only ${HOME_FREE}G free - keep everything off it" || pass "/home has ${HOME_FREE}G free"; }

hdr "12. GSM8K data in verl's schema"
if [ -f "$GSM8K_DIR/train.parquet" ]; then
    $PY - <<PY 2>&1 | sed 's/^/  /'
import pyarrow.parquet as pq
t = pq.read_table("$GSM8K_DIR/train.parquet")
cols = set(t.column_names)
need = {"data_source","prompt","reward_model","extra_info"}
missing = need - cols
print(("PASS  " if not missing else "FAIL  ") + f"train.parquet {t.num_rows} rows, cols={sorted(cols)}")
if missing: print(f"FAIL  missing {sorted(missing)} - regenerate with examples/data_preprocess/gsm8k.py")
ds = t.column("data_source")[0].as_py() if "data_source" in cols else None
print(("PASS  " if ds == "openai/gsm8k" else "FAIL  ") + f"data_source={ds!r} (must be 'openai/gsm8k' to route the reward fn)")
PY
else
    warn "no $GSM8K_DIR/train.parquet yet -- generate with:"
    echo "        python examples/data_preprocess/gsm8k.py --local_save_dir $GSM8K_DIR"
fi

hdr "13. W&B reachable and authenticated"
if [ -n "${WANDB_API_KEY:-}" ]; then pass "WANDB_API_KEY set"
elif [ -f "$HOME/.netrc" ] && grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    warn "auth via ~/.netrc -- works only while \$HOME is bound into the container; WANDB_API_KEY is safer"
else
    warn "no W&B credentials found (run 'wandb login' or export WANDB_API_KEY)"
fi

echo
if [ "$FAILS" -eq 0 ]; then printf '\033[32mAll checks passed.\033[0m\n'; else printf '\033[31m%d check(s) FAILED.\033[0m\n' "$FAILS"; fi
exit $(( FAILS > 0 ))
