#!/usr/bin/env bash
# One-shot setup: clean checkout -> runnable GRPO training.
#
#   bash env/bootstrap.sh            # set everything up, then print the run command
#   bash env/bootstrap.sh --run      # ...and launch the full GRPO run at the end
#
# Idempotent: every stage detects existing state and skips. Safe to re-run after a
# partial failure. Nothing is deleted.
#
# Paths all derive from $USER via env/gaudi_env.sh (VERL_USER -> SCRATCH_ROOT -> the
# rest), so this works for any user on the cluster with no edits. Override any of
# SCRATCH_ROOT / REPO_ROOT / VENV_DIR / GAUDI_SIF / GSM8K_DIR in the environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_AFTER=0
[ "${1:-}" = "--run" ] && RUN_AFTER=1

# shellcheck source=/dev/null
source "$HERE/gaudi_env.sh"

step() { printf '\n=== %s ===\n' "$1"; }

step "1/5  environment"
echo "  user        : $VERL_USER"
echo "  repo        : $REPO_ROOT"
echo "  venv        : $VENV_DIR"
echo "  sglang fork : $SGLANG_HPU_ROOT @ $SGLANG_HPU_COMMIT"
echo "  container   : $GAUDI_SIF"
echo "  dataset     : $GSM8K_DIR"
command -v apptainer >/dev/null || { echo "ERROR: apptainer not on PATH." >&2; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv not on PATH (install: https://astral.sh/uv)." >&2; exit 1; }

step "2/5  container image"
if [ -f "$GAUDI_SIF" ]; then
    echo "  present ($(du -h "$GAUDI_SIF" | cut -f1))"
else
    echo "  pulling -- this is ~1.6 GB and takes a few minutes"
    mkdir -p "$(dirname "$GAUDI_SIF")"
    apptainer pull "$GAUDI_SIF" \
      docker://vault.habana.ai/gaudi-docker/1.22.2/ubuntu24.04/habanalabs/pytorch-installer-2.7.1:1.22.2-32
fi

step "3/5  venv + patched SGLang fork"
# setup_uv_env.sh re-execs itself inside the container, clones the pinned SGLang commit,
# applies env/patches/*.patch in order, installs the venv --no-deps (never resolving
# upstream's CUDA torch), patches transformers, and runs verify_sglang_miles.py.
bash "$HERE/setup_uv_env.sh"

step "4/5  GSM8K dataset"
if [ -f "$GSM8K_DIR/train.parquet" ] && [ -f "$GSM8K_DIR/test.parquet" ]; then
    echo "  present ($GSM8K_DIR)"
else
    mkdir -p "$GSM8K_DIR"
    bash "$HERE/shell.sh" bash -c \
      'cd "$VERL_COMPAT" && python3 examples/data_preprocess/gsm8k.py --local_save_dir "$GSM8K_DIR"'
    echo "  written to $GSM8K_DIR"
fi

step "5/5  verify"
bash "$HERE/shell.sh" bash -c '"$VENV_DIR/bin/python" "$REPO_ROOT/env/verify_sglang_miles.py" --source-only'
hl-smi -Q index,memory.used -f csv 2>/dev/null | head -9 || echo "  (hl-smi unavailable -- not on a Gaudi node?)"

cat <<EOF

=== setup complete ===

Start the validated full GRPO run (Qwen3-4B-Base, GSM8K, 233 steps):

    bash env/run_grpo_full.sh

Or a 5-step smoke test first:

    TOTAL_TRAINING_STEPS=5 SAVE_FREQ=-1 TEST_FREQ=-1 WANDB=0 bash env/run_grpo_full.sh
EOF

if [ "$RUN_AFTER" = "1" ]; then
    step "launching GRPO"
    exec bash "$HERE/run_grpo_full.sh"
fi
