#!/usr/bin/env bash
# Prepare the pinned Miles SGLang runtime and an isolated VERL venv.
# No existing environment or Miles checkout is deleted or reset.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/gaudi_env.sh"

if [[ "$VENV_DIR" == "$SCRATCH_ROOT/venvs/verl-gaudi" || "$SGLANG_HPU_ROOT" == "$SCRATCH_ROOT/sglang-fork" ]]; then
    echo "Legacy paths are exported. Run: unset VENV_DIR SGLANG_HPU_ROOT; then rerun setup." >&2
    exit 1
fi

if [[ -z "${VERL_SETUP_INNER:-}" ]]; then
    exec env VERL_SETUP_INNER=1 bash "$HERE/shell.sh" bash "$HERE/setup_uv_env.sh"
fi

mkdir -p "$REPO_ROOT/.runtime"
exec 9>"$REPO_ROOT/.runtime/setup.lock"
flock -n 9 || { echo "Another runtime setup is active." >&2; exit 1; }
# Applied in order: the Miles Gaudi work, the container-only Torch 2.7 fix, then
# this repo's runtime fixes discovered by the VERL GRPO integration tests.
PATCHES=(
    "$HERE/patches/sglang-miles-cb05a44-gaudi.patch"
    "$HERE/patches/sglang-cb05a44-hpu-container-compat.patch"
    "$HERE/patches/sglang-cb05a44-verl-hpu-runtime.patch"
)
if [[ ! -d "$SGLANG_HPU_ROOT/.git" ]]; then
    [[ ! -e "$SGLANG_HPU_ROOT" ]] || { echo "Refusing non-Git source: $SGLANG_HPU_ROOT" >&2; exit 1; }
    SOURCE="${SGLANG_CLONE_SOURCE:-https://github.com/sgl-project/sglang.git}"
    if [[ -d "$SCRATCH_ROOT/sglang-miles/.git" && -z "${SGLANG_CLONE_SOURCE:-}" ]]; then
        SOURCE="$SCRATCH_ROOT/sglang-miles"
    fi
    git clone --no-hardlinks --no-checkout "$SOURCE" "$SGLANG_HPU_ROOT"
    git -C "$SGLANG_HPU_ROOT" checkout --detach "$SGLANG_HPU_COMMIT"
fi
actual="$(git -C "$SGLANG_HPU_ROOT" rev-parse HEAD)"
[[ "$actual" == "$SGLANG_HPU_COMMIT" ]] || {
    echo "Wrong SGLang commit: $actual (expected $SGLANG_HPU_COMMIT)." >&2
    exit 1
}
for PATCH in "${PATCHES[@]}"; do
    name="$(basename "$PATCH")"
    if git -C "$SGLANG_HPU_ROOT" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
        echo "Already applied: $name"
    elif git -C "$SGLANG_HPU_ROOT" apply --check "$PATCH" >/dev/null 2>&1; then
        git -C "$SGLANG_HPU_ROOT" apply "$PATCH"
        echo "Applied: $name"
    else
        echo "$name does not match this checkout; preserving existing changes." >&2
        exit 1
    fi
done

# uv is used only with --no-deps. Never resolve the upstream CUDA/Torch dependencies.
UV="${UV_BIN:-$(command -v uv)}"
SYS_PY=/usr/bin/python3.12
"$SYS_PY" -c 'import torch; assert "+hpu" in torch.__version__, torch.__version__'
if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
    [[ ! -e "$VENV_DIR" ]] || { echo "Refusing non-venv directory: $VENV_DIR" >&2; exit 1; }
    "$UV" venv --system-site-packages --python "$SYS_PY" "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
"$UV" pip install --python "$PY" --no-deps -r "$HERE/requirements-sglang-miles.txt"
SGLANG_BUILD_RUST_EXTS=none "$UV" pip install --python "$PY" --no-deps -e "$SGLANG_HPU_ROOT/python"
VENV_SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
HF_PATCH="$HERE/patches/transformers-5.12-hpu-annotations.patch"
if patch --dry-run -R -p1 -d "$VENV_SP" < "$HF_PATCH" >/dev/null 2>&1; then
    echo "Transformers HPU annotation patch already applied."
else
    patch --batch --forward -p1 -d "$VENV_SP" < "$HF_PATCH"
fi
"$PY" "$HERE/verify_sglang_miles.py"
echo "Runtime setup verified: $VENV_DIR"
