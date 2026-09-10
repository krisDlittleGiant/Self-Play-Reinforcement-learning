#!/usr/bin/env bash
# Mirror the Python import trees onto node-local NVMe.
#
#   eval "$(bash env/sync_local.sh)"                # sync, then export local paths
#   eval "$(bash env/sync_local.sh --exports-only)" # reuse an existing local mirror
#
# Imports are thousands of small stat()s and reads. On /scratch (BeeGFS) each one is a
# network round trip, paid again by every one of the ~20 Ray processes. /tmp here is
# /dev/nvme0n1p2, node-local. gaudi_env.sh takes all of these as ${VAR:-default}, and
# env/shell.sh already forwards them into the container, so nothing in the repo changes.
#
# Node-local means per-node: rsync makes a warm re-sync cheap when you move machines.
# Progress goes to stderr, exports to stdout, so the eval form stays clean.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/gaudi_env.sh" >/dev/null 2>&1

# Always copy from this checkout. gaudi_env.sh intentionally preserves pre-existing path
# overrides, so after a previous `eval "$(sync_local.sh)"` the shell has VENV_DIR,
# SGLANG_HPU_ROOT, and VERL_COMPAT pointing at /tmp. Reusing those variables as rsync
# sources makes the next refresh silently copy each local mirror onto itself.
CANONICAL_REPO_ROOT="$(cd "$HERE/.." && pwd)"
CANONICAL_VENV_DIR="$CANONICAL_REPO_ROOT/.runtime/venv"
CANONICAL_SGLANG_ROOT="$CANONICAL_REPO_ROOT/.runtime/sglang-miles"
CANONICAL_VERL_COMPAT="$CANONICAL_REPO_ROOT/verl_compat"

LOCAL_ROOT="${VERL_LOCAL_ROOT:-/tmp/verl-local-${VERL_USER}}"
mkdir -p "$LOCAL_ROOT"

EXPORTS_ONLY=0
case "${1:-}" in
    "") ;;
    --exports-only) EXPORTS_ONLY=1 ;;
    *) echo "usage: $0 [--exports-only]" >&2; exit 2 ;;
esac

sync_tree() {  # <src> <name>
    if [[ "$(readlink -f "$1")" == "$(readlink -f "$LOCAL_ROOT/$2")" ]]; then
        echo "ERROR: refusing to sync $2 from the destination onto itself" >&2
        exit 2
    fi
    echo "  syncing $2 -> $LOCAL_ROOT/$2 ..." >&2
    # Show aggregate byte/file progress.  A BeeGFS read can otherwise leave this command
    # completely silent for many minutes and look indistinguishable from a dead process.
    rsync -a --delete --info=progress2,stats1 "$1/" "$LOCAL_ROOT/$2/" >&2
}
if (( EXPORTS_ONLY )); then
    for component in venv sglang-miles verl_compat; do
        [[ -d "$LOCAL_ROOT/$component" ]] || {
            echo "ERROR: missing $LOCAL_ROOT/$component; run a full sync first" >&2
            exit 2
        }
    done
    echo "  reusing existing local mirror: $LOCAL_ROOT" >&2
else
    sync_tree "$CANONICAL_VENV_DIR"    venv
    sync_tree "$CANONICAL_SGLANG_ROOT" sglang-miles
    sync_tree "$CANONICAL_VERL_COMPAT" verl_compat

    # A relocated venv keeps absolute shebangs in its console scripts. Everything here is
    # invoked as `python -m`, but rewriting them keeps `bin/ray`, `bin/wandb` etc. usable.
    grep -rlZ "^#!${CANONICAL_VENV_DIR}/bin/" "$LOCAL_ROOT/venv/bin" 2>/dev/null \
      | xargs -0 -r sed -i "1s|^#!${CANONICAL_VENV_DIR}/bin/|#!${LOCAL_ROOT}/venv/bin/|"

    echo "  done: $LOCAL_ROOT ($(du -sh "$LOCAL_ROOT" | cut -f1))" >&2
fi
cat <<EXPORTS
export VENV_DIR=$LOCAL_ROOT/venv
export SGLANG_HPU_ROOT=$LOCAL_ROOT/sglang-miles
export VERL_COMPAT=$LOCAL_ROOT/verl_compat
export APPTAINER_BINDS=/scratch/${VERL_USER},/dev/shm,/tmp
EXPORTS
