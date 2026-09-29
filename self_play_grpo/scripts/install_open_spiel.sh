#!/usr/bin/env bash
# Build the pinned Quoridor engine without changing the Gaudi Torch stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
SOURCE_DIR="$REPO_DIR/.runtime/open_spiel-2.0.2-d0606878"
DOWNLOAD_DIR="$REPO_DIR/.runtime/open_spiel-download"
ARCHIVE="$DOWNLOAD_DIR/open_spiel-2.0.2.tar.gz"
ARCHIVE_URL="https://files.pythonhosted.org/packages/bb/3b/9d56d511eff27e26714aa232c945bb00335f792bc1959637ee0d2639a018/open_spiel-2.0.2.tar.gz"
ARCHIVE_SHA256="6f520dfe499b9e5e1e3a31cf77bf071abb02b98104789e7698021922a064e856"
PATCH_FILE="$PROJECT_DIR/patches/open_spiel-2.0.2-d0606878.patch"
PATCH_STAMP="$SOURCE_DIR/.self-play-grpo-patch-d0606878"

if [[ -z "${SELF_PLAY_OPEN_SPIEL_INNER:-}" ]]; then
  exec bash "$REPO_DIR/env/shell.sh" env \
    SELF_PLAY_OPEN_SPIEL_INNER=1 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    OPEN_SPIEL_BUILD_JOBS="${OPEN_SPIEL_BUILD_JOBS:-8}" \
    bash "$SCRIPT_DIR/install_open_spiel.sh"
fi

mkdir -p "$DOWNLOAD_DIR"
if [[ ! -f "$ARCHIVE" ]]; then
  curl \
    --fail \
    --location \
    --retry 3 \
    --retry-all-errors \
    --output "$ARCHIVE.part" \
    "$ARCHIVE_URL"
  printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE.part" | sha256sum --check -
  mv "$ARCHIVE.part" "$ARCHIVE"
fi
printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE" | sha256sum --check -

mkdir -p "$SOURCE_DIR"
if [[ ! -f "$PATCH_STAMP" ]]; then
  # Restore pristine files after an interrupted or partially applied patch.
  tar -xzf "$ARCHIVE" -C "$SOURCE_DIR" --strip-components=1
  patch --batch --forward -p1 -d "$SOURCE_DIR" < "$PATCH_FILE"
  touch "$PATCH_STAMP"
fi

if [[ ! -d "$SOURCE_DIR/open_spiel/abseil-cpp" ]] || \
   [[ ! -d "$SOURCE_DIR/open_spiel/json" ]]; then
  echo "ERROR: OpenSpiel source archive did not include required C++ dependencies." >&2
  exit 1
fi

if ! grep -Fq 'moves.push_back(base_for_relative_.xy);' \
    "$SOURCE_DIR/open_spiel/games/quoridor/quoridor.cc" || \
   ! grep -Fq 'OPEN_SPIEL_BUILD_JOBS' "$SOURCE_DIR/setup.py"; then
  echo "ERROR: pinned OpenSpiel patch is not applied cleanly in $SOURCE_DIR" >&2
  exit 1
fi

python -m pip install --no-deps --no-build-isolation "$SOURCE_DIR"
python - <<'PY'
from importlib.metadata import version
import pyspiel

actual = version("open_spiel")
if actual != "2.0.2":
    raise SystemExit(f"unexpected open_spiel version: {actual}")
if "quoridor" not in pyspiel.registered_names():
    raise SystemExit("pinned build does not register quoridor")
print(f"Pinned OpenSpiel ready: {actual} with Quoridor d0606878 backport")
PY
