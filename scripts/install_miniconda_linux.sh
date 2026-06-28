#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${TRACEHOUND_MINICONDA_DIR:-$HOME/miniconda3}"
INSTALLER="${TMPDIR:-/tmp}/miniconda-tracehound.sh"
URL="${TRACEHOUND_MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

if [[ -x "$INSTALL_DIR/bin/conda" ]]; then
  echo "[tracehound] Miniconda already exists: $INSTALL_DIR"
  echo "$INSTALL_DIR/bin/conda"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download Miniconda. Install curl or install conda manually." >&2
  exit 1
fi

echo "[tracehound] downloading Miniconda installer"
curl -fsSL "$URL" -o "$INSTALLER"

echo "[tracehound] installing Miniconda to $INSTALL_DIR"
bash "$INSTALLER" -b -p "$INSTALL_DIR"
rm -f "$INSTALLER"

echo "[tracehound] installed conda: $INSTALL_DIR/bin/conda"
echo "$INSTALL_DIR/bin/conda"
