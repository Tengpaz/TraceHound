#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REF="${1:-HEAD}"
SHORT_SHA="$(git rev-parse --short "$REF")"
OUTPUT_DIR="${TRACEHOUND_BUNDLE_DIR:-dist}"
OUTPUT="${OUTPUT_DIR}/tracehound-${SHORT_SHA}.tar.gz"

mkdir -p "$OUTPUT_DIR"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[tracehound] warning: working tree has uncommitted changes; bundle uses committed ref ${REF}" >&2
fi

git archive --format=tar.gz --prefix="TraceHound/" "$REF" -o "$OUTPUT"

echo "$OUTPUT"
echo "[tracehound] upload example:"
echo "scp $OUTPUT <user>@<server-ip>:~/"
echo "[tracehound] remote unpack example:"
echo "tar -xzf $(basename "$OUTPUT") && cd TraceHound && cp .env.server.example .env && bash scripts/bootstrap_remote.sh"
