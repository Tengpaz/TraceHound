#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOTE_TARGET="${TRACEHOUND_REMOTE_TARGET:-ailab-p:/mnt/petrelfs/lichunxiao/TraceHound}"
SYNC_ENV="${TRACEHOUND_SYNC_ENV:-1}"
BOOTSTRAP="${TRACEHOUND_REMOTE_BOOTSTRAP:-0}"
RUN_OFFICIAL_SMOKE="${TRACEHOUND_REMOTE_OFFICIAL_SMOKE:-0}"
RSYNC_DELETE="${TRACEHOUND_RSYNC_DELETE:-0}"
SSH_OPTS="${TRACEHOUND_SSH_OPTS:--o ConnectTimeout=20}"

REMOTE_HOST="${REMOTE_TARGET%%:*}"
REMOTE_PATH="${REMOTE_TARGET#*:}"
if [[ "$REMOTE_HOST" == "$REMOTE_PATH" || -z "$REMOTE_HOST" || -z "$REMOTE_PATH" ]]; then
  echo "[tracehound] TRACEHOUND_REMOTE_TARGET must look like host:/absolute/path" >&2
  exit 1
fi

RSYNC_ARGS=(-az --human-readable --itemize-changes -e "ssh ${SSH_OPTS}")
if [[ "$RSYNC_DELETE" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi
RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '.env'
  --exclude '.env.*'
  --exclude '!.env.example'
  --exclude '!.env.server.example'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.ruff_cache/'
  --exclude '*.egg-info/'
  --exclude 'data/tmp/'
  --exclude 'data/raw/'
  --exclude 'data/processed/'
  --exclude 'data/sft/'
  --exclude 'data/metadata/'
  --exclude 'data/rejected/'
  --exclude 'external/agentdog_official/datasets/'
  --exclude 'reports/*'
  --exclude '!reports/.gitkeep'
  --exclude 'checkpoints/'
  --exclude 'models/'
  --exclude 'outputs/'
  --exclude 'runs/'
  --exclude 'logs/'
  --exclude 'dist/'
)

echo "[tracehound] ensuring remote directory: ${REMOTE_TARGET}"
# shellcheck disable=SC2086
ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH'"

echo "[tracehound] rsync project to ${REMOTE_TARGET}"
rsync "${RSYNC_ARGS[@]}" "${RSYNC_EXCLUDES[@]}" ./ "${REMOTE_TARGET}/"

if [[ "$SYNC_ENV" == "1" && -f .env ]]; then
  echo "[tracehound] syncing .env securely without printing secrets"
  # shellcheck disable=SC2086
  scp $SSH_OPTS -q .env "${REMOTE_TARGET}/.env"
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "$REMOTE_HOST" "chmod 600 '$REMOTE_PATH/.env'"
elif [[ "$SYNC_ENV" == "1" ]]; then
  echo "[tracehound] no local .env found; remote .env was not changed"
fi

if [[ "$BOOTSTRAP" == "1" ]]; then
  echo "[tracehound] running remote bootstrap"
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "$REMOTE_HOST" "cd '$REMOTE_PATH' && bash scripts/bootstrap_remote.sh"
fi

if [[ "$RUN_OFFICIAL_SMOKE" == "1" ]]; then
  echo "[tracehound] running remote official data smoke"
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "$REMOTE_HOST" "cd '$REMOTE_PATH' && conda run -n tracehound-gpu python scripts/prepare_agentdog_official.py --download-dataset agentdog10_training --manifest reports/agentdog_official_manifest.json"
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "$REMOTE_HOST" "cd '$REMOTE_PATH' && conda run -n tracehound-gpu python scripts/build_agentdog_data.py --source agentdog10 --limit 20 --no-annotate-cot"
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "$REMOTE_HOST" "cd '$REMOTE_PATH' && conda run -n tracehound-gpu python scripts/build_agentdog_data.py --source agentdog10 --limit 2 --annotate-cot --cot-backend stub --cot-concurrency 1"
fi

echo "[tracehound] remote sync complete: ${REMOTE_TARGET}"
