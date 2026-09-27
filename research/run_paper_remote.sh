#!/usr/bin/env bash
# Paper-only, broker-free diagnostics. Never places a live order.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/paper_runs logs

# Best-effort code sync before the run.
#
# A failed pull must NOT abort the paper job: the diagnostics still have value
# on the currently-deployed code, and losing an hourly run because GitHub was
# briefly unreachable is worse than running one cycle slightly behind main.
# Untracked runtime state (.env, results/, logs/, data/) is gitignored and is
# never touched by a pull.
if git rev-parse --git-dir >/dev/null 2>&1; then
  if git fetch --quiet origin main >/dev/null 2>&1 && \
     git merge --quiet --ff-only FETCH_HEAD >/dev/null 2>&1; then
    echo "[sync] updated to $(git rev-parse --short HEAD)"
  else
    echo "[sync] skipped (offline, diverged, or local-only commits); running on $(git rev-parse --short HEAD)"
  fi
else
  echo "[sync] not a git checkout; running without update"
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
out="results/paper_runs/${run_id}.json"
.venv/bin/python research/run_paper_28c_pit_30m.py --out "$out"
ln -sfn "paper_runs/$(basename "$out")" results/paper_latest.json
