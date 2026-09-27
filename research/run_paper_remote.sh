#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/paper_runs logs
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
out="results/paper_runs/${run_id}.json"
.venv/bin/python research/run_paper_28c_pit_30m.py --out "$out"
ln -sfn "paper_runs/$(basename "$out")" results/paper_latest.json
