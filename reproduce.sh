#!/usr/bin/env bash
# Full reproduction of the paper from scratch, logged to logs/reproduce_<date>.log.
# Resumable: re-running it skips every stage whose output already exists.
#   ./reproduce.sh              everything (GPU, ~11 h on one RTX A4500)
#   ./reproduce.sh --smoke      the pipeline on a 3,000-sample subset (minutes)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
LOG="logs/reproduce_$(date +%Y%m%d_%H%M%S).log"
PYTHON="${PYTHON:-python3}"
{
  "$PYTHON" -m shapfusion download
  "$PYTHON" -m shapfusion run "$@"
  "$PYTHON" -m shapfusion report "$@"
} 2>&1 | tee "$LOG"
echo "log: $LOG"
