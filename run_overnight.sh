#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LIMIT=15
LOG="matter_test_run_$(date +%Y%m%d_%H%M%S).log"

echo "Starting RNSR matter tests — $LIMIT random directories"
echo "Log file: $LOG"
echo ""

.venv/bin/python run_matter_tests.py --limit "$LIMIT" --random 2>&1 | tee "$LOG"

echo ""
echo "Done. Results in matterAiTests/matter_test_results.json"
echo "Full log: $LOG"
