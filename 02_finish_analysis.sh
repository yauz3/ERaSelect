#!/usr/bin/env bash
# Stage 2: every figure, every table, the manuscript draft and the readiness audit.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PROFILE="${PROFILE:-full}"
case "$PROFILE" in full|fast|smoke) ;; *) echo "PROFILE must be full, fast or smoke"; exit 2;; esac
[[ -x .venv/bin/python ]] || { echo "Run 01_run_experiments.sh first."; exit 1; }
[[ -f "results/$PROFILE/fold_metrics.csv" ]] || {
  echo "results/$PROFILE/fold_metrics.csv missing. Run 01_run_experiments.sh first."; exit 1; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
mkdir -p "results/$PROFILE/logs"

echo "[2/2] analysis  profile=$PROFILE"
.venv/bin/python -u src/analysis.py --config "config/${PROFILE}.json" 2>&1 \
  | tee "results/$PROFILE/logs/analysis.log"

echo
echo "Outputs in results/$PROFILE/"
echo "  figures/   PNG + SVG"
echo "  tables/    CSV + Markdown + LaTeX"
echo "  paper/     manuscript.md, submission_readiness.md"
echo
echo "Read paper/submission_readiness.md before drawing any conclusion."
