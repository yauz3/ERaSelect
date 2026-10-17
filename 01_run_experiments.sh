#!/usr/bin/env bash
# Stage 1: curate, audit the calibration budget, run the nested benchmark, evaluate externally.
#   bash 01_run_experiments.sh                 # full profile (default)
#   PROFILE=fast  bash 01_run_experiments.sh   # reduced grid, single seed
#   PROFILE=smoke bash 01_run_experiments.sh   # minutes, for checking the install
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PROFILE="${PROFILE:-full}"
case "$PROFILE" in full|fast|smoke) ;; *) echo "PROFILE must be full, fast or smoke"; exit 2;; esac
CFG="config/${PROFILE}.json"
[[ -f "$CFG" ]] || { echo "missing $CFG"; exit 2; }

if [[ ! -x .venv/bin/python ]]; then "${PYTHON:-python3}" -m venv .venv; fi
PY=.venv/bin/python
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt
if "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$CFG')).get('pretrained') else 1)"; then
  echo "[setup] pretrained representation enabled; installing torch + transformers (CPU)"
  "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  # protobuf and sentencepiece are optional transformers extras, but without them a tokenizer
  # that has to go through the slow path fails with a misleading missing-protobuf ImportError.
  "$PY" -m pip install --quiet 'transformers>=4.45,<5' 'huggingface-hub>=0.25,<1' \
      'tokenizers>=0.20' 'protobuf>=4.25' 'sentencepiece>=0.2'
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
mkdir -p "results/$PROFILE/logs"

echo "[1/2] experiments  profile=$PROFILE"
"$PY" -u src/experiments.py --config "$CFG" 2>&1 | tee "results/$PROFILE/logs/experiments.log"
"$PY" -m pip freeze > "results/$PROFILE/environment_lock.txt"
echo
echo "Done. Now run:  PROFILE=$PROFILE bash 02_finish_analysis.sh"
