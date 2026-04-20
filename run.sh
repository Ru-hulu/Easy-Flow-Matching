#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-default}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("numpy") is None:
    sys.exit("NumPy is not installed. Run `python3 -m pip install -r requirements.txt` or set PYTHON_BIN to a Python with NumPy.")
PY

case "$MODE" in
  smoke)
    "$PYTHON_BIN" toy_flow_matching.py \
      --steps 300 \
      --samples 1000 \
      --out-dir outputs_smoke
    ;;
  default)
    "$PYTHON_BIN" toy_flow_matching.py \
      --steps 2500 \
      --samples 2500 \
      --out-dir outputs
    ;;
  long)
    "$PYTHON_BIN" toy_flow_matching.py \
      --steps 20000 \
      --samples 5000 \
      --hidden 96 \
      --out-dir outputs_long
    ;;
  *)
    echo "Usage: ./run.sh [smoke|default|long]"
    exit 2
    ;;
esac
