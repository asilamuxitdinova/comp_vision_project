#!/usr/bin/env bash
# 5-fold cross-validation of the main model with GPU logging, then summary + results export.
# Run inside tmux from the project folder:   bash scripts/run_cv.sh
# Optional: CONFIG=configs/resnet50.yaml bash scripts/run_cv.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

CONFIG=${CONFIG:-configs/default.yaml}
PREFIX=$(python -c "from src.utils import load_config; print(load_config('$CONFIG')['run_name'])")
LOG=outputs/gpu_log_${PREFIX}_cv.csv
mkdir -p outputs

echo "== GPU logging to $LOG =="
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 1 > "$LOG" &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null || true' EXIT

for k in 0 1 2 3 4; do
  echo "== Fold $k / 4 =="
  python -m src.train --config "$CONFIG" --fold "$k"
done

kill $LOGGER 2>/dev/null || true
sleep 1
python scripts/plot_gpu_log.py "$LOG" --title "$PREFIX, 5-fold CV on NVIDIA L4"
python scripts/summarize_cv.py "$PREFIX"
python scripts/collect_results.py
echo "Done. Results are in results/ (commit and push them, or copy them to the laptop)."
