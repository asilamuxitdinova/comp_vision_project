#!/usr/bin/env bash
# One-shot setup on the GPU server: Python env -> PyTorch -> libraries -> dataset -> preprocessing.
# Run from the project folder:   bash setup_server.sh
# Safe to rerun; finished steps are quick the second time.
#
# Options (environment variables):
#   DATA_SOURCE=mendeley bash setup_server.sh     # official download instead of Hugging Face
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 bash setup_server.sh   # pick a CUDA build
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo "== 1/5 Python environment (.venv) =="
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

echo "== 2/5 PyTorch =="
if python -c "import torch" 2>/dev/null; then
  echo "already installed"
elif [ -n "${TORCH_INDEX_URL:-}" ]; then
  pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
  pip install torch torchvision   # Linux wheels on PyPI ship with CUDA
fi
python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
else:
    print("WARNING: no GPU visible to PyTorch - training will be very slow.")
EOF

echo "== 3/5 Libraries =="
pip install -r requirements.txt -q

echo "== 4/5 Dataset (PAD-UFES-20) =="
python scripts/download_data.py --source "${DATA_SOURCE:-hf}"

echo "== 5/5 Resize images + patient-level split =="
python scripts/prepare_data.py

echo
echo "Done. Start training with:"
echo "  source .venv/bin/activate"
echo "  python -m src.train --config configs/default.yaml"
