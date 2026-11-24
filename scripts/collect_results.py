"""Copy the small result files (metrics, tables, plots, GPU logs) from outputs/ into results/,
which is tracked by git, so the experimental results can be pushed and submitted.
Model checkpoints (*.pt) are NOT copied.

Usage:
  python scripts/collect_results.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "outputs", ROOT / "results"
KEEP = {".json", ".csv", ".png", ".yaml"}
SKIP_RUNS = {"smoke"}

n = 0
for f in sorted(SRC.rglob("*")):
    rel = f.relative_to(SRC)
    if not f.is_file() or f.suffix not in KEEP or rel.parts[0] in SKIP_RUNS:
        continue
    out = DST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, out)
    n += 1
print(f"Copied {n} files into {DST}")
