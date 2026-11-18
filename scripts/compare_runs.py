"""Collect test metrics from every run in outputs/ into one table (for the report).

Usage:
  python scripts/compare_runs.py            # prints a table, writes outputs/comparison.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "malignant_vs_benign_auc", "epoch"]

rows = []
for f in sorted((ROOT / "outputs").glob("*/eval_test/metrics.json")):
    m = json.loads(f.read_text())
    rows.append({"run": f.parent.parent.name, **{c: m.get(c) for c in COLS}})

if not rows:
    raise SystemExit("No outputs/*/eval_test/metrics.json found yet - train a model first.")
df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
df.to_csv(ROOT / "outputs" / "comparison.csv", index=False)
print(df.to_string(index=False, float_format="%.4f"))
