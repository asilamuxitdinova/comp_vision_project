"""Summarize k-fold cross-validation runs: mean ± std, plus a 5-model ensemble on the test set.

Expects runs named <prefix>_fold0 ... <prefix>_fold{k-1} in outputs/ (made by
`python -m src.train --config ... --fold k`).

Usage:
  python scripts/summarize_cv.py convnext_tiny_384
Writes outputs/cv_<prefix>.json and prints a table.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
MALIGNANT = ["BCC", "MEL", "SCC"]
KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "malignant_vs_benign_auc"]


def metrics_from_probs(labels, probs):
    preds = probs.argmax(1)
    mal = [CLASSES.index(c) for c in MALIGNANT]
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_auc": roc_auc_score(labels, probs, multi_class="ovr", average="macro"),
        "malignant_vs_benign_auc": roc_auc_score(np.isin(labels, mal), probs[:, mal].sum(1)),
        "per_class_recall": dict(zip(CLASSES, recall_score(labels, preds, average=None,
                                                           labels=range(len(CLASSES))).round(3))),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    prefix = sys.argv[1]
    folds = sorted((ROOT / "outputs").glob(f"{prefix}_fold*"))
    if not folds:
        raise SystemExit(f"No outputs/{prefix}_fold* runs found")

    val_rows, test_rows, preds = [], [], []
    for d in folds:
        s = json.loads((d / "train_summary.json").read_text())
        val_rows.append({k: s["best_val_metrics"].get(k) for k in KEYS})
        t = d / "eval_test" / "metrics.json"
        if t.exists():
            test_rows.append({k: json.loads(t.read_text()).get(k) for k in KEYS})
            preds.append(pd.read_csv(d / "eval_test" / "predictions.csv"))

    out = {"prefix": prefix, "folds": [d.name for d in folds]}
    for name, rows in [("val", val_rows), ("test", test_rows)]:
        if rows:
            df = pd.DataFrame(rows)
            out[name] = {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=1))} for k in KEYS}

    if len(preds) > 1:  # ensemble = average class probabilities of the fold models
        base = preds[0]
        assert all((p["img_id"] == base["img_id"]).all() for p in preds)
        probs = np.mean([p[[f"p_{c}" for c in CLASSES]].to_numpy() for p in preds], axis=0)
        labels = base["label"].map(CLASSES.index).to_numpy()
        out["test_ensemble"] = {k: (float(v) if not isinstance(v, dict) else
                                    {c: float(r) for c, r in v.items()})
                                for k, v in metrics_from_probs(labels, probs).items()}

    (ROOT / "outputs" / f"cv_{prefix}.json").write_text(json.dumps(out, indent=2))
    print(f"{len(folds)} folds of {prefix}\n")
    print(f"{'metric':<26}{'CV val (mean±std)':<22}{'test (mean±std)':<22}{'test ensemble'}")
    for k in KEYS:
        v = out.get("val", {}).get(k)
        t = out.get("test", {}).get(k)
        e = out.get("test_ensemble", {}).get(k)
        fmt = lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}" if x else "-"
        print(f"{k:<26}{fmt(v):<22}{fmt(t):<22}{e:.4f}" if e is not None else
              f"{k:<26}{fmt(v):<22}{fmt(t):<22}-")
    if "test_ensemble" in out:
        print(f"\nEnsemble per-class recall: {out['test_ensemble']['per_class_recall']}")
    print(f"\nWrote outputs/cv_{prefix}.json")


if __name__ == "__main__":
    main()
