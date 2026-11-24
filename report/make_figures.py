"""Make the report figures. Uses results/ (from scripts/collect_results.py) when available.

Usage (from the project folder):  python report/make_figures.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})

# Single-split test results (fallback if results/ has not been collected yet).
RUNS = {
    "ResNet-50\n(224 px)": ("resnet50_224", {"balanced_accuracy": .6624, "macro_f1": .6773, "macro_auc": .9128}),
    "DINOv2 ViT-B/14\n(392 px)": ("dinov2_vitb_392", {"balanced_accuracy": .6978, "macro_f1": .7011, "macro_auc": .8996}),
    "ConvNeXt-T\n(384 px)": ("convnext_tiny_384", {"balanced_accuracy": .7448, "macro_f1": .7382, "macro_auc": .9332}),
}
METRICS = [("balanced_accuracy", "Balanced acc."), ("macro_f1", "Macro F1"), ("macro_auc", "Macro AUC")]


def load_metrics(run, fallback):
    f = RES / run / "eval_test" / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else fallback


def fig_comparison():
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    x = np.arange(len(METRICS))
    w = 0.26
    colors = ["#9aa5b1", "#6c8ebf", "#d67f3c"]
    for i, (label, (run, fb)) in enumerate(RUNS.items()):
        m = load_metrics(run, fb)
        vals = [m[k] for k, _ in METRICS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label.replace("\n", " "), color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=6)
    cv = RES / "cv_convnext_tiny_384.json"
    if cv.exists():  # mark the 5-fold CV mean ± std of ConvNeXt on the test set
        c = json.loads(cv.read_text())["test"]
        ax.errorbar(x + w, [c[k]["mean"] for k, _ in METRICS], yerr=[c[k]["std"] for k, _ in METRICS],
                    fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_xticks(x, [n for _, n in METRICS])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Test score")
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=1)
    fig.savefig(FIG / "comparison.pdf")
    fig.savefig(FIG / "comparison.png")
    plt.close(fig)


def copy_if_exists(src, name):
    if src.exists():
        shutil.copy2(src, FIG / name)
        return True
    return False


if __name__ == "__main__":
    fig_comparison()
    copy_if_exists(RES / "convnext_tiny_384" / "eval_test" / "confusion_matrix.png", "confusion_convnext.png")
    copy_if_exists(RES / "convnext_tiny_384" / "eval_test" / "gradcam.png", "gradcam_convnext.png")
    copy_if_exists(RES / "gpu_log_dinov2.png", "gpu_dinov2.png")
    print("Figures in", FIG, ":", sorted(p.name for p in FIG.iterdir()))
