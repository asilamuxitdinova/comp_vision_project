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
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(METRICS))
    series = [(label.replace("\n", " "), [load_metrics(run, fb)[k] for k, _ in METRICS], None)
              for label, (run, fb) in RUNS.items()]
    cv = RES / "cv_convnext_tiny_384.json"
    if cv.exists():  # 5-fold CV of ConvNeXt on the fixed test set: mean ± std
        c = json.loads(cv.read_text())["test"]
        series.append(("ConvNeXt-T, 5-fold mean ± sd",
                       [c[k]["mean"] for k, _ in METRICS], [c[k]["std"] for k, _ in METRICS]))
    colors = ["#9aa5b1", "#6c8ebf", "#d67f3c", "#f2c094"]
    w = 0.8 / len(series)
    for i, (label, vals, err) in enumerate(series):
        pos = x + (i - (len(series) - 1) / 2) * w
        ax.bar(pos, vals, w, yerr=err, capsize=1.5, error_kw={"lw": 0.7}, label=label, color=colors[i])
        for p, v, e in zip(pos, vals, err or [0] * len(vals)):
            ax.text(p, v + e + 0.008, f"{v:.2f}", ha="center", fontsize=5)
    ax.set_xticks(x, [n for _, n in METRICS])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Test score")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=1)
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
    cam = RES / "convnext_tiny_384" / "eval_test" / "gradcam.png"
    if cam.exists():  # keep the first two rows (ACK, BCC, MEL examples) to save space
        from PIL import Image
        im = Image.open(cam)
        im.crop((0, 50, im.width, 713)).save(FIG / "gradcam_convnext.png")
    print("Figures in", FIG, ":", sorted(p.name for p in FIG.iterdir()))
