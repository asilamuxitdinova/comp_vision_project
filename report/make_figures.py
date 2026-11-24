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

# 5-fold cross-validation of each backbone, all evaluated on the same fixed test set.
RUNS = {"ResNet-50 (224 px)": "resnet50_224", "DINOv2 ViT-B/14 (392 px)": "dinov2_vitb_392",
        "ConvNeXt-T (384 px)": "convnext_tiny_384"}
METRICS = [("balanced_accuracy", "Balanced\nacc."), ("macro_f1", "Macro\nF1"), ("macro_auc", "Macro\nAUC"),
           ("malignant_vs_benign_auc", "Malignant\nAUC")]


def fig_comparison():
    """Bars: CV mean +- sd on the test set; diamonds: ensemble of the five fold models."""
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(METRICS))
    colors = ["#9aa5b1", "#6c8ebf", "#d67f3c"]
    w = 0.8 / len(RUNS)
    for i, (label, run) in enumerate(RUNS.items()):
        cv = json.loads((RES / f"cv_{run}.json").read_text())
        mean = [cv["test"][k]["mean"] for k, _ in METRICS]
        sd = [cv["test"][k]["std"] for k, _ in METRICS]
        ens = [cv["test_ensemble"][k] for k, _ in METRICS]
        pos = x + (i - (len(RUNS) - 1) / 2) * w
        ax.bar(pos, mean, w, yerr=sd, capsize=1.5, error_kw={"lw": 0.7}, label=label, color=colors[i])
        ax.scatter(pos, ens, marker="D", s=7, color="black", zorder=3,
                   label="5-fold ensemble" if i == len(RUNS) - 1 else None)
        for p, m, e, en in zip(pos, mean, sd, ens):
            ax.text(p, max(m + e, en) + 0.01, f"{m:.2f}", ha="center", fontsize=4.5)
    ax.set_xticks(x, [n for _, n in METRICS])
    ax.set_ylim(0.5, 1.04)
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
