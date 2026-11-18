"""Evaluate a trained checkpoint: metrics, confusion matrix, ROC curves,
results by Fitzpatrick skin type, and Grad-CAM heatmaps.

Usage:
  python -m src.evaluate --checkpoint outputs/convnext_tiny_384/best.pt [--split test] [--gradcam 24] [--no-tta]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_auc_score, roc_curve)

from .data import build_transforms, load_splits, make_loader
from .model import build_model
from .utils import amp_dtype, get_device, save_json

MALIGNANT = ["BCC", "MEL", "SCC"]  # ACK is pre-cancerous; NEV and SEK are benign


# ----------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict(model, loader, device, dtype=None, tta=False):
    """Return (probabilities [N, C], labels [N]) as numpy arrays."""
    model.eval()
    all_p, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            p = model(x).float().softmax(1)
            if tta:
                p += model(torch.flip(x, dims=[3])).float().softmax(1)
                p += model(torch.flip(x, dims=[2])).float().softmax(1)
                p /= 3
        all_p.append(p.cpu())
        all_y.append(y)
    return torch.cat(all_p).numpy(), torch.cat(all_y).numpy()


def compute_metrics(labels, probs, classes) -> dict:
    preds = probs.argmax(1)
    k = len(classes)
    m = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", labels=range(k), zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", labels=range(k), zero_division=0),
        "log_loss": log_loss(labels, np.clip(probs, 1e-7, 1), labels=range(k)),
    }
    try:
        m["macro_auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=range(k))
    except ValueError:  # a class is absent from this subset
        m["macro_auc"] = float("nan")
    mal = [classes.index(c) for c in MALIGNANT if c in classes]
    is_mal = np.isin(labels, mal)
    if 0 < is_mal.sum() < len(is_mal):
        m["malignant_vs_benign_auc"] = roc_auc_score(is_mal, probs[:, mal].sum(1))
    return {key: float(v) for key, v in m.items()}


# ----------------------------------------------------------------------------- plots
def plot_confusion(labels, preds, classes, path):
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted", ylabel="True", title="Confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(labels, probs, classes, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, c in enumerate(classes):
        if 0 < (labels == i).sum() < len(labels):
            fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
            auc = roc_auc_score(labels == i, probs[:, i])
            ax.plot(fpr, tpr, label=f"{c} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_skin_type(df, labels, probs, classes) -> pd.DataFrame:
    preds = probs.argmax(1)
    skin = df["fitspatrick"].fillna(-1).astype(int).astype(str).replace("-1", "unknown").to_numpy()
    rows = []
    for g in sorted(set(skin)):
        idx = skin == g
        rows.append({
            "fitzpatrick": g,
            "n": int(idx.sum()),
            "accuracy": accuracy_score(labels[idx], preds[idx]),
            "balanced_accuracy": balanced_accuracy_score(labels[idx], preds[idx])
            if len(set(labels[idx])) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Grad-CAM
def _cam_layer(model):
    if hasattr(model, "stages"):          # ConvNeXt, EfficientNetV2 (timm)
        return model.stages[-1], None
    if hasattr(model, "layer4"):          # ResNet
        return model.layer4[-1], None
    if hasattr(model, "blocks") and hasattr(model, "patch_embed"):  # ViT
        n_prefix = getattr(model, "num_prefix_tokens", 1)
        gh, gw = model.patch_embed.grid_size

        def reshape(t):
            t = t[:, n_prefix:, :]
            return t.reshape(t.size(0), gh, gw, t.size(2)).permute(0, 3, 1, 2)
        return model.blocks[-1].norm1, reshape
    if hasattr(model, "conv_head"):       # EfficientNet
        return model.conv_head, None
    raise ValueError("Don't know which layer to use for Grad-CAM with this model")


def gradcam_grid(model, df, dataset, classes, mean, std, n, path, device):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Pick images spread across classes.
    per_class = max(1, n // len(classes))
    idx = (df.reset_index(drop=True).groupby("label_idx", group_keys=False)
           .apply(lambda g: g.head(per_class)).index.tolist())[:n]
    layer, reshape = _cam_layer(model)
    model.eval().float()
    mean_t, std_t = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)

    cols = 6
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.7))
    axes = np.atleast_1d(axes).ravel()
    with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
        for ax, i in zip(axes, idx):
            x, y = dataset[i]
            with torch.no_grad():
                pred = int(model(x[None].to(device)).argmax(1))
            heat = cam(input_tensor=x[None].to(device), targets=[ClassifierOutputTarget(pred)])[0]
            rgb = (x * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(show_cam_on_image(rgb, heat, use_rgb=True))
            ok = "✓" if pred == y else "✗"
            ax.set_title(f"true {classes[y]} / pred {classes[pred]} {ok}", fontsize=8,
                         color="green" if pred == y else "red")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Grad-CAM: regions driving the predicted class", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main entry
def run(checkpoint, split="test", tta=None, gradcam=None, out_dir=None) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, classes = ckpt["cfg"], ckpt["classes"]
    tta = cfg["eval"].get("tta", True) if tta is None else tta
    gradcam = cfg["eval"].get("gradcam_images", 0) if gradcam is None else gradcam
    out_dir = Path(out_dir or Path(checkpoint).parent) / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = build_model(cfg, len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).to(memory_format=torch.channels_last)

    _, eval_tf = build_transforms(cfg["data"]["img_size"], ckpt["mean"], ckpt["std"])
    _, train_df, val_df, test_df = load_splits(cfg, ckpt.get("fold"))
    df = {"train": train_df, "val": val_df, "test": test_df}[split].reset_index(drop=True)
    loader = make_loader(df, cfg, eval_tf, shuffle=False)

    probs, labels = predict(model, loader, device, amp_dtype(cfg, device), tta=tta)
    preds = probs.argmax(1)
    metrics = compute_metrics(labels, probs, classes)
    metrics.update(split=split, n=len(labels), tta=tta, epoch=ckpt.get("epoch"))
    save_json(metrics, out_dir / "metrics.json")
    save_json(classification_report(labels, preds, labels=range(len(classes)), target_names=classes,
                                    output_dict=True, zero_division=0), out_dir / "per_class.json")

    pred_df = df[["img_id", "patient_id", "label", "fitspatrick"]].copy()
    pred_df["pred"] = [classes[p] for p in preds]
    for i, c in enumerate(classes):
        pred_df[f"p_{c}"] = probs[:, i]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    skin = by_skin_type(df, labels, probs, classes)
    skin.to_csv(out_dir / "by_skin_type.csv", index=False)
    plot_confusion(labels, preds, classes, out_dir / "confusion_matrix.png")
    plot_roc(labels, probs, classes, out_dir / "roc_curves.png")
    if gradcam:
        gradcam_grid(model, df, loader.dataset, classes, ckpt["mean"], ckpt["std"], gradcam,
                     out_dir / "gradcam.png", device)

    print(f"\n== {split} results ({len(labels)} images, TTA={tta}) ==")
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "malignant_vs_benign_auc"]:
        if k in metrics:
            print(f"  {k:<24} {metrics[k]:.4f}")
    print(classification_report(labels, preds, labels=range(len(classes)), target_names=classes,
                                digits=3, zero_division=0))
    print("By Fitzpatrick skin type:\n" + skin.to_string(index=False, float_format="%.3f"))
    print(f"\nSaved to {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--gradcam", type=int, default=None, help="number of Grad-CAM images (0 = off)")
    args = ap.parse_args()
    run(args.checkpoint, args.split, tta=False if args.no_tta else None, gradcam=args.gradcam)


if __name__ == "__main__":
    main()
