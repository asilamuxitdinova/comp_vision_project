"""Train a skin-lesion classifier, keep the best epoch on validation, then test it.

Usage:
  python -m src.train --config configs/default.yaml
  python -m src.train --config configs/resnet50.yaml
  python -m src.train --config configs/default.yaml --fold 0          # cross-validation fold
  python -m src.train --config configs/default.yaml --set train.epochs=5 run_name=quick
"""
import argparse
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from . import evaluate
from .data import build_transforms, class_weights, load_splits, make_loader
from .model import build_model, normalization
from .utils import ROOT, amp_dtype, get_device, load_config, save_json, set_seed


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)  # no weight decay on norms/biases
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def warmup_cosine(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def plot_history(hist: pd.DataFrame, monitor: str, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(hist["epoch"], hist["train_loss"], label="train")
    a.plot(hist["epoch"], hist["val_log_loss"], label="val")
    a.set(xlabel="epoch", ylabel="loss", title="Loss")
    a.legend()
    b.plot(hist["epoch"], hist["train_acc"], label="train accuracy")
    b.plot(hist["epoch"], hist["val_accuracy"], label="val accuracy")
    b.plot(hist["epoch"], hist[f"val_{monitor}"], label=f"val {monitor}")
    b.set(xlabel="epoch", title="Validation metrics")
    b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fold", type=int, default=None, help="use cv_fold==k as validation")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys")
    ap.add_argument("--no-test", action="store_true", help="skip the final test-set evaluation")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    run_name = cfg["run_name"] + (f"_fold{args.fold}" if args.fold is not None else "")
    out = ROOT / "outputs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    set_seed(cfg["seed"])
    device = get_device()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = amp_dtype(cfg, device)
    t = cfg["train"]

    classes, train_df, val_df, _ = load_splits(cfg, args.fold)
    print(f"Device: {device}  |  AMP: {dtype}  |  run: {run_name}")
    print(f"Train {len(train_df)} / val {len(val_df)} images, classes {classes}")

    model = build_model(cfg, len(classes))
    mean, std = normalization(model)
    model.to(device).to(memory_format=torch.channels_last)
    print(f"Model {cfg['model']['name']}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    train_tf, eval_tf = build_transforms(cfg["data"]["img_size"], mean, std)
    train_loader = make_loader(train_df, cfg, train_tf, shuffle=True)
    val_loader = make_loader(val_df, cfg, eval_tf, shuffle=False)

    w = class_weights(train_df, len(classes), t.get("class_weights", "none"))
    criterion = nn.CrossEntropyLoss(weight=None if w is None else w.to(device),
                                    label_smoothing=t.get("label_smoothing", 0.0))
    optimizer = make_optimizer(model, t["lr"], t["weight_decay"])
    steps = len(train_loader)
    scheduler = warmup_cosine(optimizer, t.get("warmup_epochs", 0) * steps, t["epochs"] * steps)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    monitor, best, bad_epochs, history = t.get("monitor", "balanced_accuracy"), -math.inf, 0, []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False):
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                logits = model(x)
                loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if t.get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        probs, labels = evaluate.predict(model, val_loader, device, dtype)
        val = evaluate.compute_metrics(labels, probs, classes)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "lr": scheduler.get_last_lr()[0], "time_s": time.time() - t0,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        improved = val[monitor] > best
        print(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} | train acc {row['train_acc']:.3f} | "
              f"val acc {val['accuracy']:.3f} | val bal-acc {val['balanced_accuracy']:.3f} | "
              f"val AUC {val['macro_auc']:.3f} | {row['time_s']:.0f}s" + ("  *best*" if improved else ""))
        if improved:
            best, bad_epochs = val[monitor], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "classes": classes, "epoch": epoch,
                        "fold": args.fold, "mean": mean, "std": std, "val_metrics": val},
                       out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= t.get("early_stopping_patience", 10**9):
                print(f"Early stopping: no {monitor} improvement for {bad_epochs} epochs")
                break

    hist = pd.DataFrame(history)
    plot_history(hist, monitor, out / "training_curves.png")
    best_row = hist.loc[hist[f"val_{monitor}"].idxmax()]
    save_json({
        "best_epoch": int(best_row["epoch"]),
        f"best_val_{monitor}": best,
        "best_val_metrics": {k[4:]: float(v) for k, v in best_row.items() if k.startswith("val_")},
        "epochs_run": len(hist),
        "train_time_min": float(hist["time_s"].sum() / 60),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "fold": args.fold,
    }, out / "train_summary.json")
    print(f"\nBest val {monitor} {best:.4f} at epoch {int(best_row['epoch'])}. Checkpoint: {out / 'best.pt'}")

    if not args.no_test:
        evaluate.run(out / "best.pt", split="test")


if __name__ == "__main__":
    main()
