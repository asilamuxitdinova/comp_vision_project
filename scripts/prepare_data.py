"""Resize PAD-UFES-20 photos once and make a patient-level train/val/test split.

Reads   data/raw/metadata.csv + data/raw/**/*.png
Writes  data/processed/images/<img_id>.jpg   (shorter side = --size px, JPEG q95)
        data/processed/splits.csv            (one row per image, with `split` and `cv_fold`)
        data/processed/classes.json

All photos of one patient land in the same split, so the model is never tested on
a person it saw during training. Splits are also stratified by diagnosis.

Usage:
  python scripts/prepare_data.py [--size 512] [--test-frac 0.15] [--val-frac 0.15] [--seed 42]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CLASS_NAMES = {
    "ACK": "Actinic keratosis",
    "BCC": "Basal cell carcinoma",
    "MEL": "Melanoma",
    "NEV": "Nevus",
    "SCC": "Squamous cell carcinoma",
    "SEK": "Seborrheic keratosis",
}


def resize_one(job):
    src, dst, size = job
    if dst.exists():
        return
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    scale = size / min(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
    img.save(dst, "JPEG", quality=95)


def pick_one_fold(y, groups, n_splits, seed):
    """Return a boolean mask selecting ~1/n_splits of the rows, grouped + stratified."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(np.zeros(len(y)), y, groups))
    mask = np.zeros(len(y), dtype=bool)
    mask[held] = True
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=512, help="shorter image side after resizing")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--cv-folds", type=int, default=5, help="folds over train+val for cross-validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    df = pd.read_csv(RAW_DIR / "metadata.csv")
    df = df[df["diagnostic"].isin(CLASSES)].reset_index(drop=True)

    # Locate every image regardless of how the archives were unpacked.
    paths = {p.name: p for p in RAW_DIR.rglob("*.png")}
    missing = set(df["img_id"]) - set(paths)
    if missing:
        raise SystemExit(f"{len(missing)} images missing; run scripts/download_data.py first")

    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    df["path"] = "images/" + df["img_id"].str.replace(".png", ".jpg", regex=False)
    jobs = [(paths[i], OUT_DIR / p, args.size) for i, p in zip(df["img_id"], df["path"])]
    print(f"Resizing {len(jobs)} images to shorter side {args.size}px with {args.workers} workers ...")
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(resize_one, jobs, chunksize=16))

    # Patient-level, stratified splits.
    df["label"] = df["diagnostic"]
    df["label_idx"] = df["label"].map(CLASSES.index)
    y, groups = df["label_idx"].to_numpy(), df["patient_id"].to_numpy()

    test = pick_one_fold(y, groups, round(1 / args.test_frac), args.seed)
    rest = np.flatnonzero(~test)
    val_in_rest = pick_one_fold(y[rest], groups[rest], round((1 - args.test_frac) / args.val_frac), args.seed)
    df["split"] = "train"
    df.loc[test, "split"] = "test"
    df.loc[rest[val_in_rest], "split"] = "val"

    # Optional k-fold CV over train+val (test stays untouched).
    df["cv_fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    for k, (_, held) in enumerate(sgkf.split(np.zeros(len(rest)), y[rest], groups[rest])):
        df.loc[rest[held], "cv_fold"] = k

    # Sanity check: no patient in more than one split.
    assert df.groupby("patient_id")["split"].nunique().max() == 1

    cols = ["img_id", "path", "label", "label_idx", "patient_id", "lesion_id",
            "fitspatrick", "age", "gender", "region", "biopsed", "split", "cv_fold"]
    df[cols].to_csv(OUT_DIR / "splits.csv", index=False)
    (OUT_DIR / "classes.json").write_text(json.dumps(
        {"classes": CLASSES, "names": CLASS_NAMES}, indent=2))

    print("\nImages per split and class:")
    print(pd.crosstab(df["label"], df["split"], margins=True).to_string())
    print(f"\nPatients per split: {df.groupby('split')['patient_id'].nunique().to_dict()}")
    print(f"Wrote {OUT_DIR / 'splits.csv'}")


if __name__ == "__main__":
    main()
