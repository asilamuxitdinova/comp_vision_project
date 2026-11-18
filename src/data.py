"""Dataset, augmentations and data loaders for the processed PAD-UFES-20 split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .utils import ROOT


class SkinLesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")
        return self.transform(img), int(row["label_idx"])


def build_transforms(img_size: int, mean, std):
    train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(30)], p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    evaluate = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train, evaluate


def load_splits(cfg: dict, fold: int | None = None):
    """Return (classes, train_df, val_df, test_df).

    fold=None uses the fixed `split` column; fold=k uses cv_fold==k as validation
    and the other folds as training (test is always the same held-out patients).
    """
    pdir = ROOT / cfg["data"]["processed_dir"]
    df = pd.read_csv(pdir / "splits.csv")
    classes = json.loads((pdir / "classes.json").read_text())["classes"]
    test = df[df["split"] == "test"]
    if fold is None:
        train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    else:
        pool = df[df["split"] != "test"]
        train, val = pool[pool["cv_fold"] != fold], pool[pool["cv_fold"] == fold]
    return classes, train, val, test


def class_weights(train_df: pd.DataFrame, num_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(train_df["label_idx"], minlength=num_classes).astype(float)
    w = 1.0 / np.maximum(counts, 1)
    if mode == "sqrt_inverse":
        w = np.sqrt(w)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def make_loader(df, cfg, transform, shuffle: bool) -> DataLoader:
    root = ROOT / cfg["data"]["processed_dir"]
    nw = cfg["data"].get("num_workers", 8)
    return DataLoader(
        SkinLesionDataset(df, root, transform),
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        drop_last=shuffle and len(df) > cfg["train"]["batch_size"],
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )
