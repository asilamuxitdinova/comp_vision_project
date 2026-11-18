"""Download PAD-UFES-20 into data/raw/ (run this on the server).

Two sources:
  hf        (default) Hugging Face mirror SalmaneExploring/pad-ufes-20, pinned to a
            fixed revision. Fast; uses your `hf auth login` token if present.
  mendeley  The official release on Mendeley Data (3 zip files, ~3.6 GB).

Either way the script checks that metadata.csv is byte-identical to the official
release (SHA-256) and that every image listed in it exists.

Usage:
  python scripts/download_data.py                 # Hugging Face
  python scripts/download_data.py --source mendeley
"""
import argparse
import csv
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# SHA-256 of metadata.csv in the official Mendeley release (v1).
METADATA_SHA256 = "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"
EXPECTED_IMAGES = 2298

HF_REPO = "SalmaneExploring/pad-ufes-20"
HF_REVISION = "086dd94836d7b2094132a21406a75dcb6bf7ec2c"  # pinned for reproducibility

MENDELEY = "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files"
MENDELEY_FILES = {
    "metadata.csv": ("fa850265-57da-48f0-ba3e-998b3e44b1f6", METADATA_SHA256),
    "imgs_part_1.zip": ("1cc2f71f-20a2-412d-b746-a9b9bc20c966",
                        "0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca"),
    "imgs_part_2.zip": ("559a60ed-5504-475d-996c-6a8bc253b5e7",
                        "e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b"),
    "imgs_part_3.zip": ("34dcdf8e-e5f1-4b35-aa0b-5135051ff852",
                        "ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_hf() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is missing: pip install -r requirements.txt")
    print(f"Downloading {HF_REPO}@{HF_REVISION[:8]} from Hugging Face ...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=RAW_DIR,
        allow_patterns=["metadata.csv", "all_images/**"],
        max_workers=16,
    )


def _fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def download_mendeley() -> None:
    for name, (file_id, digest) in MENDELEY_FILES.items():
        dest = RAW_DIR / name
        if dest.exists() and sha256(dest) == digest:
            print(f"  {name}: already downloaded")
        else:
            _fetch(f"{MENDELEY}/{file_id}/file_downloaded", dest)
            if sha256(dest) != digest:
                dest.unlink()
                sys.exit(f"Checksum mismatch for {name}; deleted it, please rerun.")
        if name.endswith(".zip"):
            print(f"  extracting {name} ...")
            with zipfile.ZipFile(dest) as z:
                z.extractall(RAW_DIR / "all_images")
            dest.unlink()  # free ~1.2 GB per archive


def verify() -> None:
    meta = RAW_DIR / "metadata.csv"
    if not meta.exists():
        sys.exit(f"{meta} not found")
    if sha256(meta) != METADATA_SHA256:
        sys.exit("metadata.csv does not match the official release (SHA-256 mismatch)")

    with open(meta, newline="") as f:
        wanted = {row["img_id"] for row in csv.DictReader(f)}
    found = {p.name for p in RAW_DIR.rglob("*.png")}
    missing = wanted - found
    if missing:
        sys.exit(f"{len(missing)} images listed in metadata.csv are missing, e.g. {sorted(missing)[:3]}")
    print(f"OK: metadata.csv verified, {len(wanted)} / {EXPECTED_IMAGES} images present in {RAW_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "mendeley"], default="hf")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_hf() if args.source == "hf" else download_mendeley()
    verify()


if __name__ == "__main__":
    main()
