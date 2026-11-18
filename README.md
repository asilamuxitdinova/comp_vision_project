# Skin Lesion Classifier (PAD-UFES-20)

Classifies smartphone photos of skin lesions into 6 diagnoses with transfer learning
(PyTorch + timm), using a **patient-level** train/validation/test split.

| Code | Diagnosis | Type |
|---|---|---|
| ACK | Actinic keratosis | pre-cancerous |
| BCC | Basal cell carcinoma | malignant |
| MEL | Melanoma | malignant |
| NEV | Nevus (mole) | benign |
| SCC | Squamous cell carcinoma | malignant |
| SEK | Seborrheic keratosis | benign |

> This is a course project. It is not a medical device and must not be used for diagnosis.

## Project layout

```
setup_server.sh          one-shot setup on the GPU server (env, PyTorch, data, preprocessing)
requirements.txt
configs/
  default.yaml           ConvNeXt-Tiny @ 384 px (main model)
  resnet50.yaml          ResNet-50 @ 224 px (CNN baseline)
  dinov2_vitb.yaml       DINOv2 ViT-B/14 @ 392 px (Vision Transformer)
scripts/
  download_data.py       downloads PAD-UFES-20 (Hugging Face mirror or official Mendeley)
  prepare_data.py        resizes images, makes patient-level stratified splits
  compare_runs.py        collects test metrics of all runs into one table
  plot_gpu_log.py        GPU-usage chart + summary from an nvidia-smi log
src/
  data.py                Dataset, augmentations, loaders, class weights
  model.py               timm model builder
  train.py               training loop (bf16, warmup + cosine LR, early stopping)
  evaluate.py            metrics, confusion matrix, ROC, skin-type breakdown, Grad-CAM
data/      (git-ignored) raw + processed images
outputs/   (git-ignored) checkpoints, logs, plots
```

## Quick start (GPU server)

```bash
# copy this folder to the server, then on the server:
cd skin-lesion-classifier
hf auth login          # optional: uses your Hugging Face account for faster downloads
bash setup_server.sh   # ~5-15 min: env, PyTorch, dataset (~3.6 GB), preprocessing
```

Then train. Run long jobs inside `tmux` so they survive a dropped SSH connection:

```bash
tmux new -s train
source .venv/bin/activate
python -m src.train --config configs/default.yaml       # ConvNeXt-Tiny
python -m src.train --config configs/resnet50.yaml      # baseline
python -m src.train --config configs/dinov2_vitb.yaml   # ViT
python scripts/compare_runs.py                          # results table
```

(Detach from tmux with `Ctrl-b d`, reattach with `tmux attach -t train`.)

Before a full run, a 2-epoch smoke test catches problems in about a minute:

```bash
python -m src.train --set train.epochs=2 run_name=smoke
```

### Useful options

```bash
# override any config value from the command line
python -m src.train --config configs/default.yaml --set train.lr=5e-5 train.epochs=40 run_name=cnx_lr5e-5

# 5-fold cross-validation (test set stays the same held-out patients)
for k in 0 1 2 3 4; do python -m src.train --config configs/default.yaml --fold $k; done

# re-evaluate a checkpoint (e.g. more Grad-CAM images, or without test-time augmentation)
python -m src.evaluate --checkpoint outputs/convnext_tiny_384/best.pt --gradcam 36
python -m src.evaluate --checkpoint outputs/convnext_tiny_384/best.pt --no-tta
```

### GPU usage chart

Record GPU stats once per second in a separate tmux session while training runs,
then turn the log into a chart and summary (mean utilization, peak memory, power, energy):

```bash
tmux new -s gpu
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 1 | tee outputs/gpu_log_convnext.csv
# Ctrl-c when training is done, then:
python scripts/plot_gpu_log.py outputs/gpu_log_convnext.csv --title "ConvNeXt-Tiny @ 384 px on NVIDIA L4"
```

This writes `outputs/gpu_log_convnext.png` and `outputs/gpu_log_convnext_summary.json`.
Use one log per model to compare their GPU cost.

If data loading is the bottleneck (GPU utilization in `nvidia-smi` well below 90%), set
`data.num_workers` to the number of CPU cores (`nproc`).

## What each run produces (`outputs/<run_name>/`)

| File | Contents |
|---|---|
| `best.pt` | weights of the best epoch (by validation balanced accuracy) |
| `config.yaml` | exact config used |
| `history.csv`, `training_curves.png` | per-epoch loss and validation metrics |
| `eval_test/metrics.json` | accuracy, balanced accuracy, macro F1, macro AUC, malignant-vs-benign AUC |
| `eval_test/per_class.json` | precision / recall / F1 per class |
| `eval_test/confusion_matrix.png`, `roc_curves.png` | plots for the report |
| `eval_test/by_skin_type.csv` | accuracy per Fitzpatrick skin type (fairness check) |
| `eval_test/gradcam.png` | heatmaps of image regions behind each prediction |
| `eval_test/predictions.csv` | per-image predictions and class probabilities |

## Method notes

- **Patient-level split** (70 / 15 / 15, stratified by diagnosis, `StratifiedGroupKFold`):
  no patient appears in more than one split, so test scores are not inflated by
  near-duplicate photos of the same lesion.
- **Class imbalance** (845 BCC vs 52 MEL): class-weighted cross-entropy
  (`sqrt_inverse`), and **balanced accuracy** (mean per-class recall) as the main metric.
- **Transfer learning**: ImageNet-pretrained backbones fine-tuned end-to-end with AdamW,
  linear warmup + cosine decay, label smoothing, bf16 mixed precision.
- **Augmentation**: random resized crop, horizontal/vertical flips, rotation, color jitter.
  Test-time augmentation averages the original and flipped images.
- **Caveat**: 35% of images have no Fitzpatrick skin type recorded, and very dark skin
  types (5–6) are rare, so the skin-type breakdown is indicative only.

## Data and citation

PAD-UFES-20 (CC BY 4.0), downloaded by `scripts/download_data.py` from the Hugging Face
mirror `SalmaneExploring/pad-ufes-20` (pinned revision) or the official Mendeley release.
The script checks `metadata.csv` against the official file's SHA-256.

> Pacheco, A. G. C. et al. *PAD-UFES-20: a skin lesion dataset composed of patient data and
> clinical images collected from smartphones.* Data in Brief 32 (2020) 106221.
> https://doi.org/10.17632/zr7vgbcyr2.1
