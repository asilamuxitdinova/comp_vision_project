# Results and Discussion

Skin-lesion classification on PAD-UFES-20 with three pretrained backbones, fine-tuned on an
NVIDIA L4 GPU. Numbers come from a patient-level train/val/test split (seed 42), and every model was
also 5-fold cross-validated (15 extra training runs, all evaluated on the same test set). The 4-page paper is in `report/report.pdf`.

## 1. Setup

### Data

| | Images | Patients |
|---|---|---|
| Train | 1,640 | 980 |
| Validation | 329 | 196 |
| Test | 329 | 197 |
| **Total** | **2,298** | **1,373** |

- 6 classes: ACK, BCC, MEL, NEV, SCC, SEK (smartphone photos, not dermoscopy).
- **Patient-level, stratified split** (`StratifiedGroupKFold`): no patient appears in more than
  one split, so test scores are not inflated by other photos of the same person or lesion.
- **Heavy class imbalance**: 845 BCC vs 52 MEL in total; only **7 melanomas in the test set**.

Test-set class counts:

| ACK | BCC | MEL | NEV | SCC | SEK |
|---|---|---|---|---|---|
| 104 | 121 | 7 | 35 | 28 | 34 |

### Hardware and software

- GPU: NVIDIA L4, 23 GB memory, 72 W power limit; driver 595.91, CUDA 13.2.
- PyTorch + timm, bf16 mixed precision, `channels_last` memory format.

### Training recipe (shared by all models)

- Transfer learning: pretrained backbone with a new 6-class head; the whole network is fine-tuned.
- Optimizer AdamW, linear warmup then cosine learning-rate decay, up to 30 epochs.
- Early stopping (patience 7) on **validation balanced accuracy**; the best epoch is kept.
- Loss: cross-entropy with label smoothing 0.1 and **class weights** (square root of inverse
  class frequency), so rare classes like MEL count more.
- Augmentation: random resized crop (scale 0.6–1.0), horizontal/vertical flips, rotation ±30°,
  color jitter.
- Test-time augmentation: average of the original, horizontally flipped and vertically
  flipped image.

### Models compared

| Model | timm name | Type | Pretraining | Params* | Image size | Batch | LR | Best epoch |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 | `resnet50.a1_in1k` | CNN (2015) | ImageNet-1k | ~24M | 224 | 64 | 3e-4 | 26 |
| ConvNeXt-Tiny | `convnext_tiny.fb_in22k_ft_in1k` | modern CNN (2022) | ImageNet-22k → 1k | ~28M | 384 | 32 | 1e-4 | 12 |
| DINOv2 ViT-B/14 | `vit_base_patch14_dinov2.lvd142m` | Vision Transformer | self-supervised, 142M images | ~86M | 392 | 16 | 2e-5 | 18 |

\* Approximate; the exact count is printed on the `Model ... params` line of each training log.

## 2. Results on the test set

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro AUC | Malignant vs. benign AUC |
|---|---|---|---|---|---|
| *Single run (main split)* | | | | | |
| **ConvNeXt-Tiny @ 384** | **79.3%** | **74.5%** | **0.738** | **0.933** | 0.930 |
| DINOv2 ViT-B/14 @ 392 | 76.9% | 69.8% | 0.701 | 0.900 | 0.915 |
| ResNet-50 @ 224 | 74.5% | 66.2% | 0.677 | 0.913 | 0.930 |
| *5-fold CV, mean ± std* | | | | | |
| **ConvNeXt-Tiny** | 73.7 ± 6.6% | **69.2 ± 3.2%** | **0.679 ± 0.048** | **0.922 ± 0.005** | **0.930 ± 0.008** |
| DINOv2 ViT-B/14 | 74.0 ± 2.2% | 66.6 ± 2.1% | 0.674 ± 0.030 | 0.916 ± 0.011 | 0.924 ± 0.017 |
| ResNet-50 | 73.9 ± 1.5% | 66.6 ± 2.5% | 0.679 ± 0.026 | 0.913 ± 0.009 | 0.928 ± 0.007 |
| *Ensemble of the 5 fold models* | | | | | |
| **ConvNeXt-Tiny** | **77.0%** | **70.9%** | **0.711** | **0.940** | **0.942** |
| DINOv2 ViT-B/14 | 76.3% | 68.7% | 0.700 | 0.930 | 0.938 |
| ResNet-50 | 74.5% | 67.1% | 0.688 | 0.929 | 0.937 |

**Cross-validation.** Each model was retrained 5 times on different patient-level folds of the
train+val pool, and each of the 15 models was evaluated on the same test set. Balanced accuracy per fold:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| ConvNeXt-Tiny | 0.654 | 0.677 | 0.717 | 0.679 | 0.731 |
| DINOv2 ViT-B/14 | 0.677 | 0.668 | 0.689 | 0.634 | 0.661 |
| ResNet-50 | 0.675 | 0.646 | 0.680 | 0.694 | 0.633 |

- The first single runs of ConvNeXt (0.745) and DINOv2 (0.698) lie at the favourable end, so the
  **CV means are the realistic estimates**.
- Run-to-run spread (±2–3 points) is as large as the gaps between models. In the single runs DINOv2
  seemed 3.6 points better than ResNet-50; with CV they are equal (66.6% each).
- Fold by fold, ConvNeXt has a higher macro AUC than ResNet-50 in 5 of 5 folds and a higher balanced
  accuracy than DINOv2 in 4 of 5 folds: a consistent but modest lead (about one standard deviation).
- Averaging the five models' probabilities (ensemble) improves every backbone.

- **Balanced accuracy** = average recall over the 6 classes; the main metric, because plain
  accuracy rewards predicting the common classes (BCC, ACK).
- **Malignant vs. benign AUC**: probability of BCC + MEL + SCC vs. ACK / NEV / SEK.
- Reference points: random guessing ≈ 17% balanced accuracy; always predicting BCC gives
  37% accuracy.

### Per-class results: ResNet-50

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| ACK | 0.832 | 0.808 | 0.820 | 104 |
| BCC | 0.734 | 0.843 | 0.785 | 121 |
| MEL | 0.833 | 0.714 | 0.769 | 7 |
| NEV | 0.711 | 0.771 | 0.740 | 35 |
| SCC | 0.467 | **0.250** | 0.326 | 28 |
| SEK | 0.667 | 0.588 | 0.625 | 34 |

### Per-class recall: all models

| Model | ACK | BCC | MEL | NEV | SCC | SEK |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.81 | 0.84 | 0.71 | 0.77 | 0.25 | 0.59 |
| DINOv2 ViT-B/14 | 0.84 | 0.83 | 0.71 | 0.77 | 0.21 | 0.82 |
| ConvNeXt-Tiny | 0.82 | 0.84 | 0.71 | 0.80 | **0.50** | 0.79 |
| ResNet-50 ensemble | 0.79 | 0.84 | 0.71 | 0.83 | 0.21 | 0.65 |
| DINOv2 ensemble | 0.83 | 0.84 | 0.71 | 0.80 | 0.18 | 0.77 |
| ConvNeXt ensemble | 0.84 | 0.80 | 0.71 | 0.83 | 0.25 | 0.82 |

ConvNeXt confusion matrix: 39% of SCC cases are predicted as BCC.

### Results by Fitzpatrick skin type (accuracy / balanced accuracy)

| Skin type | Test images | ConvNeXt-Tiny | ResNet-50 |
|---|---|---|---|
| 1 | 16 | 0.750 / 0.449 | 0.812 / 0.474 |
| 2 | 137 | 0.723 / 0.667 | 0.657 / 0.503 |
| 3 | 46 | 0.739 / 0.680 | 0.826 / 0.801 |
| 4 | 10 | 0.800 / 0.889 | 0.500 / 0.278 |
| 5 | 2 | 1.000 / 1.000 | 0.500 / 0.500 |
| unknown | 118 | 0.898 / 0.870 | 0.831 / 0.785 |

## 3. GPU usage during training

Logged once per second with `nvidia-smi` and summarized with `scripts/plot_gpu_log.py`
(charts in `results/`: `gpu_log_resnet50_224_cv.png`, `gpu_log_dinov2_vitb_392_cv.png`,
`gpu_log_convnext_tiny_384_cv.png`; the single-run logs `gpu_log_resnet50.png`, `gpu_log_dinov2.png` too).

Per training run, averaged over the 5 cross-validation runs of each model:

| Model | GPU busy time | Mean GPU use (busy) | Peak memory | Mean power (busy) | Peak power | Energy (busy) | Peak temp. |
|---|---|---|---|---|---|---|---|
| ResNet-50 @ 224 | 1.8 min | 89.4% | 3.7 GB (16%) | 64.6 W | 76.0 W | 1.9 Wh | 81 °C |
| ConvNeXt-Tiny @ 384 | 5.3 min | 96.7% | 6.9 GB (30%) | 68.7 W | 75.8 W | 6.1 Wh | 81 °C |
| DINOv2 ViT-B/14 @ 392 | 14.8 min | 98.4% | 6.6 GB (29%) | 71.0 W | 76.4 W | 17.6 Wh | 82 °C |

All 15 CV runs together: 110 GPU-minutes of training.

- "Busy time" counts seconds with GPU use above 10%; energy (busy) = busy time × mean busy power.
  The `energy_wh` field in the `*_summary.json` files also includes idle time between runs.
- **The GPU was the bottleneck, not data loading**: 90–99% utilization while training.
- **Memory headroom was large**: even the ViT at 392 px used under a third of the 23 GB.
- **Power sat at the card's 72 W limit** during training; short peaks slightly above it are
  normal.
- The L4 has no fan of its own and is cooled by the server; 80 °C under full load is within
  its normal operating range.

## 4. Discussion

1. **ConvNeXt-Tiny is the best model** in the single run, over CV and as an ensemble, while being about
   3× smaller than DINOv2 (27.8M vs 86.1M parameters). Its lead (69.2% vs 66.6% balanced accuracy) is
   about one standard deviation, so it is consistent but not statistically certain; its 5-fold
   ensemble gives the best AUCs (0.940 / 0.942).
2. **The biggest model was not the best.** DINOv2 used about **8× the GPU time and 9× the energy**
   of ResNet-50 (14.8 vs 1.8 min, 17.6 vs 1.9 Wh per run) and about 3× that of ConvNeXt, without being
   more accurate: over CV it ties with ResNet-50 (66.6% balanced accuracy).
   Vision Transformers generally need more data than CNNs; with ~1,600 training images, the
   built-in assumptions of convolutions (local patterns, textures) help more. ViTs also tend
   to need more careful fine-tuning (for example lower learning rates for early layers),
   which was not done here.
3. **Malignant vs. benign is about equally easy for all three models** (AUC 0.92–0.94).
   The differences come from the harder 6-class distinctions.
4. **SCC is the hardest class**: the ensembles find only 18–25% of SCC cases (ConvNeXt's first run
   found 50%, an outlier). SCC is
   typically confused with BCC and ACK, which can look very similar in photos.
5. **Ranking is strong even when the top prediction is wrong**: macro AUC ≥ 0.90 for all
   models, so predicted probabilities are useful for triage (flagging likely cancers for a
   doctor) even where the single top-1 answer is wrong.
6. **Higher resolution plus a modern architecture pays off**: ConvNeXt at 384 px beats
   ResNet-50 at 224 px by about 8 points of balanced accuracy for a moderate extra cost.
7. **Skin type**: types 2 and 3 (the only groups with enough images) perform similarly for
   ConvNeXt (balanced accuracy 0.67 vs 0.68). Types 1 and 4–6 have too few test images
   (16, 10, 2) to draw conclusions, and very dark skin (types 5–6) is almost absent from the
   dataset. The high score of the "unknown" group is most likely due to its different mix
   of diagnoses, not skin tone.

## 5. Limitations

- **Run-to-run variance.** 5-fold CV shows ±2–3 points of balanced accuracy between training runs,
  as large as the gaps between architectures. Five folds give only a rough estimate, so ConvNeXt's
  2.6-point lead is suggestive rather than statistically significant.
- **Possible shortcut.** Grad-CAM shows some photos contain pen circles drawn before biopsy,
  which correlate with cancer diagnoses.
- **Small and imbalanced dataset**: per-class results for MEL (7 test images) and SCC
  (28) are noisy.
- **Limited skin-tone diversity**: 35% of images have no skin type recorded, and types 5–6
  are nearly absent, so fairness across skin tones cannot be properly evaluated.
- **No hyperparameter search**: each model used one reasonable configuration.
- **One dataset, one hospital program (Brazil)**: no test on photos from other sources.
- Not a medical device; results are for research and education only.

## 6. Future work

- More folds or repeated cross-validation with different seeds for tighter confidence intervals.
- Better ViT fine-tuning: layer-wise learning-rate decay, longer training, lower LR.
- Add the clinical metadata (age, lesion region, itching, bleeding, growth) to the image
  model; published work on PAD-UFES-20 reports gains from this.
- External test set (for example Fitzpatrick17k or DDI) to check generalization and skin-tone
  fairness.

## 7. How to reproduce

```bash
bash setup_server.sh                                   # env, data, preprocessing
python -m src.train --config configs/default.yaml      # ConvNeXt-Tiny
python -m src.train --config configs/resnet50.yaml     # ResNet-50
python -m src.train --config configs/dinov2_vitb.yaml  # DINOv2 ViT-B/14
python scripts/compare_runs.py                         # results table
CONFIG=configs/resnet50.yaml bash scripts/run_cv.sh    # 5-fold CV (likewise default.yaml, dinov2_vitb.yaml)
python scripts/plot_gpu_log.py outputs/gpu_log_<model>.csv   # GPU chart + summary
```

Per-run outputs (`outputs/<run>/`): `training_curves.png`, `eval_test/confusion_matrix.png`,
`eval_test/roc_curves.png`, `eval_test/gradcam.png`, `eval_test/metrics.json`,
`eval_test/per_class.json`, `eval_test/by_skin_type.csv`.
