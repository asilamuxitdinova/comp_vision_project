# Results and Discussion

Skin-lesion classification on PAD-UFES-20 with three pretrained backbones, fine-tuned on an
NVIDIA L4 GPU. All numbers are from a single patient-level train/val/test split (seed 42).

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
| **ConvNeXt-Tiny @ 384** | **79.3%** | **74.5%** | **0.738** | **0.933** | 0.930 |
| DINOv2 ViT-B/14 @ 392 | 76.9% | 69.8% | 0.701 | 0.900 | 0.915 |
| ResNet-50 @ 224 | 74.5% | 66.2% | 0.677 | 0.913 | 0.930 |

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

> TODO: add the same table for ConvNeXt-Tiny and DINOv2 from
> `outputs/<run>/eval_test/per_class.json` (or the training log).

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
(charts: `outputs/gpu_log_resnet50.png`, `outputs/gpu_log_dinov2.png`).

| Model | GPU busy time | Mean GPU use (busy) | Peak memory | Mean power (busy) | Peak power | Energy | Peak temp. |
|---|---|---|---|---|---|---|---|
| ResNet-50 @ 224 | 2.4 min | 90.1% | 3.7 GB (16%) | 65.2 W | 77.0 W | 3.9 Wh | 80 °C |
| DINOv2 ViT-B/14 @ 392 | 15.5 min | 98.6% | 6.6 GB (29%) | 71.1 W | 75.6 W | 26.0 Wh | 80 °C |
| ConvNeXt-Tiny @ 384 | *not logged* | | | | | | |

- "Busy time" counts seconds with GPU use above 10%. The total log length (5.1 and 40 min)
  also includes idle time before and after training, so busy time is the fair comparison.
- **The GPU was the bottleneck, not data loading**: 90–99% utilization while training.
- **Memory headroom was large**: even the ViT at 392 px used under a third of the 23 GB.
- **Power sat at the card's 72 W limit** during training; short peaks slightly above it are
  normal.
- The L4 has no fan of its own and is cooled by the server; 80 °C under full load is within
  its normal operating range.

## 4. Discussion

1. **ConvNeXt-Tiny wins on every main metric** while being about 3× smaller than DINOv2
   (≈28M vs ≈86M parameters). It also reached its best validation score earliest (epoch 12).
2. **The biggest model was not the best.** DINOv2 used about **6.5× the GPU time and energy**
   of ResNet-50 (15.5 vs 2.4 min, 26 vs 3.9 Wh) and still scored below ConvNeXt.
   Vision Transformers generally need more data than CNNs; with ~1,600 training images, the
   built-in assumptions of convolutions (local patterns, textures) help more. ViTs also tend
   to need more careful fine-tuning (for example lower learning rates for early layers),
   which was not done here.
3. **Malignant vs. benign is about equally easy for all three models** (AUC 0.92–0.93).
   The differences come from the harder 6-class distinctions.
4. **SCC is the hardest class**: ResNet-50 found only 25% of SCC cases (7 of 28). SCC is
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

- **Single train/test split.** With 329 test images (7 melanomas), differences of 2–3 points
  may be partly chance. ConvNeXt's lead over ResNet-50 (8 points of balanced accuracy) is
  convincing; its lead over DINOv2 (5 points) is less certain.
- **Small and imbalanced dataset**: per-class results for MEL (7 test images) and SCC
  (28) are noisy.
- **Limited skin-tone diversity**: 35% of images have no skin type recorded, and types 5–6
  are nearly absent, so fairness across skin tones cannot be properly evaluated.
- **No hyperparameter search**: each model used one reasonable configuration.
- **One dataset, one hospital program (Brazil)**: no test on photos from other sources.
- Not a medical device; results are for research and education only.

## 6. Future work

- 5-fold cross-validation to report mean ± standard deviation:
  `python -m src.train --config configs/default.yaml --fold k` for k = 0–4.
- Better ViT fine-tuning: layer-wise learning-rate decay, longer training, lower LR.
- Add the clinical metadata (age, lesion region, itching, bleeding, growth) to the image
  model; published work on PAD-UFES-20 reports gains from this.
- External test set (for example Fitzpatrick17k or DDI) to check generalization and skin-tone
  fairness.
- Log GPU usage for ConvNeXt-Tiny to complete the cost comparison.

## 7. How to reproduce

```bash
bash setup_server.sh                                   # env, data, preprocessing
python -m src.train --config configs/default.yaml      # ConvNeXt-Tiny
python -m src.train --config configs/resnet50.yaml     # ResNet-50
python -m src.train --config configs/dinov2_vitb.yaml  # DINOv2 ViT-B/14
python scripts/compare_runs.py                         # results table
python scripts/plot_gpu_log.py outputs/gpu_log_<model>.csv   # GPU chart + summary
```

Per-run outputs (`outputs/<run>/`): `training_curves.png`, `eval_test/confusion_matrix.png`,
`eval_test/roc_curves.png`, `eval_test/gradcam.png`, `eval_test/metrics.json`,
`eval_test/per_class.json`, `eval_test/by_skin_type.csv`.
