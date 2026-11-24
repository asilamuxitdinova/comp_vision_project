// Course project report: compile with `typst compile report/paper.typ report/report.pdf`
// (run from the project folder; figures come from report/figures/, made by report/make_figures.py)

#let todo(body) = text(fill: red, weight: "bold")[#body]

#set document(title: "Patient-Level Evaluation of Transfer Learning for Skin Lesion Classification from Smartphone Photos")
#set page(paper: "a4", margin: (x: 1.6cm, top: 1.7cm, bottom: 1.8cm), numbering: "1")
#set text(font: ("Times New Roman", "New Computer Modern"), size: 9.5pt, lang: "en")
#set par(justify: true, leading: 0.52em, spacing: 0.62em, first-line-indent: 0em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => block(above: 0.95em, below: 0.5em)[
  #set text(size: 10.5pt, weight: "bold"); #counter(heading).display() #h(0.3em) #upper(it.body)]
#show heading.where(level: 2): it => block(above: 0.7em, below: 0.4em)[
  #set text(size: 9.5pt, weight: "bold", style: "italic"); #it.body]
#show figure.caption: set text(size: 8pt)
#show figure: set block(above: 0.7em, below: 0.7em)
#set figure(gap: 0.45em)
#show figure.where(kind: table): set figure.caption(position: top)
#set table(stroke: none, inset: (x: 3pt, y: 1.8pt))
#show table: set text(size: 8pt)
#let hline = table.hline(stroke: 0.5pt)
#show link: set text(fill: rgb("#1f4e9c"))

// ---------------------------------------------------------------- title block
#align(center)[
  #text(size: 15pt, weight: "bold")[Patient-Level Evaluation of Transfer Learning for\ Skin Lesion Classification from Smartphone Photos]
  #v(0.5em)
  #text(size: 10pt)[Asila Muxitdinova (230009) #h(1.5em) Tanja Schmidt (260550)] \
  #text(size: 9pt)[Computer Vision course project, New Uzbekistan University, November 2026] \
  #text(size: 8.5pt)[Code: #link("https://github.com/asilamuxitdinova/comp_vision_project")[github.com/asilamuxitdinova/comp_vision_project]]
]
#v(0.6em)

#pad(x: 1.2cm)[
  #set text(size: 8.8pt)
  #set par(leading: 0.5em)
  *Abstract.* Skin cancer is among the most common cancers worldwide, and early detection depends on access to
  dermatologists that many patients lack. We study whether convolutional and transformer image models can
  classify six skin lesion types from ordinary smartphone photographs. Using the public PAD-UFES-20 dataset
  (2,298 photos, 1,373 patients, all cancers biopsy-proven), we fine-tune three pretrained backbones
  (ResNet-50, ConvNeXt-Tiny and DINOv2 ViT-B/14) under a strict _patient-level_ split, so that no person
  appears in both training and test data. On held-out patients, ConvNeXt-Tiny performs best with
  79.3% accuracy, 74.5% balanced accuracy and 0.933 macro AUC, and all models separate malignant from benign
  lesions with an AUC of about 0.93. The largest model (DINOv2, 86M parameters) is not the most accurate and
  needs 6.5× the GPU time and energy of ResNet-50. We report GPU utilization, memory and energy for every
  model, Grad-CAM explanations, results per Fitzpatrick skin type, and the limitations of the data.
]
#v(0.4em)

#show: rest => columns(2, gutter: 0.55cm, rest)

= Introduction
Skin cancers (basal cell carcinoma, squamous cell carcinoma and melanoma) are among the most frequently diagnosed
cancers, and outcomes depend strongly on early detection. Specialist dermatologists are scarce in many regions,
whereas smartphones are everywhere. A model that flags suspicious lesions from a phone photo could therefore
support triage in primary care and telemedicine. Deep networks have reached dermatologist-level accuracy on
large dermoscopic collections @esteva2017, but dermoscopes are specialist tools. Photos taken with a phone are
harder: lighting, distance, focus and background vary, and public datasets are small.

This project asks a practical question: _how well do current pretrained vision models classify skin lesions
from smartphone photos when evaluated honestly on new patients, and what does each model cost to train?_
Our contributions are: (i) a reproducible pipeline with a leakage-free, patient-level, stratified split;
(ii) a comparison of a classic CNN, a modern CNN and a self-supervised Vision Transformer under one training
recipe; (iii) a measurement of GPU utilization, memory, time and energy for each model; and (iv) an analysis of
errors, explanations (Grad-CAM) and performance across skin types.

= Related Work
Esteva et al. @esteva2017 showed that a CNN trained on 129k clinical images matches dermatologists on
key binary tasks. Public benchmarks such as HAM10000 @tschandl2018 and the ISIC challenges are dominated by
dermoscopic images. PAD-UFES-20 @pacheco2020 was released specifically for smartphone images and comes with
patient metadata. Pacheco and Krohling showed that combining images with clinical metadata improves
classification on this dataset @pacheco2020impact @pacheco2021metablock. Fairness studies found that
dermatology models degrade on darker skin and that most datasets under-represent it
@groh2021 @daneshjou2022. Methodologically, we build on residual networks @he2016, ConvNeXt @liu2022, Vision
Transformers @dosovitskiy2021 with DINOv2 self-supervised pretraining @oquab2023, and Grad-CAM
@selvaraju2017 for visual explanations.

= Dataset
We use *PAD-UFES-20* @pacheco2020, collected by the Dermatological and Surgical Assistance Program of the
Federal University of Espírito Santo (Brazil) with several smartphone models. It contains 2,298 photos of
1,641 lesions from 1,373 patients, each with up to 21 clinical attributes. The dataset is well suited to this
study: it is public under CC BY 4.0, it reflects real clinical practice, *every cancer diagnosis (BCC, SCC,
MEL) is confirmed by biopsy* (58.4% of all images are biopsy-proven), and patient identifiers make
leakage-free splits possible. @tab:data lists the six classes. The download link and a
checksum-verified download script are part of our code release (see _Code and data availability_).

#figure(
  placement: top, scope: "parent",
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right),
    hline,
    table.header([*Code*], [*Diagnosis*], [*Images*], [*Biopsy-proven*], [*Train*], [*Val*], [*Test*]),
    hline,
    [ACK], [Actinic keratosis (pre-cancer)], [730], [24%], [522], [104], [104],
    [BCC], [Basal cell carcinoma], [845], [100%], [603], [121], [121],
    [MEL], [Melanoma], [52], [100%], [37], [8], [7],
    [NEV], [Nevus (benign)], [244], [25%], [174], [35], [35],
    [SCC], [Squamous cell carcinoma], [192], [100%], [136], [28], [28],
    [SEK], [Seborrheic keratosis (benign)], [235], [6%], [168], [33], [34],
    hline,
    [], [*Total images / patients*], [2,298], [58%], [1,640 / 980], [329 / 196], [329 / 197],
    hline,
  ),
  caption: [PAD-UFES-20 classes, biopsy-confirmation rate, and our patient-level split.],
) <tab:data>

The data are strongly imbalanced (845 BCC vs. 52 MEL). The Fitzpatrick skin type is recorded for 65% of
images and is dominated by types 2–3; types 5–6 appear only 11 times. Importantly, skin type is missing
_only_ for non-cancer lesions (61–80% of ACK, NEV and SEK images, but 0% of cancers), which matters for the
fairness analysis in @sec:results.

*Split.* We split by patient with stratification by diagnosis (`StratifiedGroupKFold`) into 70% training,
15% validation and 15% test, and verify that no patient appears in two splits. Images are resized once
(shorter side 512 px). For cross-validation, the training and validation patients are further divided into
five patient-level folds while the test patients stay fixed.

= Method
== Models
We compare three ImageNet-family backbones from the `timm` library @wightman2019, each with a new 6-way
linear head (@tab:models): *ResNet-50* @he2016, a standard CNN baseline at 224 px; *ConvNeXt-Tiny*
@liu2022, a modernized CNN pretrained on ImageNet-22k, at 384 px; and *DINOv2 ViT-B/14* @oquab2023, a
Vision Transformer with self-supervised pretraining on 142M images, at 392 px (a multiple of its 14 px
patch size).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    hline,
    table.header([*Model*], [*Params*], [*Input*], [*Batch*], [*LR*], [*Pretraining*]),
    hline,
    [ResNet-50], [24M], [224], [64], [3e-4], [IN-1k],
    [ConvNeXt-Tiny], [28M], [384], [32], [1e-4], [IN-22k→1k],
    [DINOv2 ViT-B/14], [86M], [392], [16], [2e-5], [LVD-142M (SSL)],
    hline,
  ),
  caption: [Backbones and their main hyperparameters (parameter counts rounded).],
) <tab:models>

== Training recipe
All networks are fine-tuned end-to-end with AdamW @loshchilov2019 (weight decay 0.05, 0.01 for ResNet),
a linear warm-up followed by cosine decay, gradient clipping at 1.0, and up to 30 epochs with early stopping
(patience 7) on *validation balanced accuracy*; the best epoch is kept. To counter class imbalance we use
cross-entropy with class weights proportional to the inverse square root of class frequency and label
smoothing 0.1 @szegedy2016. Augmentation consists of random resized crops (scale 0.6–1.0), horizontal and
vertical flips, rotations up to 30° and color jitter. At test time we average the predictions for the
original and the two flipped images. Training uses bfloat16 mixed precision.

== Evaluation
Because accuracy rewards predicting the frequent classes, our main metric is *balanced accuracy* (mean
per-class recall). We also report accuracy, macro F1, macro one-vs-rest ROC AUC, and a clinically motivated
*malignant-vs-benign AUC*, where the malignant score is the summed probability of BCC, MEL and SCC. We break
results down by Fitzpatrick skin type and visualize decisions with Grad-CAM @selvaraju2017.

== Experimental setup and resources
We had full, dedicated access to a GPU server with an *NVIDIA L4 (24 GB memory, 72 W)*, CUDA 13.2,
PyTorch and timm. This allowed us to fine-tune every network end-to-end at high resolution, compare
three architectures, and repeat training for 5-fold cross-validation, which would not have been feasible on
a laptop CPU. During every run we logged GPU utilization, memory, power and temperature once per second
with `nvidia-smi`.

= Results <sec:results>
== Model comparison
@tab:results and @fig:comparison summarize test performance on the 329 images of 197 unseen
patients. ConvNeXt-Tiny is the best model on every multi-class metric, improving balanced accuracy by 8.2
points over ResNet-50 and 4.7 points over DINOv2. For context, random guessing gives about 17% balanced
accuracy and always predicting BCC gives 37% accuracy. All three models reach a malignant-vs-benign AUC of
0.92–0.93: distinguishing cancer from non-cancer is about equally easy for all of them, and the differences
arise in the finer six-way distinctions.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    hline,
    table.header([*Model*], [*Acc.*], [*Bal. acc.*], [*Macro F1*], [*AUC*], [*Mal. AUC*]),
    hline,
    [ResNet-50], [0.745], [0.662], [0.677], [0.913], [*0.930*],
    [DINOv2 ViT-B/14], [0.769], [0.698], [0.701], [0.900], [0.915],
    [ConvNeXt-Tiny], [*0.793*], [*0.745*], [*0.738*], [*0.933*], [0.930],
    hline,
    [ConvNeXt, 5-fold (mean±sd)], table.cell(colspan: 5, align: center)[#todo[cross-validation results pending]],
    hline,
  ),
  caption: [Test results on held-out patients. Mal. AUC: malignant (BCC, MEL, SCC) vs. benign.],
) <tab:results>

#figure(
  image("figures/comparison.pdf", width: 100%),
  caption: [Test balanced accuracy, macro F1 and macro AUC of the three backbones.],
) <fig:comparison>

== Per-class behaviour
#todo[Per-class table for ConvNeXt pending.] For ResNet-50, per-class recall is 0.81 (ACK), 0.84 (BCC),
0.71 (MEL, 5 of 7), 0.77 (NEV), 0.59 (SEK) and only *0.25 for SCC* (7 of 28). SCC is mostly confused with
BCC and ACK, which can look alike in photos and share the same body regions; this is the hardest
distinction in the dataset.

== Skin type
For ConvNeXt, balanced accuracy is similar for the two skin types with enough test images, type 2 (137
images, 0.667) and type 3 (46 images, 0.680). Types 1, 4 and 5 have only 16, 10 and 2 test images, too few for
conclusions, and type 6 is absent from the test set. The "unknown" group scores highest (0.870), but
because skin type is missing only for non-cancer lesions, this group contains no cancers at all and
measures an easier task, not a skin-tone effect.

== Computational cost
@tab:gpu reports GPU usage. All runs kept the GPU at 90–99% utilization, so data loading was not a
bottleneck, and memory stayed well below the 24 GB available. DINOv2 needed *6.5× the GPU time and
6.7× the energy* of ResNet-50 while being less accurate than ConvNeXt-Tiny.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    hline,
    table.header([*Model*], [*GPU time*], [*Util.*], [*Peak mem.*], [*Power*], [*Energy*]),
    hline,
    [ResNet-50], [2.4 min], [90%], [3.7 GB], [65 W], [3.9 Wh],
    [DINOv2 ViT-B/14], [15.5 min], [99%], [6.6 GB], [71 W], [26.0 Wh],
    [ConvNeXt-Tiny], table.cell(colspan: 5, align: center)[#todo[pending (from cross-validation log)]],
    hline,
  ),
  caption: [GPU cost of one training run on the NVIDIA L4 (busy time with utilization above 10%; mean
  utilization and power while busy).],
) <tab:gpu>

= Discussion and Limitations
*Bigger is not better on small data.* The 86M-parameter ViT was outperformed by a 28M-parameter CNN.
Transformers have weaker built-in assumptions about images (locality, translation equivariance) and usually
need more data; with about 1,600 training images, the convolutional prior helps. ViTs also tend to benefit
from more careful fine-tuning, such as layer-wise learning-rate decay, which we did not tune.

*Resolution and modern design pay off.* ConvNeXt at 384 px clearly beats ResNet-50 at 224 px for a modest
extra cost, consistent with fine lesion texture being important.

*Limitations.* (1) The test set is small (329 images, 7 melanomas), so differences of 2–3 points, and all
per-class melanoma numbers, are uncertain; ConvNeXt's 8-point lead over ResNet-50 is robust, its lead over
DINOv2 less so. (2) Very dark skin (types 5–6) is nearly absent and skin type is missing for 35% of images,
so fairness across skin tones cannot be properly assessed. (3) All images come from one program in one
country; we did not test on external data. (4) Each model used a single hyperparameter setting, and we used
images only, although clinical metadata is known to help @pacheco2020impact. (5) The system is a research
prototype, not a medical device.

= Conclusion
Under a strict patient-level protocol, a fine-tuned ConvNeXt-Tiny classifies six skin lesion types from
smartphone photos with 74.5% balanced accuracy and separates malignant from benign lesions with an AUC of
0.93, outperforming both a classic CNN and a much larger Vision Transformer at a fraction of the latter's
compute. Future work: adding clinical metadata, layer-wise fine-tuning for ViTs, and external validation on
more diverse skin types (e.g., Fitzpatrick17k @groh2021 and DDI @daneshjou2022).

#[#set par(justify: false)
*Code and data availability.* Code, configurations, results and GPU logs:
#link("https://github.com/asilamuxitdinova/comp_vision_project")[github.com/asilamuxitdinova/comp_vision_project].
Dataset: PAD-UFES-20, #link("https://data.mendeley.com/datasets/zr7vgbcyr2")[data.mendeley.com/datasets/zr7vgbcyr2]
(CC BY 4.0); our script `scripts/download_data.py` downloads it and verifies it against the official checksum.]

#set text(size: 7.5pt)
#show bibliography: set heading(numbering: none)
#bibliography("refs.yml", title: "References", style: "ieee")
