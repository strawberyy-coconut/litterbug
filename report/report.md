# litterbug — Trash Detection & Instance Segmentation on a Conveyor Belt

> **Working notes, not a deliverable.** This file collects measurements, caveats and reasoning as
> source material for the graded report. It is written for the author, not for a reader, and is not
> bound by the report's page limit. The graded report is a separate document, written by hand in
> Typst, and this text is not reproduced in it.

Instance segmentation and detection of waste objects on a conveyor belt, built on the **BUU Waste
Occlusion** dataset with Ultralytics **YOLO26**.

**Status:** all required components complete — segmentation and detection trained, evaluated on `valid`
and on a held-out `test` split, with per-instance error analysis, dataset EDA, commented error
examples and tracked video inference.

---

## 1. Headline results

Segmentation, `test` split — 200 images / 2,507 instances, evaluated **once**, after every model
decision had already been made:

| Metric | Box | Mask |
| --- | --- | --- |
| mAP50-95 | **0.837** | **0.796** |
| mAP50 | 0.954 | 0.955 |
| Precision | 0.939 | 0.944 |
| Recall | 0.901 | 0.905 |

Mean IoU over matched pairs (§7.6): **0.901** for masks against **0.924** for boxes.

Inference: **8.6 ms/image** on an RTX 3060 Laptop (≈ 117 FPS), excluding pre/post-processing.

`valid` carried the same numbers one to two points higher — box 0.857 / mask 0.814 mAP50-95. That gap is
the honest measure of how much the iteration split flattered the model, and §5.4 shows nearly all of it
sits in the smallest size quartile.

The mask head trails the box head by **0.041 mAP50-95** on `test` (0.043 on `valid`). That is the cost
of predicting outlines rather than rectangles within a single model.

The detection baseline was trained on the same splits with the same schedule, to ask whether the mask
task costs anything at all:

| Split | Detection only | Segmentation (box head) | Difference |
| --- | --- | --- | --- |
| `valid` | 0.8583 | 0.8566 | 0.0017 |
| `test` | **0.8376** | **0.8372** | **0.0004** |

**The difference is nothing on both splits** — and smaller on the held-out one. Segmentation delivers
masks at 0.796 mask mAP50-95 on unseen data while giving up nothing measurable in detection quality.
See §6.

---

## 2. Problem

Waste arrives on a conveyor belt in dense, overlapping batches. The dataset's annotations trace
**only the visible boundary** of each object, and objects under roughly 30% visible are
deliberately excluded. Two consequences follow, and they shape the whole evaluation:

- Overlap is a labelling convention, not an error. Partial objects are correct data.
- Success is not just "find the object" but "trace where it stops being visible" — a genuinely
  harder target than detection, and the one this project is actually about.

Segmentation is the product. Detection is trained as a **comparison baseline on the same split with
the same schedule**, so the report can attribute any difference to the head rather than to the recipe.

---

## 3. Dataset

**Source:** [BUU Waste Occlusion Dataset (BUU-WOD)](https://www.kaggle.com/datasets/visionlab1buu/buu-waste-occlusion-dataset),
VisionLab, Burapha University. Licensed **CC BY 4.0**. Our copy was exported from Roboflow Universe —
same dataset, same licence. Full citation and licence terms in §11.

BUU Waste Occlusion, 2,000 images at 640×640, pre-split and used as-is.

| Split | Images | Instances | Instances/image |
| --- | --- | --- | --- |
| train | 1,400 | 17,400 | 12.43 |
| valid | 400 | 4,941 | 12.35 |
| test | 200 | 2,507 | 12.54 |

Classes, in fixed order: `0 Glass`, `1 Metal`, `2 Paper`, `3 Plastic`. The order is a contract with
the annotations and is declared exactly once in `src/common/constants.py`.

A standalone validator confirmed image/label pairing, polygon validity, coordinate ranges and class
balance before training, and continues to reproduce the baseline counts above exactly.

`valid` is used for iteration. **`test` has not been touched** — it is reserved for a single final
evaluation, so the reported figures are not the ones any decision was made on.

### 3.1 Exploratory analysis

Produced by `uv run litterbug eda`, which reads the annotations and samples the images rather than
transcribing figures from an earlier run. Output: `runs/dataset-eda/dataset_eda.json`.

| Split | Images | Instances | Inst/image | No instances | Malformed rows | Imbalance |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1,400 | 17,400 | 12.43 | 0 | 0 | 1.102 |
| valid | 400 | 4,941 | 12.35 | 0 | 0 | 1.074 |
| test | 200 | 2,507 | 12.54 | 0 | 0 | 1.172 |

Instances per class, with each class's share of its split:

| Split | Glass | Metal | Paper | Plastic |
| --- | --- | --- | --- | --- |
| train | 4,197 (24.1 %) | 4,256 (24.5 %) | 4,623 (26.6 %) | 4,324 (24.9 %) |
| valid | 1,203 (24.3 %) | 1,198 (24.2 %) | 1,287 (26.1 %) | 1,253 (25.4 %) |
| test | 584 (23.3 %) | 577 (23.0 %) | 676 (27.0 %) | 670 (26.7 %) |

**The dataset is balanced, and Paper is its largest class in every split.** The largest-to-smallest
ratio is 1.07–1.17, and Paper's share is the highest of the four. That closes a hypothesis §7.4 could
not close on its own: Paper's poor recall cannot be a class-imbalance artefact, because Paper is not
the scarce class — it is the most numerous one. §7.4 ruled out size and confusion; the class counts
rule out imbalance independently.

No image on any split is unlabelled, and no label row is malformed.

**Geometry is uniform.** Every sampled image is 640×640, three-channel, single resolution — as
expected from the provider's stretch resize (§11).

**Lighting is essentially invariant.** Mean brightness over a 200-image sample per split:

| Split | p10 | Median | p90 |
| --- | --- | --- | --- |
| train | 71.1 | 75.8 | 80.7 |
| valid | 72.1 | 76.9 | 81.4 |
| test | 71.3 | 76.1 | 80.9 |

On a 0–255 scale the p10–p90 spread is about **ten levels**, and no image falls below 60 or above
180. There is effectively no exposure variation: the dataset is dim (median ≈ 30 % of range) and
almost perfectly uniform.

That uniformity has a consequence worth carrying forward. The model has never seen a scene lit
differently from this one — a far more precise statement than "the composition differs", and one that
fits the transfer failures in §8, where detections latch onto colour and structure rather than
material. A model with no lighting variation in its training data has no lighting invariance to draw
on.

Sharpness (variance of the Laplacian) measures 675–699 across the splits. That is a usable baseline
for comparing footage sharpness, though no such comparison is made here.

---

## 4. Method

### 4.1 Architecture

`yolo26s-seg` — the Ultralytics YOLO26 segmentation family, "s" size. Fine-tuned from COCO-pretrained
weights; 830 of 844 parameter tensors transferred.

| | Value |
| --- | --- |
| Layers | 309 (136 fused) |
| Parameters | 11,437,172 (10,366,888 fused) |
| GFLOPs | 37.3 (34.3 fused) |
| Head | `Segment26` |

The model name lives in exactly one place (`MODELS` in `src/common/constants.py`), so changing
architecture is a one-line edit. A YOLO11 pair is recorded adjacent as `MODELS_FALLBACK`; the Day-1
gate passed, so it was not needed.

### 4.2 The Day-1 architecture gate — and a warning about short smoke tests

Before committing to the architecture, a single epoch was run to check that the segmentation head
produced non-degenerate masks rather than whole-image blobs. It did, so YOLO26 was kept. Wall-clock
cost of the gate: ~2 minutes.

**The gate's metrics were worthless, and it would be easy to mistake them for evidence.** The gate
reported mask mAP50-95 of **0.159** at epoch 1. The real 100-epoch run reported **0.564** at epoch 1 —
same model, same data, same seed.

The cause is that `warmup_epochs=3` and `close_mosaic=10` are configured as absolute epoch counts
while the gate ran with `epochs=1`. Both settings become degenerate in a one-epoch run: the learning
rate never completes warmup and mosaic augmentation is disabled from the first batch. "Epoch 1 of a
1-epoch schedule" is therefore a different training regime from "epoch 1 of a 100-epoch schedule",
and the two cannot be compared.

The lesson recorded for the sprint: a short smoke test is valid for checking that the *plumbing*
works and for measuring *throughput*, but never for reading *quality*.

### 4.3 Schedule

Frozen before training and not tuned. Values live in `src/common/config.py`.

| Setting | Value |
| --- | --- |
| Epochs | 100 (completed, no early stop) |
| Patience | 20 |
| Image size | 640 |
| Batch | 8 |
| Seed | 42 |
| AMP / cosine LR | enabled |
| `copy_paste` | 0.1 |
| `overlap_mask` / `mask_ratio` | True / 4 |
| `close_mosaic` | 10 |
| Optimizer | AdamW, lr 0.00125, momentum 0.9 (auto-selected) |

Batch size is derived from VRAM rather than chosen: the pipeline reads the visible device and applies
fixed thresholds (≥11 GiB → 16, ≥5 GiB → 8). On the 5.7 GiB RTX 3060 that resolves to 8, which peaked
at **3.9 GB** — comfortable headroom, no OOM.

`copy_paste` is enabled specifically because this dataset's difficulty is occlusion.

### 4.4 Hardware and cost

RTX 3060 Laptop, 5.7 GiB VRAM. Training completed **100 epochs in ≈ 2.7 h** of GPU time, with a
steady-state throughput of **86.6 s/epoch** measured across epochs 40–100.

Note when reading the recorded timings: the run was interrupted and resumed once (the checkpoint's
`args.yaml` records the resume, and the internal timer resets on resume — the counter drops from
1158 s to 107 s at epoch 11). The ≈2.7 h figure is the sum of both sessions; the run's own final
counter shows only the second.

### 4.5 Reproducibility

- All randomness seeded: `torch.manual_seed`, `np.random.seed`, `random.seed`, `PYTHONHASHSEED`, plus
  Ultralytics `seed=42`.
- Every run prints its fully resolved configuration and environment block before touching the GPU, and
  Ultralytics writes the same arguments to `args.yaml` in the run directory.
- `--dry-run` resolves and prints the configuration without spending GPU time.

---

## 5. Results — segmentation

### 5.1 Learning curve (`valid`)

| Epoch | Wall (s) | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| 1 | 106 | 0.766 | 0.589 | 0.765 | 0.564 |
| 10 | 1158 | 0.903 | 0.740 | 0.900 | 0.712 |
| 20 | — | 0.923 | 0.769 | 0.921 | 0.745 |
| 40 | 3430 | 0.945 | 0.813 | 0.946 | 0.784 |
| 60 | 5428 | 0.956 | 0.839 | 0.955 | 0.801 |
| 80 | 7210 | 0.962 | 0.853 | 0.962 | 0.811 |
| **89** | 7981 | **0.964** | **0.857** | **0.963** | **0.814** |
| 100 | 8629 | 0.962 | 0.855 | 0.961 | 0.813 |

Best checkpoint is epoch **89** (mask mAP50-95 0.8145). Epoch 100 is within 0.0015 of it, so the
model had converged rather than being cut off by the schedule, and patience never triggered.

The curve shape is worth noting for the report's argument: over half the total quality arrives in the
first 10 epochs, and the final 60 epochs add ~0.03 mAP50-95.

### 5.2 Per-class results

| Class | Box mAP50-95 | Mask mAP50-95 | Box − Mask |
| --- | --- | --- | --- |
| Glass | 0.892 | 0.844 | 0.048 |
| Metal | 0.881 | 0.832 | 0.049 |
| Plastic | 0.861 | 0.819 | 0.042 |
| **Paper** | **0.791** | **0.763** | 0.028 |

**Paper is the weakest class; Glass is the strongest.** Errors are not concentrated in any one class
pair — §7 shows the largest cross-class confusion in the entire validation set is 31 instances (Paper
predicted as Metal). What the aggregate numbers show:

- Glass and Plastic are **not** the weak classes. Glass is the best-scoring class in both heads.
- Paper is weakest by ~0.06 mAP50-95 in both heads — a gap larger than the box/mask difference.

Two caveats before this becomes a conclusion:

1. **Per-class mAP cannot distinguish confusion from missed detections.** That question is settled in
   §7: Paper's errors are dominated by misses, and confusion between any two classes is negligible.
2. At the 1-epoch gate, Glass was the *worst* class (mask 0.059), which is consistent with an
   unconverged head rather than any property of Glass. Early-epoch per-class rankings should not be
   trusted.

### 5.3 Box head versus mask head, within the segmentation model

Mask mAP50-95 is 0.043 below box mAP50-95, and the ranking of classes is identical in both heads. The
mask head adds outline precision at modest cost, and it does not reorder class difficulty.

§7.6 corroborates this from a second, independent metric: mean IoU over matched pairs is **0.907** for
masks against **0.929** for boxes — a 0.022 gap, the same direction and the same order of magnitude as
the mAP difference.

**This is a within-model comparison.** It shows what the mask head costs relative to the box head of
the *same* network. It is **not** a segmentation-versus-detection result — that comparison requires
the detection baseline and is covered in §6.

### 5.4 Test split

`test` was evaluated once, after training and all iteration stopped.

| Metric | `valid` | `test` | Change |
| --- | --- | --- | --- |
| Box mAP50-95 | 0.857 | 0.837 | −0.020 |
| Mask mAP50-95 | 0.814 | 0.796 | −0.018 |
| mAP50 | 0.963 | 0.954 | −0.009 |
| Precision (box) | 0.948 | 0.939 | −0.009 |
| Recall (box) | 0.923 | 0.901 | −0.022 |

**The generalisation gap is small and the model is not overfitted to `valid`** — about two points of
mAP50-95, well inside the ±0.01 seed noise the report already declines to interpret (§6).

Where the drop sits is more interesting than its size:

| Size quartile | `valid` recall | `test` recall | Change |
| --- | --- | --- | --- |
| q1 — smallest | 0.8947 | **0.8262** | **−0.069** |
| q2 | 0.9789 | 0.9744 | −0.005 |
| q3 | 0.9854 | 0.9904 | +0.005 |
| q4 — largest | 0.9919 | 0.9904 | −0.002 |

**The entire generalisation gap lives in the smallest size quartile.** Recall on q2–q4 is flat to
within half a point; q1 falls by nearly seven. The model has not degraded uniformly on unseen data — it
has lost small objects specifically, which is the same failure §7.3 identified as the dominant error
source on `valid`, now amplified on data the model had never seen.

---

## 6. Results — detection baseline

Same split, same 100-epoch schedule, same seed, same `s` size, so the head and the training objective
are the only things that differ. Detection completed 100 epochs in **1.80 h** (against 2.40 h of
recorded training time for segmentation — it is a smaller model and carries no mask loss) and peaked at
roughly 2.5 GB VRAM against 3.9 GB, so no resource limit can explain the outcome.

| Split | Model | Box mAP50 | Box mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| `valid` | Detection only | 0.9633 | 0.8583 | 0.939 | 0.931 |
| `valid` | Segmentation (box head) | 0.9634 | 0.8566 | 0.948 | 0.923 |
| **`test`** | Detection only | 0.9557 | **0.8376** | 0.944 | 0.913 |
| **`test`** | Segmentation (box head) | 0.9542 | **0.8372** | 0.939 | 0.901 |

**Difference in box mAP50-95: 0.0017 on `valid`, 0.0004 on `test` — in both cases, none.**

The held-out split repeats the `valid` result and sharpens it: the gap is four times smaller on data
neither model has seen. Whatever the mask head costs, it is not detectably visible in the box head.

0.0017 is roughly a fifth of one percent of the metric's range, from a single seed. It is not a
detectable effect. The mask task did not cost detection accuracy; nor did it measurably improve it.

Per-class differences give the same verdict — mixed signs, every one inside ±0.01:

| Class | Detection | Segmentation (box) | Difference |
| --- | --- | --- | --- |
| Glass | 0.895 | 0.892 | +0.003 |
| Metal | 0.873 | 0.881 | −0.008 |
| Paper | 0.800 | 0.791 | +0.009 |
| Plastic | 0.864 | 0.861 | +0.003 |

No consistent direction and no class where the effect is systematic: this is the signature of noise,
not of a real difference.

### What this means

The mask task was **free**. Segmentation yields masks at 0.814 mask mAP50-95 while surrendering nothing
measurable in detection quality — its box head is statistically indistinguishable from a model trained
on boxes alone.

This is the strongest available justification for the segmentation choice. The honest counter-challenge
— "could we have got the same boxes from a model a third cheaper?" — is answered: yes for boxes, and
those boxes would have come with no shapes.

One nuance worth noting: precision and recall traded against each other. Segmentation is slightly more
precise (0.948 vs 0.939) and slightly less sensitive (0.923 vs 0.931). Those roughly cancel, which is
why the aggregate lands in the same place — a useful reminder that a single mAP number hides two
countervailing behaviours.

**Caveat.** One seed per model. Differences below ~0.01 should not be read as real, which unfortunately
includes every row of the per-class table above. Several seeds would be needed to tighten this, and
that is outside the sprint's budget.

---

## 7. Error analysis

All numbers below come from the `analyze` command, which writes one file per split and per matching
mode:

```bash
uv run litterbug analyze --weights runs/litterbug-segment-yolo26s-seg/weights/best.pt --split val
uv run litterbug analyze --weights runs/litterbug-segment-yolo26s-seg/weights/best.pt --split test
```

- `runs/litterbug-segment-yolo26s-seg-val-analysis/error_analysis.json` — §7.1–§7.7
- `runs/litterbug-segment-yolo26s-seg-val-box-analysis/error_analysis.json` — the box column of §7.6
- `runs/litterbug-segment-yolo26s-seg-test-analysis/error_analysis.json` — §7.8
- `runs/litterbug-segment-yolo26s-seg-test-box-analysis/error_analysis.json` — the box column of §7.8

Add `--iou-mode box` to either split for the box-matching run. The split and the matching mode are
part of the run name because they are not part of the path: before that, analysing `test` silently
overwrote the `valid` analysis, and a box run overwrote the mask run.

### 7.1 How the numbers were derived

Ultralytics computes per-instance matches internally but does not expose them, so the confusion matrix
normally exists only as a PNG. This analysis re-derives the matching from scratch.

| Setting | Value |
| --- | --- |
| Matching criterion | mask IoU ≥ 0.5 |
| Confidence floor | 0.34 — the F1-optimal point on the validation curve, not the 0.25 default |
| Assignment | predictions by descending confidence, greedy one-to-one, **class-agnostic** |

Class-agnostic matching is deliberate: a Paper prediction landing on a Plastic object must register as
a *confusion*, not as a miss plus a false positive. Conventions differ here, so the choice is stated
rather than assumed.

Result: **4,757 matched** of 5,125 predictions against 4,941 ground-truth instances — recall **0.963**,
precision **0.928**.

### 7.2 Confusion between classes is negligible

Across 4,941 instances, the largest off-diagonal cell in the whole matrix is **31** — Paper predicted as
Metal — and Glass↔Plastic confusion specifically accounts for **12 errors (0.24 %)**.

| truth ↓ / predicted → | Glass | Metal | Paper | Plastic | missed |
| --- | --- | --- | --- | --- | --- |
| **Glass** | 1168 | 4 | 1 | **10** | 20 |
| **Metal** | 3 | 1151 | 6 | 6 | 32 |
| **Paper** | 2 | 31 | 1139 | 19 | **96** |
| **Plastic** | **2** | 21 | 30 | 1164 | 36 |

Glass→Plastic is 10 and Plastic→Glass is 2, against roughly 1,200 instances of each — small enough that
no class pair is a meaningful error source in this model. The error budget is dominated by misses, not
by confusions.

### 7.3 Size is the dominant driver of misses

| Size quartile (ground-truth mask px) | Instances | Recall |
| --- | --- | --- |
| q1 — 15 to 3,600 | 1,234 | **0.895** |
| q2 — 3,600 to 5,713 | 1,235 | 0.979 |
| q3 — 5,713 to 9,030 | 1,236 | 0.985 |
| q4 — 9,030 to 59,264 | 1,236 | **0.992** |

A ten-point recall gap between the smallest and largest quartile, almost entirely concentrated in q1.
**Small instances are the single largest source of error in this model** — larger than every class
confusion combined.

### 7.4 Paper is intrinsically hard, not merely small

Paper is the weakest class (recall 0.885). The obvious explanation is that paper objects are small. The
data says otherwise. Cross-tabulating recall by class *within* the same size quartiles:

| Class | Instances | Median size (px) | q1 | q2 | q3 | q4 |
| --- | --- | --- | --- | --- | --- | --- |
| Glass | 1,203 | 4,982 | 0.953 | 0.990 | 0.997 | 1.000 |
| Metal | 1,198 | 4,570 | 0.936 | 0.989 | 0.996 | 0.994 |
| Plastic | 1,253 | 7,914 | 0.874 | 0.970 | 0.982 | 0.998 |
| **Paper** | 1,287 | **6,116** | **0.798** | **0.957** | **0.967** | **0.980** |

Two things follow.

1. **Paper's instances are not the smallest.** Its median (6,116 px) is larger than Glass (4,982) and
   Metal (4,570). The weakness is not a size artefact.
2. **Paper is the worst class in every size quartile**, by 6–12 points in the smallest bucket. Size
   degrades every class; it does not explain Paper.

Paper's errors also skew towards **misses rather than confusion** — 96 missed against 52 confused. And
its largest confusion is with **Metal (31)**, not Plastic (19).

### 7.5 On occlusion — what this analysis cannot claim

The dataset is named for occlusion, but **true occlusion is not measurable from it.** Annotations trace
only visible boundaries, and objects below ~30 % visibility were excluded rather than annotated with
their hidden geometry. The hidden fraction of an object cannot be recovered.

Two proxies were computed instead:

| Proxy | q1 | q4 | Verdict |
| --- | --- | --- | --- |
| Mask area (size) | 0.895 | 0.992 | Strong and monotonic — see §7.3 |
| Solidity (area ÷ convex hull) | 0.924 | 0.963 | Weak and **non-monotonic** |

Solidity is confounded: naturally irregular materials (crumpled paper, plastic film) score low whether
or not they are occluded. Its non-monotonicity — q3 scores 0.989 while perfectly convex q4 scores
0.963 — is consistent with it tracking material rather than occlusion. **It must not be read as an
occlusion result.** No occlusion-stratified claim is made anywhere in this report.

The likeliest place for a reader to read occlusion into these results is §8, and it should not be read
there either. The facility clips do contain heavy occlusion — hands, overlapping objects, machinery —
but occlusion is confounded with viewpoint, human presence and scene composition, and the experiment
does not separate them. §8 is a composition finding, not an occlusion measurement.

### 7.6 Mask quality — IoU

Recall and precision say whether an object was *found*. IoU says whether it was *traced correctly*, and
the brief asks for it as a metric in its own right.

Across the 4,757 matched pairs, mean IoU is **0.907** (median 0.928, p10 0.839, p90 0.960), and
**96 % of pairs score ≥ 0.75**. When the model finds an object, it outlines it well.

Running the same analysis with `--iou-mode box` gives a like-for-like comparison:

| | Mean IoU | Median | p10 | p90 |
| --- | --- | --- | --- | --- |
| Box | 0.9291 | 0.9519 | 0.8628 | 0.9789 |
| **Mask** | **0.9074** | 0.9278 | 0.8391 | 0.9597 |

**The mask head gives up 0.022 IoU.** That is a second, independent measurement of the effect §5.3
reports as 0.043 mAP50-95 — two metrics that measure different things, agreeing in direction and
magnitude. It is the corroboration §5.3 previously lacked.

Per class, mask IoU over true positives only. A cross-class match measures how well a Plastic
prediction happens to cover a Paper object, which says nothing about the predicted class's mask
quality, so those pairs are excluded rather than quietly averaged in:

| Class | Pairs | Mask IoU | Box IoU | Gap |
| --- | --- | --- | --- | --- |
| Glass | 1,168 | 0.9129 | 0.9374 | 0.0245 |
| Metal | 1,151 | 0.9143 | 0.9363 | 0.0220 |
| Plastic | 1,164 | 0.9113 | 0.9345 | 0.0232 |
| **Paper** | 1,139 | **0.8953** | **0.9120** | 0.0167 |

**Paper is again the worst class — in both modes.** §7.4 established that Paper is the weakest class in
recall at every size quartile, with size and confusion ruled out as explanations. IoU agrees
independently: Paper's masks fit worst and its boxes fit worst. Two metrics measuring different things
both single out the same class, which makes "Paper is intrinsically hard" materially harder to dismiss
as an artefact of one metric.

IoU also degrades with size, in both modes:

| Size quartile | Mask IoU | Box IoU | Gap |
| --- | --- | --- | --- |
| q1 — smallest | 0.8724 | 0.9043 | **0.0319** |
| q2 | 0.9118 | 0.9352 | 0.0234 |
| q3 | 0.9189 | 0.9373 | 0.0184 |
| q4 — largest | 0.9231 | 0.9373 | **0.0142** |

Small instances are penalised **twice over**: missed more often (recall 0.895 against 0.992, §7.3) *and*
outlined less accurately when found. The mask head's cost is concentrated there too — it gives up 0.032
IoU on the smallest quartile against 0.014 on the largest, roughly twice as heavy on small objects.

**Caveat on the comparison.** The two runs match under different criteria — mask IoU ≥ 0.5 versus box
IoU ≥ 0.5 — so the pair sets are not identical: 4,757 mask-matched against 4,747 box-matched. Recall and
precision differ marginally for the same reason (0.9628 / 0.9282 against 0.9607 / 0.9262). The
difference is immaterial to the conclusion, but the two columns are not computed over the same pairs.

### 7.7 Commented examples

`uv run litterbug examples --weights <run>/weights/best.pt --split val` renders the clearest examples
of each error type and writes measured commentary beside them:

- figures — `report/figures/examples-val/figures/`
- commentary — `report/figures/examples-val/examples.md`

Each figure is two panels: ground truth on the left, predictions on the right. **Red** outlines mark
ground-truth instances the model missed; **magenta** marks predictions that matched nothing. Class
fills use the dataset's own colour convention, so the panels can be read against each other directly.

#### Paper dominates every error type

Every confusion example has recall **1.00** — the model found everything and still mislabelled some —
and **Paper appears in all six confusion pairs across the three figures** (Paper→Metal ×3, Metal→Paper,
Paper→Plastic ×2). The same bias runs through the other two kinds: Paper accounts for 4 of the 4
misses in one figure and 7 of 10 across the three, and dominates the unmatched predictions.

That is a third, independent line on Paper. §7.4 ruled out size and confusion by cross-tabulation,
§7.6 found Paper worst on IoU, and now the raw example images single out the same class. Three
methods measuring different things converge, which is why §7.4's conclusion is stated with confidence
rather than as one metric's opinion.

#### False positives are two different failures

Reviewing the figures exposed something the aggregate counts conceal. **Matching is one-to-one**, so a
second prediction sitting on an object that was already detected cannot match and is scored as a false
positive — even though it covers a real object. Those are duplicates, not hallucinations, and they are
not the same problem as predicting on bare belt.

Separating them across the whole split:

| Category | Count | Share |
| --- | --- | --- |
| Matched | 4,757 | — |
| Missed instances | 184 | — |
| Confused (matched, wrong class) | 135 | — |
| False positives — duplicates | **156** | 42 % of FPs |
| False positives — over background | **212** | 58 % of FPs |

**42 % of the model's false positives are duplicate detections.** Precision is therefore not one
behaviour described by one number: 0.928 is 212 genuine background errors plus 156 redundant masks,
and the two have different causes and would need different remedies.

As arithmetic rather than a claim — suppressing duplicates and nothing else, with recall held fixed,
would move precision from 0.928 to 0.957 (4,757 / 4,969). No simple threshold achieves that cleanly,
so it is an upper bound on what duplicate suppression could buy, not a result.

The `analyze` output cannot make this distinction on its own: an unmatched prediction is an unmatched
prediction, and only re-examining the overlap against *all* instances, matched or not, separates the
two. That is why this is reported here rather than in §7.2.

### 7.8 Test split

The same analysis re-run on the held-out split, which §7.1–§7.7 never touched.

| Metric | `valid` | `test` |
| --- | --- | --- |
| Ground-truth instances | 4,941 | 2,507 |
| Predictions | 5,125 | 2,548 |
| Matched | 4,757 | 2,370 |
| Recall | 0.9628 | 0.9454 |
| Precision | 0.9282 | 0.9301 |
| Mean IoU — mask | 0.9074 | 0.9010 |
| Mean IoU — box | 0.9291 | 0.9244 |

Recall falls 1.7 points and mask IoU 0.6; precision is fractionally *higher*. Nothing here suggests the
`valid` figures were structurally optimistic.

**The per-class findings replicate.**

| Class | `valid` recall | `test` recall | `valid` IoU | `test` IoU |
| --- | --- | --- | --- | --- |
| Glass | 0.9709 | 0.9846 | 0.9129 | 0.9065 |
| Metal | 0.9608 | 0.9376 | 0.9143 | 0.9066 |
| Plastic | 0.9290 | 0.9045 | 0.9113 | 0.9083 |
| **Paper** | **0.8850** | **0.8432** | **0.8953** | **0.8876** |

**Paper is the weakest class on `test` in both recall and IoU**, exactly as on `valid`. This matters
more than the other replications: "Paper is intrinsically hard" was the report's most substantive
interpretive claim, and it rested entirely on the iteration split. It now holds on data the model never
saw — a third independent line after §7.4's cross-tabulation and §7.6's IoU. Glass, meanwhile,
*improves* on test (0.9709 → 0.9846), so the class differences are not a uniform shrinkage toward the
mean.

The one figure that moves sharply is the smallest size quartile — q1 recall 0.8947 → 0.8262 while q2–q4
hold steady. §5.4 carries that table; the short version is that the entire generalisation gap is a
small-object gap.

**The false-positive decomposition replicates almost exactly.** On `valid`, 368 false positives split
into 156 duplicates and 212 over background — 42 % / 58 %. On `test`, 178 false positives split into
**75 and 103 — 42 % / 58 %.** The same proportions, on held-out data. Duplicate detections are a stable
property of this model, not an artefact of the split it was measured on.

Test totals: 2,370 matched, 137 missed, 78 confused, 178 false positives.

Commented examples for `test` are written to `report/figures/examples-test/`.

### 7.9 What remains open

- **Why is Paper hard?** Size is ruled out. Plastic confusion is ruled out. The remaining candidates —
  low-contrast material appearance (white paper against white metal), annotation density, genuine
  occlusion — are not distinguished by anything measured here.
- **A real occlusion measure** would need either the pre-filtering full-object annotations, or a proxy
  validated against data where occlusion is known.

---

## 8. Video inference

### 8.1 Footage and provenance

Eight third-party stock clips were evaluated — four from Shutterstock, four from Dreamstime (§11). All
are watermarked preview assets, used for academic evaluation only and **not redistributed**; the
pipeline is re-run against the source URLs instead.

Nothing was applied beyond concatenation. The slowed variants use frame duplication (`setpts`), never
interpolation, so no synthetic frame ever entered the model. That matters: an interpolating filter
would have manufactured frames the detector then "detected", which is not a measurement.

The clips fall into two groups, and the split turns out to be the whole finding.

| Group | Resolution | Character |
| --- | --- | --- |
| Close belt views | 580×326, 504×900 | waste fills the frame, no people, raised/overhead angle |
| Facility views | 596×336 – 898×506 | hands and gloves in frame, machinery, walls and floor dominating |

### 8.2 Deliverable

`runs/media/demo_montage.mp4` — 828 frames, 580×326, 25 fps, **33.12 s, at native speed.**

The four close-view clips share a resolution and frame rate, so they concatenate without re-encoding
and clear the ≥30 s requirement with real footage at real speed: no slow motion, no duplicated frames,
no padding. 5,377 detections across the 828 frames.

Earlier attempts at this deliverable ran 5.3 s and 9.8 s and would have needed a 6× slowdown to reach
30 s — inflating duration without adding a single frame of evidence.

### 8.3 Detection statistics

On `valid` the model predicts a near-uniform class mix (Glass 23.6 %, Metal 25.1 %, Paper 26.6 %,
Plastic 24.8 %), matching the training prior. "Drift" is the total-variation distance of each clip's
predicted class distribution from that reference.

| Clip | View | Det/frame | Glass | Metal | Paper | Plastic | Drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 230274526 | close | 7.60 | 23.0 % | 32.2 % | 16.1 % | 28.7 % | **11.0 %** |
| 232194788 | close | 5.76 | 17.8 % | 18.7 % | 46.1 % | 17.4 % | 19.5 % |
| 232591973 | close | 4.27 | 34.0 % | 5.7 % | 36.6 % | 23.8 % | 20.4 % |
| 261034938 | close | 8.16 | 14.1 % | 34.4 % | 2.6 % | 48.9 % | 33.4 % |
| travels | close | 4.78 | 5.6 % | 30.0 % | 38.1 % | 26.4 % | 18.0 % |
| konin | facility | 1.52 | 33.3 % | 31.3 % | 29.6 % | 5.8 % | 19.0 % |
| recycling | facility | 2.67 | 13.0 % | 56.4 % | 13.9 % | 16.7 % | 31.3 % |
| gloves | facility | 5.35 | 1.7 % | 6.1 % | 61.9 % | 30.3 % | 40.8 % |

Median detection size tracks the same split: **2.8–8.6 % of frame** on the close views against
**10.3–11.3 %** on the facility views, where detections span structure rather than objects. Dataset
instances have a median of 1.4 % of frame.

**Drift is not a quality metric, and konin proves it.** It scores 19.0 % — close to the best clip — yet
failed visibly, drawing two boxes on a background wall while a belt packed with glass bottles went
undetected. Distribution matching is supporting evidence only. Where it disagrees with the qualitative
read, the qualitative read wins.

### 8.4 The transfer result

**The model transfers to a matching viewpoint, and not otherwise.**

On the close-view clips, detections land on the waste:

- green plastic bags → `Plastic 0.87`, `0.81`, `0.59` — correct, repeatedly, across clips
- corrugated cardboard → `Paper 0.86`, `0.45` — correct
- glass bottles in a kerbside bin → `Glass 0.71`, `0.61` — correct

On the facility clips, detections land on the scene rather than the objects:

- `Metal 0.70` across a background wall, over a belt of glass — konin
- `Metal 0.48` on a worker's yellow glove — gloves
- `Paper 0.55`, `0.57` on large ochre background regions — gloves

Same weights, same model. The variable is the view, not the video.

This refines the earlier reading. The model does **not** fail because the input is video; motion blur and
compression are secondary. It fails when the scene departs from the training composition, and that
departure is absent from the dataset, which contains only unobstructed views of one belt.

Occlusion differs between the two groups as well — the facility clips are full of hands, overlapping
objects and machinery — but it is confounded with the other variables and is **not separable** here. No
occlusion claim is made from this footage; see §7.5.

The mechanism is worth stating precisely: the model has four classes and **no way to say "not waste."**
Its implicit background model is "the BUU belt" — that texture, that lighting, those surfaces. Anything
unfamiliar must be forced into Glass, Metal, Paper or Plastic, and it does so confidently. That is a
property of the training data and the class contract rather than of video, and it would recur on any
real installation with guarding, hands or unfamiliar conveyor furniture.

### 8.5 Failure modes visible in the montage

Recorded rather than excluded — the montage keeps all four clips, including the weakest:

- **White or translucent film → `Metal`.** The one error no clip escapes. The cause is not established
  here; it is not attributable to the montage alone.
- **Small objects on a moving belt are missed entirely.** In `230274526` a bottle and a bag on the belt
  are never detected, consistent with the small-instance weakness measured on `valid` (§7.3).
- **Colour shortcuts.** Yellow and ochre surfaces are repeatedly labelled `Glass`; green regions are
  labelled `Plastic` whether or not they are plastic. The same over-reliance on colour that the balanced
  per-class scores on `valid` conceal.
- **Large blob detections** spanning several objects — visible in `232194788`.

### 8.6 Why there is no "image → video drop" number

**The drop cannot be computed.** Every clip is unlabelled, and mAP requires ground truth. There is no
label file for stock footage and none derivable from the dataset. Any percentage quoted here would be
fabricated.

The honest position is the one above: detection statistics and qualitative behaviour, with the
transfer result stated as a scoped claim rather than a metric. Two routes would produce a real number,
neither taken:

1. hand-annotate a sample of frames — 20–30 would suffice — and compute precision and recall against
   them;
2. obtain footage from a comparable viewpoint that already carries labels.

For a measurement that needs neither, §8.7 counts something an mAP cannot: how long detections
persist. It is not a substitute for the drop — persistence and accuracy are different quantities —
but it is the only quantitative video measurement available without labels.

Until then video performance is reported qualitatively and the omission is explicit.

### 8.7 Tracking

`uv run litterbug predict --source <video> --track` links detections across frames with ByteTrack
(`model.track`, `persist=True`), writes a track id onto every detection and into the overlay, and
reports how long each track survived.

**Deliverable:** `runs/media/demo_tracked.mp4` — the four close-view clips, 828 frames, 33.12 s,
every frame annotated with class, confidence and track id. Sources are the raw clips, so the model
never sees its own earlier annotations.

Tracking collapses 5,101 per-frame detections into **138 tracks**:

| | Value |
| --- | --- |
| Frames | 828 |
| Detections | 5,101 (6.16 per frame) |
| Unique tracks | **138** |
| Mean track length | 37.0 frames |
| Median track length | 14.0 frames (0.56 s at 25 fps) |
| Longest track | 217 frames |
| Single-frame tracks | 16 (11.6 %) |

#### Track length is a label-free reliability signal

Splitting the detections by how long their track survived:

| Track length | Detections | Mean confidence | Median confidence |
| --- | --- | --- | --- |
| single frame | 16 | 0.425 | 0.396 |
| 2–5 frames | 101 | 0.513 | 0.449 |
| 6–25 frames | 530 | 0.515 | 0.474 |
| 26+ frames | 4,454 | **0.625** | **0.630** |

**Confidence rises monotonically with persistence** — 0.425 to 0.625 in the mean, 0.20 of confidence
between the flickers and the stable tracks. The detections the model is least sure about are the same
ones that fail to survive.

That is usable without labels: on footage that cannot be annotated, track length is a free filter.
It also revises the per-frame reading. 5,101 detections sounded like instability, but **87 % of them
(4,454) belong to tracks of 26 frames or longer** — the detector is more temporally stable than the
raw count suggests, and the instability is concentrated in a few hundred detections.

#### Per-class persistence varies 4.5×

Median track length by class: **Glass 32 frames, Paper 18, Plastic 10, Metal 7.**

Metal tracks are the shortest by a wide margin. §8.5 already records white film being labelled
`Metal`; short-lived Metal predictions are a plausible expression of the same error. The connection
is not established here and should not be read as one.

#### Two methodological notes

**Track ids do not survive a cut.** Running over a montage of concatenated clips gives ByteTrack no
boundary to reset at, so ids leak across scenes — the montage run reported a longest track of 322
frames inside scenes only ~200 frames long. The figures above come from per-clip runs, where a track
cannot span a cut, aggregated afterwards. The montage figure is recorded only as the caution it is.

**Tracked and untracked counts are not directly comparable.** Tracking adds its own thresholds, so
the tracked runs report 5,101 detections against 5,377 for the same clips without `--track` — 5.1 %
fewer. That is the tracker filtering, not a change in the detector, and the two figures should not be
read as a before-and-after.

---

## 9. Reproducing

```bash
uv sync                                           # install dependencies
nvidia-smi                                        # confirm the GPU is visible

uv run litterbug train --task segment --dry-run   # show resolved config, no GPU cost
uv run litterbug train --task segment             # fine-tune

uv run litterbug val --weights runs/<run>/weights/best.pt --split val
uv run litterbug val --weights runs/<run>/weights/best.pt --split test    # once, at the end
uv run litterbug predict --weights runs/<run>/weights/best.pt --source <image|dir|video>
uv run litterbug predict --weights runs/<run>/weights/best.pt --source <video> --track  # ByteTrack

uv run litterbug analyze --weights runs/<run>/weights/best.pt --split val  # per-instance errors
uv run litterbug examples --weights runs/<run>/weights/best.pt --split val # commented FP/FN figures
uv run litterbug eda                                                       # dataset EDA

uv run pytest                                     # contracts + config
```

`python -m cli <command>` is equivalent to `uv run litterbug <command>`.

Hyperparameters are **not** restated in documentation. They live in `src/common/config.py`; every run
prints its resolved configuration on startup, and to see one without training, use `--dry-run`.

---

## 10. Limitations and threats to validity

- **`valid` is not `test`, and the report now carries both.** `test` was evaluated once, after all
  iteration had stopped, so the figures in §1, §5.4, §6 and §7.8 are not ones any decision was made
  on. Everything else comes from `valid`; each table states its split.
- **Seed noise is comparable to the `valid`/`test` gap.** The two-point mAP50-95 drop between splits is
the same size as the segmentation-versus-detection difference the report declines to interpret. It is
  reported as a generalisation estimate, not as a precise measurement.
- **Single seed.** One run per architecture. Differences below ~0.01 mAP should not be treated as
  meaningful — which is exactly the size of the segmentation-versus-detection gap, so that result
  reads as "no detectable difference" rather than "detection wins by 0.0017".
- **The two models are not parameter-matched.** Segmentation adds 1,486,212 parameters (+14.9 %) and
  14.5 GFLOPs (+63.6 %) over detection (11,437,172 vs 9,950,960 parameters; 37.3 vs 22.8 GFLOPs).
  "Same size" means the same `s` scale, not equal capacity, so this is a comparison of the two models
  as shipped rather than a parameter-matched ablation.
- **True occlusion is unmeasurable in this dataset.** Annotations trace only visible boundaries and
  exclude objects below ~30 %, so no occlusion-stratified result is possible. §7 reports size and
  solidity proxies instead, and its solidity proxy is confounded by material rather than occlusion.
- **Environment constraint worth disclosing.** `/dev/shm` is 63 MiB in this container, so data loading
  ran with `workers=0` (single process). This affects throughput, not results, but it means the 86.6
  s/epoch figure is not a well-tuned benchmark. The pipeline derives the worker count from `/dev/shm`
  and falls back automatically rather than crashing.
- **A 1-epoch smoke test is not a preview.** See §4.2 — short-schedule metrics are misleading when
  warmup and mosaic-close exceed the schedule length.
- **The model cannot decline to detect.** There are four classes and no background class, so anything
  unfamiliar must be assigned to one of them, confidently. §8.4 shows this happening: a background
  wall, a worker's glove and a conveyor chute each attract a prediction. That is a property of the
  label set rather than of the footage, and it bounds how the model can behave on any scene it was
  not trained for.
- **The training data contains no lighting variation.** §3.1 measures a p10–p90 brightness spread of
  roughly ten levels on a 0–255 scale, with no image below 60 or above 180. The model has therefore
  never seen a differently lit scene, which is a concrete and testable reason to expect the transfer
  failures in §8 rather than an appeal to "domain shift" in general.

---

## 11. References

**Dataset**

> VisionLab, Burapha University. *BUU Waste Occlusion Dataset (BUU-WOD)*. Kaggle.
> <https://www.kaggle.com/datasets/visionlab1buu/buu-waste-occlusion-dataset>
> Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
> (<https://creativecommons.org/licenses/by/4.0/>).

Our copy was exported from Roboflow Universe (workspace
`ggdlbngngfnogfhnolpgdolhpfdlgflgfsbgdtndgtjnt`, project `BUU_WasteOcclusion`, id
`dfaef0e9r93rjedf9ijdffjedsfoeofj`), which publishes the same dataset under the same licence. The
provider's preprocessing — EXIF orientation stripping and a stretch resize to 640×640 — was inherited
unchanged, and **no modification was made to the images or annotations**. The class order and the split
boundaries are also the provider's, used as-is.

**Software**

> Ultralytics. *Ultralytics YOLO* (v8.4.148). <https://github.com/ultralytics/ultralytics> — AGPL-3.0.

**Video footage**

Evaluation clips are third-party stock previews, used for academic evaluation only and **not
redistributed**. Clip IDs are the stable identifiers; the URLs are the libraries' preview assets, and
the canonical clip page should be cited in the final report.

| Library | Clip ID | Used for |
| --- | --- | --- |
| Shutterstock | 4058202333 | facility view — counterexample |
| Shutterstock | 3503618281 | facility view — counterexample |
| Shutterstock | 15499249 | facility view — counterexample |
| Shutterstock | 4058202301 | close view |
| Dreamstime | 230274526 | montage |
| Dreamstime | 232194788 | montage |
| Dreamstime | 232591973 | montage |
| Dreamstime | 261034938 | montage |

Both libraries license this footage royalty-free for a fee. The assets used here are watermarked
previews and carry no redistribution right.

Two further Shutterstock links were supplied (clip IDs 3503655899 and 4058202323) but the corresponding
clips are **not present in the repository** and were never evaluated, so they are omitted.

---

## 12. Appendix — run record

| Item | Value |
| --- | --- |
| Segmentation run | `runs/litterbug-segment-yolo26s-seg` |
| Best checkpoint | `runs/litterbug-segment-yolo26s-seg/weights/best.pt` (epoch 89) |
| Validation run | `runs/litterbug-segment-yolo26s-seg-val` |
| Machine-readable metrics | `runs/litterbug-segment-yolo26s-seg-val/metrics.json` |
| Detection run | `runs/litterbug-detect-yolo26s` |
| Detection checkpoint | `runs/litterbug-detect-yolo26s/weights/best.pt` (epoch 85) |
| Detection validation | `runs/litterbug-detect-yolo26s-val` (incl. `metrics.json`) |
| Error analysis — `valid` mask | `runs/litterbug-segment-yolo26s-seg-val-analysis/error_analysis.json` |
| Error analysis — `valid` box | `runs/litterbug-segment-yolo26s-seg-val-box-analysis/error_analysis.json` |
| Test validation — segment | `runs/litterbug-segment-yolo26s-seg-test/` (incl. `metrics.json`) |
| Test validation — detect | `runs/litterbug-detect-yolo26s-test/` (incl. `metrics.json`) |
| Error analysis — `test` mask | `runs/litterbug-segment-yolo26s-seg-test-analysis/error_analysis.json` |
| Error analysis — `test` box | `runs/litterbug-segment-yolo26s-seg-test-box-analysis/error_analysis.json` |
| Dataset EDA | `runs/dataset-eda/dataset_eda.json` |
| Commented examples | `report/figures/examples-val/`, `report/figures/examples-test/` |
| Video montage | `runs/media/demo_montage.mp4` (828 frames, 33.12 s) |
| Tracked demo | `runs/media/demo_tracked.mp4` (828 frames, 33.12 s, track ids) |
| Clean montage input | `runs/media/raw_montage.webm` |
| Track statistics | `runs/predict/track-<clip>/detections.json` (`tracks` block) |
| Per-clip detections | `runs/predict/<clip>/detections.json` |

Environment: Python 3.12.3, torch 2.14.0+cu130, CUDA 13.0, Ultralytics 8.4.148, RTX 3060 Laptop
(5,804 MiB), Ubuntu 24.04 dev container.
