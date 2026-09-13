# litterbug — Trash Detection & Instance Segmentation on a Conveyor Belt


Instance segmentation and detection of waste objects on a conveyor belt, built on the BUU Waste Occlusion
dataset [1] with Ultralytics YOLO26 [3].

**Status.** All required components complete: segmentation and detection trained, evaluated on `valid` and
on a held-out `test` split, with per-instance error analysis, dataset EDA, commented error examples and
tracked video inference.

**Revision.** Code revision `5744ee96` (2026-09-12). Sources, licences and access dates: §11.

---

## 1. Headline results

Segmentation on the held-out `test` split — 200 images, 2,507 instances, evaluated once after every
model decision had already been made:

| Metric | Box | Mask |
| --- | --- | --- |
| mAP50-95 | 0.837 | 0.796 |
| mAP50 | 0.954 | 0.955 |
| Precision | 0.939 | 0.944 |
| Recall | 0.901 | 0.905 |

Mean IoU over matched pairs (§7.8): 0.901 for masks against 0.924 for boxes. Inference costs 8.6
ms/image on an RTX 3060 Laptop (≈117 FPS), excluding pre- and post-processing.

The same figures run one to two points higher on `valid` — box 0.857, mask 0.814 mAP50-95. That gap is
the cost of reporting on data the model never saw, and §5.4 shows nearly all of it sits in the smallest
size quartile.

Within the segmentation model, the mask head scores 0.041 mAP50-95 below the box head on `test` and
0.042 below on `valid`. That is what predicting outlines instead of rectangles costs inside one model.

The detection baseline was trained on the same splits with the same schedule, to test whether the mask
task costs anything at all:

| Split | Detection only | Segmentation (box head) | Difference |
| --- | --- | --- | --- |
| `valid` | 0.8583 | 0.8566 | 0.0017 |
| `test` | 0.8376 | 0.8372 | 0.0004 |

Both differences sit below the ~0.01 seed-noise floor established in §6, so the segmentation box head is
indistinguishable from a model trained on boxes alone. Segmentation delivers masks at 0.796 mAP50-95 on
unseen data at no measurable cost to detection. §6 carries that comparison; §7 carries the error
analysis.

---

## 2. Problem

Waste arrives on a conveyor belt in dense, overlapping batches. Annotations trace only the visible
boundary of each object, and objects below roughly 30 % visibility are excluded by the provider's
protocol (§11). Two consequences shape the evaluation:

- Overlap is a labelling convention, not an error: a partially occluded object is correctly annotated
  data.
- Success is not only finding an object but tracing where it stops being visible. That is a harder
  target than detection, and it is what this project measures.

Segmentation is the product. Detection is trained as a comparison baseline on the same splits with the
same schedule, so any difference is attributable to the head rather than to the recipe.

---

## 3. Dataset

**Source.** BUU-WOD (Burapha University Waste Occlusion Dataset), VisionLab, Burapha University,
licensed CC BY 4.0. Cited as [1] and [2]; no image or annotation was modified.

The public record has two components. This work uses only the first.

| Component | Contents | Used here |
| --- | --- | --- |
| `1_Model_Training_Data/` | 2,000 images at 640×640, YOLO segmentation polygons | yes |
| `2_Experiment_Sample/` | 366 images at 640×480, COCO polygons and boxes, 600 objects | no |

This matters for scope. The second component is a separate conveyor-belt evaluation set; because it
was not used, none of its 600 objects appears in any figure or metric here. The provider is explicit
that it "should not be used alone to support broad claims about unrestricted real-world robustness"
(§10).

Split counts, as delivered and as reproduced by `uv run litterbug validate`:

| Split | Images | Instances | Instances/image |
| --- | --- | --- | --- |
| train | 1,400 | 17,400 | 12.43 |
| valid | 400 | 4,941 | 12.35 |
| test | 200 | 2,507 | 12.54 |

Classes in fixed order: `0 Glass`, `1 Metal`, `2 Paper`, `3 Plastic`. The order is a contract with the
annotations and is declared once, in `src/litterbug/common/constants.py`.

The validator checked image/label pairing, polygon validity, coordinate ranges and class balance
before training, and still reproduces the counts above.

**Split policy.** `valid` is the iteration split. `test` was evaluated once, after training and every
model decision had stopped, so none of the reported figures is one a decision was made on. Each table
states its split; the CLI names them `--split val` and `--split test`, which are the same splits as
`valid` and `test` in the filenames. §7.1 gives the one place where `test` also carries a derived
choice: the confidence floor.

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

The dataset is balanced. The largest-to-smallest class ratio is 1.07–1.17, and Paper is the largest
class in every split. That independently rules out class imbalance as an explanation for Paper's poor
recall, since Paper is the most numerous class rather than the scarcest (§7.4 rules out size and
confusion). No image on any split is unlabelled and no label row is malformed.

Geometry is uniform: every sampled image is 640×640, three-channel, single resolution, as expected
from the provider's stretch resize (§11).

**Lighting is invariant.** Mean brightness over a 200-image sample per split:

| Split | p10 | Median | p90 |
| --- | --- | --- | --- |
| train | 71.1 | 75.8 | 80.7 |
| valid | 72.1 | 76.9 | 81.4 |
| test | 71.3 | 76.1 | 80.9 |

The p10–p90 spread is about ten levels on a 0–255 scale, and no image falls below 60 or above 180:
the dataset is dim (median ≈30 % of range) and almost perfectly uniform. The model has therefore never
seen a scene lit differently from this one, which is a concrete reason to expect the transfer failures
in §8 and is carried into §10.

Sharpness (variance of the Laplacian) measures 675–699 across the splits. It is a usable baseline for
comparing footage sharpness; no such comparison is made here.

---

## 4. Method

### 4.1 Architecture

`yolo26s-seg` from the Ultralytics YOLO26 segmentation family, "s" size, fine-tuned from the
COCO-pretrained weights in [5]. 830 of 844 parameter tensors transferred. The figures below come from
the run's startup model summary, which is printed to stdout and not written to a file.

| | Value |
| --- | --- |
| Layers | 309 (136 fused) |
| Parameters | 11,437,172 (10,366,888 fused) |
| GFLOPs | 37.3 (34.3 fused) |
| Head | `Segment26` |

The model name is declared once, in `MODELS` (`src/litterbug/common/constants.py`), so changing
architecture is a one-line edit. A YOLO11 pair is recorded adjacent as `MODELS_FALLBACK`; the Day-1
gate passed, so it was not needed.

### 4.2 The Day-1 architecture gate, and what a short smoke test cannot show

A single epoch was run before committing to the architecture, to check that the segmentation head
produced non-degenerate masks rather than whole-image blobs. It did, so YOLO26 was kept. The gate cost
about two minutes of wall clock.

Its metrics should not be read as evidence. The gate reported mask mAP50-95 of 0.159 at epoch 1; the
100-epoch run reported 0.564 at epoch 1 on the same model, data and seed. The cause is that
`warmup_epochs=3` and `close_mosaic=10` are absolute epoch counts while the gate ran with `epochs=1`.
Both settings degenerate in a one-epoch run: the learning rate never completes warmup, and mosaic
augmentation is disabled from the first batch. "Epoch 1 of a 1-epoch schedule" is a different training
regime from "epoch 1 of a 100-epoch schedule", so the two are not comparable.

A short smoke test is valid for checking that the plumbing works and for measuring throughput. It is
never valid for reading quality.

### 4.3 Schedule

Frozen before training and not tuned. The settings are declared in `src/litterbug/common/config.py` and
reproduced here for convenience; every run also writes its resolved arguments to `args.yaml`.

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
| Optimizer | AdamW, lr 0.00125, momentum 0.9 |

The optimizer row is the value Ultralytics resolved, not the value in `args.yaml`. That file records the
pre-selection defaults (`optimizer: auto`, `lr0: 0.01`, `momentum: 0.937`); the resolved AdamW settings
appear only in the run log and in the `lr/pg0` column of `results.csv`.

Batch size is derived from VRAM rather than chosen. The pipeline reads the visible device and applies
fixed thresholds (≥11 GiB → 16, ≥5 GiB → 8); on the 5.7 GiB RTX 3060 that resolves to 8, which peaked at
3.9 GB. Peak VRAM comes from the run log and is not written to any artifact. `copy_paste` is enabled
because this dataset's difficulty is occlusion.

### 4.4 Hardware and cost

RTX 3060 Laptop, 5.7 GiB VRAM. Training completed 100 epochs in ≈2.7 h of GPU time, with a steady-state
throughput of 86.6 s/epoch measured across epochs 40–100.

The run was interrupted and resumed once, which changes how the recorded timings read. The internal
timer resets on resume — the counter drops from 1158 s to 107 s at epoch 11 — so `results.csv` ends at
8,629 s (2.40 h) covering the second session alone. The 2.7 h figure is both sessions summed
(1,158 s + 8,629 s = 9,787 s), and it is the one used for any comparison against the detection run,
which ran uninterrupted (§6).

Data loading ran with `workers=0` because `/dev/shm` is 63 MiB in this container. This affects
throughput, not results, but it means 86.6 s/epoch is not a well-tuned benchmark. The pipeline derives
the worker count from `/dev/shm` and falls back automatically rather than crashing.

### 4.5 Reproducibility

- All randomness is seeded: `torch.manual_seed`, `np.random.seed`, `random.seed`, `PYTHONHASHSEED`, and
  Ultralytics `seed=42`.
- Every run prints its resolved configuration and environment block before touching the GPU, and
  Ultralytics writes the same arguments to `args.yaml` in the run directory.
- `--dry-run` resolves and prints the configuration without spending GPU time.
- What is *not* pinned, and therefore limits exact reproduction: this report describes code revision
  `5744ee96`, but `dataset/`, `weights/` and `runs/` are excluded from version control, the dataset has
  no recorded checksum, and `pyproject.toml` leaves dependencies unpinned. Dependency versions are
  those in `uv.lock` ([3]).

---

## 5. Results — segmentation

### 5.1 Learning curve (`valid`)

| Epoch | Wall (s) | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| 1 | 106 | 0.766 | 0.589 | 0.765 | 0.564 |
| 10 | 1158 | 0.903 | 0.740 | 0.900 | 0.712 |
| 20 | 1175 | 0.923 | 0.769 | 0.921 | 0.745 |
| 40 | 3430 | 0.945 | 0.813 | 0.946 | 0.784 |
| 60 | 5428 | 0.955 | 0.839 | 0.955 | 0.801 |
| 80 | 7210 | 0.962 | 0.853 | 0.962 | 0.811 |
| 89 (best) | 7981 | 0.963 | 0.857 | 0.963 | 0.815 |
| 100 | 8629 | 0.962 | 0.855 | 0.961 | 0.813 |

Best checkpoint is epoch 89 (mask mAP50-95 0.81451). Epoch 100 is 0.0016 below it, so the model had
converged rather than being cut off by the schedule, and patience never triggered.

Over half the total quality arrives in the first 10 epochs; the final 60 add about 0.03 mAP50-95. The
wall-clock column resets at the resume described in §4.4: 1158 s is the first session's total, and the
counter restarts from epoch 11.

### 5.2 Per-class results (`valid`)

| Class | Box mAP50-95 | Mask mAP50-95 | Box − Mask |
| --- | --- | --- | --- |
| Glass | 0.892 | 0.844 | 0.048 |
| Metal | 0.881 | 0.832 | 0.049 |
| Plastic | 0.861 | 0.819 | 0.042 |
| Paper | 0.791 | 0.763 | 0.028 |

Paper is the weakest class in both heads: 0.070 below the next-weakest (Plastic, 0.861 box) and 0.101
below the strongest (Glass, 0.892). That gap exceeds the box/mask difference. Glass is the strongest
class in both heads, so neither Glass nor Plastic is a weak class here.

Two qualifications. Per-class mAP cannot separate confusion from missed detections; §7.4 settles that
question and finds Paper's errors dominated by misses, with cross-class confusion negligible. And at the
1-epoch gate Glass was the worst class (mask 0.059), consistent with an unconverged head rather than any
property of Glass — early-epoch per-class rankings should not be trusted.

### 5.3 Box head versus mask head, within the segmentation model

Mask mAP50-95 is 0.042 below box mAP50-95 on `valid` (0.814 against 0.857), and the ranking of classes
is identical in both heads. The mask head adds outline precision at modest cost and does not reorder
class difficulty.

§7.6 corroborates this from an independent metric: mean IoU over matched pairs is 0.907 for masks
against 0.929 for boxes, a 0.022 gap in the same direction and of the same order as the mAP difference.

This is a within-model comparison — what the mask head costs relative to the box head of the *same*
network. It is not a segmentation-versus-detection result; that requires the detection baseline of §6.

### 5.4 Test split

`test` was evaluated once, after training and all iteration had stopped (§3).

| Metric | `valid` | `test` | Change |
| --- | --- | --- | --- |
| Box mAP50-95 | 0.857 | 0.837 | −0.019 |
| Mask mAP50-95 | 0.814 | 0.796 | −0.018 |
| mAP50 | 0.963 | 0.954 | −0.009 |
| Precision (box) | 0.948 | 0.939 | −0.009 |
| Recall (box) | 0.923 | 0.901 | −0.021 |

The gap is about two points of mAP50-95, the same size as the ~0.01 seed-noise floor discussed in §6.
That is small enough to say the model is not overfitted to `valid`, and coarse enough that the estimate
should not be read too finely.

Where the drop sits is more informative than its size.

| Size quartile | `valid` recall | `test` recall | Change |
| --- | --- | --- | --- |
| q1 — smallest | 0.8947 | 0.8262 | −0.069 |
| q2 | 0.9789 | 0.9744 | −0.005 |
| q3 | 0.9854 | 0.9904 | +0.005 |
| q4 — largest | 0.9919 | 0.9904 | −0.002 |

Recall on q2–q4 is flat to within half a point; q1 falls by 0.069. The model has not degraded uniformly
on unseen data — it has lost small objects specifically, the same failure §7.3 identifies as the
dominant error source on `valid`, now amplified on data the model had never seen.

---

## 6. Results — detection baseline

The baseline was trained on the same splits with the same 100-epoch schedule, seed and `s` size, so the
head and the training objective are the only differences. It completed 100 epochs in 1.80 h against
2.72 h for segmentation (a smaller model, carrying no mask loss) and peaked at roughly 2.5 GB VRAM
against 3.9 GB, so no resource limit can explain the outcome.

| Split | Model | Box mAP50 | Box mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| `valid` | Detection only | 0.9633 | 0.8583 | 0.939 | 0.931 |
| `valid` | Segmentation (box head) | 0.9634 | 0.8566 | 0.948 | 0.923 |
| `test` | Detection only | 0.9557 | 0.8376 | 0.944 | 0.913 |
| `test` | Segmentation (box head) | 0.9542 | 0.8372 | 0.939 | 0.901 |

Difference in box mAP50-95: 0.0017 on `valid`, 0.0004 on `test`. Both are far below the ~0.01 floor that
one seed per model can resolve, so the two box heads are indistinguishable on either split. The
held-out split repeats the `valid` result rather than narrowing it.

Per-class differences on `valid` agree, with mixed signs and every value inside ±0.01:

| Class | Detection | Segmentation (box) | Difference |
| --- | --- | --- | --- |
| Glass | 0.895 | 0.892 | +0.003 |
| Metal | 0.873 | 0.881 | −0.008 |
| Paper | 0.800 | 0.791 | +0.009 |
| Plastic | 0.864 | 0.861 | +0.003 |

No consistent direction and no class where the effect is systematic — and since the ±0.01 floor applies
to every row of that table, none of them should be read as a class-level effect either.

### 6.1 What this means

Adding the mask head at the same `s` scale cost no measurable box accuracy. Segmentation yields masks at
0.796 mask mAP50-95 on `test` while its box head stays indistinguishable from a model trained on boxes
alone, so the outlines came at no detectable detection cost.

Two qualifications bound that claim.

The models are not parameter-matched. Segmentation adds 1,486,212 parameters (+14.9 %) and 14.5 GFLOPs
(+63.6 %) over detection — 11,437,172 against 9,950,960 parameters, 37.3 against 22.8 GFLOPs. "Same
size" means the same `s` scale, not equal capacity, so this compares the two models as shipped rather
than as a parameter-matched ablation. The defensible form of the claim is "no measurable detection cost
at 14.9 % more parameters", not "a free mask head".

Precision and recall traded against each other. Segmentation is slightly more precise (0.948 against
0.939) and slightly less sensitive (0.923 against 0.931). Those roughly cancel, which is why the
aggregate lands in the same place — a single mAP number hiding two countervailing behaviours.

One seed per model. Differences below ~0.01 should not be treated as real; that floor applies to the
comparison above and to every per-class row in it.

---

## 7. Error analysis

§7.1–§7.6 and §7.8 below come from the `analyze` command. §7.7's false-positive decomposition comes from
the `examples` command, which writes a different file.

```bash
uv run litterbug analyze --weights runs/litterbug-segment-yolo26s-seg/weights/best.pt --split val
uv run litterbug analyze --weights runs/litterbug-segment-yolo26s-seg/weights/best.pt --split test
```

- `runs/litterbug-segment-yolo26s-seg-val-analysis/error_analysis.json` — §7.1–§7.6
- `runs/litterbug-segment-yolo26s-seg-val-box-analysis/error_analysis.json` — the box column of §7.6
- `runs/litterbug-segment-yolo26s-seg-test-analysis/error_analysis.json` — §7.8
- `runs/litterbug-segment-yolo26s-seg-test-box-analysis/error_analysis.json` — the box column of §7.8
- `report/figures/examples-val/error_examples.json` — the decomposition in §7.7
- `report/figures/examples-test/error_examples.json` — the `test` column of §7.8

Add `--iou-mode box` to either split for the box-matching run. The split and the matching mode are part
of the run name because they are not part of the path: before that change, analysing `test` silently
overwrote the `valid` analysis, and a box run overwrote the mask run.

### 7.1 How the numbers were derived

Ultralytics computes per-instance matches internally but does not expose them, so the confusion matrix
normally exists only as a PNG. This analysis re-derives the matching from scratch.

| Setting | Value |
| --- | --- |
| Matching criterion | mask IoU ≥ 0.5 |
| Confidence floor | 0.34 — a recall-oriented operating point, not the F1 optimum (see below) |
| Assignment | predictions by descending confidence, greedy one-to-one, class-agnostic |

**The confidence floor is a free parameter, and this is where it was set.** Sweeping it over the `valid`
split under exactly the convention above puts the F1 optimum at **0.555** (F1 0.9559, precision 0.9690,
recall 0.9431). This section runs lower, at 0.34 — F1 0.9452, precision 0.9282, recall 0.9628 — which
gives up 0.011 of F1 for 0.020 of recall. The trade is deliberate: the questions below are about which
objects the model fails to *find*, and a higher floor would start manufacturing misses rather than
measuring them. The `BoxF1_curve.png` and `MaskF1_curve.png` files in the run directories peak in the
same region, 0.49–0.56.

Class-agnostic matching is deliberate: a Paper prediction landing on a Plastic object must register as a
confusion, not as a miss plus a false positive. Conventions differ here, so the choice is stated rather
than assumed.

Result at that floor: 4,757 matched of 5,125 predictions against 4,941 ground-truth instances — recall
0.9628, precision 0.9282.

**Rounding.** Metrics are given to three decimals, except IoU and gaps smaller than 0.01, which are given
to four. Deltas are recomputed from the artifact values rather than from the rounded figures in the
tables, which is why the box/mask gap in §5.3 is 0.042 and not the 0.043 that subtracting two rounded
headline numbers would give.

### 7.2 Confusion between classes is negligible (`valid`)

Across 4,941 instances the largest off-diagonal cell is 31 — Paper predicted as Metal — and Glass↔Plastic
confusion accounts for 12 errors (0.24 %).

| truth ↓ / predicted → | Glass | Metal | Paper | Plastic | missed |
| --- | --- | --- | --- | --- | --- |
| Glass | 1168 | 4 | 1 | 10 | 20 |
| Metal | 3 | 1151 | 6 | 6 | 32 |
| Paper | 2 | 31 | 1139 | 19 | 96 |
| Plastic | 2 | 21 | 30 | 1164 | 36 |

Glass→Plastic is 10 and Plastic→Glass 2, against roughly 1,200 instances of each. No class pair is a
meaningful error source in this model; the error budget is dominated by misses rather than confusions.

### 7.3 Size is the dominant driver of misses (`valid`)

| Size quartile (ground-truth mask px) | Instances | Recall |
| --- | --- | --- |
| q1 — 15 to 3,600 | 1,234 | 0.895 |
| q2 — 3,600 to 5,713 | 1,235 | 0.979 |
| q3 — 5,713 to 9,030 | 1,236 | 0.985 |
| q4 — 9,030 to 59,264 | 1,236 | 0.992 |

Recall differs by ten points between the smallest and largest quartile, with almost all of the loss
concentrated in q1. Small instances are the largest single source of error in this model, larger than
every class confusion combined.

### 7.4 Paper is intrinsically hard, not merely small (`valid`)

Paper has the weakest recall (0.885). The obvious explanation is that paper objects are small; the data
says otherwise. Recall by class within the same size quartiles:

| Class | Instances | Median size (px) | q1 | q2 | q3 | q4 |
| --- | --- | --- | --- | --- | --- | --- |
| Glass | 1,203 | 4,982 | 0.953 | 0.990 | 0.997 | 1.000 |
| Metal | 1,198 | 4,570 | 0.936 | 0.989 | 0.996 | 0.994 |
| Plastic | 1,253 | 7,914 | 0.874 | 0.970 | 0.982 | 0.998 |
| Paper | 1,287 | 6,116 | 0.798 | 0.957 | 0.967 | 0.980 |

Two things follow.

1. Paper's instances are not the smallest. Its median (6,116 px) is larger than Glass (4,982) and Metal
   (4,570), so the weakness is not a size artefact.
2. Paper is the worst class in every quartile. In q1 it trails Glass by 15.5 points, Metal by 13.8 and
   Plastic by 7.6; in q2–q4 it trails by 1.3–3.3. Size degrades every class, but it does not explain
   Paper.

Paper's errors also skew to misses rather than confusion — 96 missed against 52 confused — and its largest
confusion is with Metal (31), not Plastic (19). Class imbalance is ruled out separately in §3.1.

### 7.5 On occlusion — what this analysis cannot claim

The dataset is named for occlusion, but true occlusion is not measurable from it. Annotations trace only
visible boundaries, and objects below ~30 % visibility were excluded rather than annotated with their
hidden geometry, so the hidden fraction of an object cannot be recovered.

Two proxies were computed instead.

| Proxy | q1 | q4 | Verdict |
| --- | --- | --- | --- |
| Mask area (size) | 0.895 | 0.992 | Strong and monotonic (`valid`, §7.3) |
| Solidity (area ÷ convex hull) | 0.924 | 0.963 | Weak and non-monotonic (`valid`) |

Solidity is confounded: naturally irregular materials (crumpled paper, plastic film) score low whether or
not they are occluded, and its non-monotonicity — q3 scores 0.989 while the perfectly convex q4 scores
0.963 — is consistent with it tracking material rather than occlusion. It must not be read as an
occlusion result, and no occlusion-stratified claim is made anywhere in this report.

The same applies to §8. The facility clips do contain heavy occlusion, but it is confounded with
viewpoint, human presence and scene composition, and the experiment does not separate them. §8 is a
composition finding, not an occlusion measurement.

### 7.6 Mask quality — IoU (`valid`)

Recall and precision say whether an object was found; IoU says whether it was traced correctly. Across
the 4,757 matched pairs, mean IoU is 0.9074 (median 0.9278, p10 0.8391, p90 0.9597) and 95.7 % of pairs
score ≥ 0.75.

Running the same analysis with `--iou-mode box` gives a like-for-like comparison:

| | Mean IoU | Median | p10 | p90 |
| --- | --- | --- | --- | --- |
| Box | 0.9291 | 0.9519 | 0.8628 | 0.9789 |
| Mask | 0.9074 | 0.9278 | 0.8391 | 0.9597 |

The mask head gives up 0.022 IoU — an independent measurement of the effect §5.3 reports as 0.042
mAP50-95. Two metrics that measure different things agree in direction and magnitude. Per class, over
true positives only, since a cross-class match says nothing about the predicted class's mask quality:

| Class | Pairs | Mask IoU | Box IoU | Gap |
| --- | --- | --- | --- | --- |
| Glass | 1,168 | 0.9129 | 0.9374 | 0.0245 |
| Metal | 1,151 | 0.9143 | 0.9363 | 0.0220 |
| Plastic | 1,164 | 0.9113 | 0.9345 | 0.0232 |
| Paper | 1,139 | 0.8953 | 0.9120 | 0.0167 |

Paper is worst in both modes, as in §7.4 — a second metric singling out the same class. IoU also degrades
with size:

| Size quartile | Mask IoU | Box IoU | Gap |
| --- | --- | --- | --- |
| q1 — smallest | 0.8724 | 0.9043 | 0.0319 |
| q2 | 0.9118 | 0.9352 | 0.0234 |
| q3 | 0.9189 | 0.9373 | 0.0184 |
| q4 — largest | 0.9231 | 0.9373 | 0.0142 |

Small instances are penalised twice over: missed more often (recall 0.895 against 0.992, §7.3) and
outlined less accurately when found. The mask head's cost is concentrated there too — 0.032 IoU on the
smallest quartile against 0.014 on the largest.

**Caveat on the comparison.** The two runs match under different criteria — mask IoU ≥ 0.5 against box
IoU ≥ 0.5 — so the pair sets differ: 4,757 mask-matched against 4,747 box-matched, with recall and
precision marginally apart for the same reason (0.9628 / 0.9282 against 0.9607 / 0.9262). Immaterial to
the conclusion, but the columns are not computed over the same pairs.

### 7.7 Commented examples (`valid`)

`uv run litterbug examples --weights <run>/weights/best.pt --split val` renders the clearest examples of
each error type and writes measured commentary beside them: figures in
`report/figures/examples-val/figures/`, commentary in `report/figures/examples-val/examples.md`, counts in
`report/figures/examples-val/error_examples.json`. Each figure is two panels — ground truth left,
predictions right — with red outlines marking missed ground-truth instances and magenta marking
predictions that matched nothing. The figures are derived images rather than dataset files; licence and
change statement: `report/figures/README.md`.

**Paper dominates every error type.** All three confusion figures have recall 1.00, and Paper appears in
all seven directed confusions across them (Paper→Metal ×3, Metal→Paper ×2, Paper→Plastic ×2). Paper also
accounts for 4 of the 4 misses in the first miss figure and 7 of the 10 across the three, and is the
majority of unmatched predictions in every false-positive figure. That is a third independent line on
Paper, after §7.4's cross-tabulation and §7.6's IoU.

#### False positives are two different failures

Matching is one-to-one, so a second prediction on an already-detected object cannot match and is scored
as a false positive even though it covers a real object. Those are duplicates, not hallucinations, and
they are a different problem from predicting on bare belt. Separating them across the split:

| Category | Count | Share |
| --- | --- | --- |
| Matched | 4,757 | — |
| Missed instances | 184 | — |
| Confused (matched, wrong class) | 135 | — |
| False positives — duplicates | 156 | 42 % of FPs |
| False positives — over background | 212 | 58 % of FPs |

42 % of the model's false positives are duplicate detections, so precision is not one behaviour described
by one number: 0.9282 is 212 background errors plus 156 redundant masks, which have different causes and
would need different remedies. Suppressing duplicates and nothing else, with recall held fixed, would
move precision to 0.9573 (4,757 / 4,969) — an upper bound on what that could buy rather than a result,
since no simple threshold achieves it cleanly.

The `analyze` output cannot make this distinction: an unmatched prediction is an unmatched prediction,
and only re-examining overlap against all instances separates the two.

### 7.8 Test split

The same analysis re-run on the held-out split.

| Metric | `valid` | `test` |
| --- | --- | --- |
| Ground-truth instances | 4,941 | 2,507 |
| Predictions | 5,125 | 2,548 |
| Matched | 4,757 | 2,370 |
| Recall | 0.9628 | 0.9454 |
| Precision | 0.9282 | 0.9301 |
| Mean IoU — mask | 0.9074 | 0.9010 |
| Mean IoU — box | 0.9291 | 0.9244 |

Recall falls 1.7 points and mask IoU 0.6; precision is fractionally higher. Nothing suggests the `valid`
figures were structurally optimistic, and the per-class findings replicate:

| Class | `valid` recall | `test` recall | `valid` IoU | `test` IoU |
| --- | --- | --- | --- | --- |
| Glass | 0.9709 | 0.9846 | 0.9129 | 0.9065 |
| Metal | 0.9608 | 0.9376 | 0.9143 | 0.9066 |
| Plastic | 0.9290 | 0.9045 | 0.9113 | 0.9083 |
| Paper | 0.8850 | 0.8432 | 0.8953 | 0.8876 |

Paper is the weakest class on `test` in both metrics, as on `valid` — so the claim of §7.4 now holds on
data the model never saw, which matters because it rested on the iteration split alone. Glass improves on
`test` (0.9709 → 0.9846), so the class differences are not a uniform shrinkage toward the mean. The one
figure that moves sharply is q1 recall, 0.8947 → 0.8262 while q2–q4 hold steady (§5.4): the entire
generalisation gap is a small-object gap.

The false-positive decomposition replicates almost exactly — `valid` 156 duplicates and 212 over
background from 368, `test` 75 and 103 from 178, 42 % / 58 % on both. Duplicate detections are a stable
property of this model rather than an artefact of the split. Test totals: 2,370 matched, 137 missed,
78 confused, 178 false positives. Examples for `test` are in `report/figures/examples-test/`.

### 7.9 What remains open

- Why is Paper hard? Size (§7.4), class imbalance (§3.1) and Plastic confusion are all ruled out. The
  remaining candidates — low-contrast material appearance, annotation density, genuine occlusion — are
  not distinguished by anything measured here.
- A real occlusion measure would need the pre-filtering full-object annotations, or a proxy validated
  against data where occlusion is known.

---

## 8. Video inference

### 8.1 Footage and provenance

Eight third-party stock clips were evaluated, four from Shutterstock and four from Dreamstime; each is
cited by key in §11. They are watermarked preview assets, used for academic evaluation only and not
redistributed.

Nothing was applied beyond concatenation. The slowed variants use frame duplication (`setpts`) rather
than interpolation, so no synthetic frame entered the model. That matters: an interpolating filter would
have manufactured frames the detector then "detected", which is not a measurement.

The clips fall into two groups, and the split turns out to be the whole finding.

| Group | Resolution | Character |
| --- | --- | --- |
| Close belt views | 580×326, 504×900 | waste fills the frame, no people, raised or overhead angle |
| Facility views | 596×336 – 898×506 | hands and gloves in frame, machinery, walls and floor dominating |

### 8.2 Deliverable

`runs/media/demo_montage.mp4` — 828 frames, 580×326, 25 fps, 33.12 s, at native speed.

The four close-view clips share a resolution and frame rate, so they concatenate without re-encoding and
clear the ≥30 s requirement with real footage at real speed: no slow motion, no duplicated frames, no
padding. The run records 5,377 detections across the 828 frames.

Earlier attempts at this deliverable ran 5.3 s and 9.8 s and would have needed a 6× slowdown to reach
30 s, inflating duration without adding a frame of evidence.

### 8.3 Detection statistics

On `valid` the model predicts a near-uniform class mix (Glass 23.6 %, Metal 25.1 %, Paper 26.6 %,
Plastic 24.8 %), matching the training prior. Drift is the total-variation distance between each clip's
predicted class distribution and that reference.

| Clip | View | Det/frame | Glass | Metal | Paper | Plastic | Drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [6] 230274526 | close | 7.60 | 23.0 % | 32.2 % | 16.1 % | 28.7 % | 11.0 % |
| [7] 232194788 | close | 5.76 | 17.8 % | 18.7 % | 46.1 % | 17.4 % | 19.5 % |
| [8] 232591973 | close | 4.27 | 34.0 % | 5.7 % | 36.6 % | 23.8 % | 20.4 % |
| [9] 261034938 | close | 8.16 | 14.1 % | 34.4 % | 2.6 % | 48.9 % | 33.4 % |
| [10] travels | close | 4.78 | 5.6 % | 30.0 % | 38.1 % | 26.4 % | 18.0 % |
| [11] konin | facility | 1.52 | 33.3 % | 31.3 % | 29.6 % | 5.8 % | 19.0 % |
| [12] recycling | facility | 2.67 | 13.0 % | 56.4 % | 13.9 % | 16.7 % | 31.3 % |
| [13] gloves | facility | 5.35 | 1.7 % | 6.1 % | 61.9 % | 30.3 % | 40.8 % |

Each row is one clip; the bracketed key indexes §11. Median detection size tracks the same split: 2.8–8.6 % of frame on the close views against 10.3–11.3 %
on the facility views, where detections span structure rather than objects. Dataset instances have a
median of 1.4 % of frame. Both figures are derived by hand from the `box_xyxy` fields in
`runs/predict/<clip>/detections.json` and the video dimensions; no command emits them.

**Drift is not a quality metric, and konin is the clearest counterexample.** It scores 19.0 %, close to
the best clip, yet it failed visibly — two boxes on a background wall while a belt packed with glass
bottles went undetected. Distribution matching is supporting evidence only; where it disagrees with the
qualitative read, the qualitative read wins.

### 8.4 The transfer result

The model transfers to a matching viewpoint, and not otherwise.

On the close-view clips detections land on the waste: green plastic bags → `Plastic 0.87`, `0.81`, `0.59`;
corrugated cardboard → `Paper 0.86`, `0.45`; glass bottles in a kerbside bin → `Glass 0.71`, `0.61`. On
the facility clips they land on the scene instead: `Metal 0.70` across a background wall over a belt of
glass (konin), `Metal 0.48` on a worker's yellow glove, and `Paper 0.55`, `0.57` on large ochre
background regions (gloves).

Same weights, same model. The variable is the view, not the video. The model does not fail because the
input is video — motion blur and compression are secondary — but when the scene departs from the training
composition, and that departure is absent from the dataset, which contains only unobstructed views of one
belt. Occlusion differs between the two groups as well, but it is confounded with the other variables and
is not separable here (§7.5).

The mechanism is a property of the label set rather than of video. With four classes and no way to say
"not waste", the model's implicit background model is "the BUU belt" — that texture, that lighting, those
surfaces — so anything unfamiliar must be forced into one of the four, confidently. It would recur on any
real installation with guarding, hands or unfamiliar conveyor furniture.

### 8.5 Failure modes visible in the montage

Recorded rather than excluded; the montage keeps all four clips, including the weakest.

- White or translucent film → `Metal`. The one error no clip escapes; the cause is not established here
  and it is not attributable to the montage alone.
- Small objects on a moving belt are missed entirely. In `230274526` a bottle and a bag on the belt are
  never detected, consistent with the small-instance weakness measured on `valid` (§7.3).
- Colour shortcuts. Yellow and ochre surfaces are repeatedly labelled `Glass`, and green regions are
  labelled `Plastic` whether or not they are plastic — the over-reliance on colour that the balanced
  per-class scores on `valid` conceal.
- Large blob detections spanning several objects, visible in `232194788`.

### 8.6 Why there is no "image → video drop" number

The drop cannot be computed. Every clip is unlabelled and mAP requires ground truth; there is no label
file for stock footage and none derivable from the dataset, so any percentage quoted here would be
fabricated.

Two routes would produce a real number, neither taken: hand-annotate a sample of frames — 20–30 would
suffice — and compute precision and recall against them, or obtain footage from a comparable viewpoint
that already carries labels. §8.7 measures something that needs neither, but track persistence is not a
substitute for the drop. Until one of the routes is taken, video performance is reported qualitatively
and the omission is explicit.

### 8.7 Tracking

`uv run litterbug predict --source <video> --track` links detections across frames with ByteTrack [4]
(`model.track`, `persist=True`), writes a track id onto every detection and into the overlay, and reports
how long each track survived.

**Deliverable:** `runs/media/demo_tracked.mp4` — the four close-view clips, 828 frames, 33.12 s, every
frame annotated with class, confidence and track id. Sources are the raw clips, so the model never sees
its own earlier annotations.

Tracking collapses 5,101 per-frame detections into 138 tracks:

| | Value |
| --- | --- |
| Frames | 828 |
| Detections | 5,101 (6.16 per frame) |
| Unique tracks | 138 |
| Mean track length | 37.0 frames |
| Median track length | 14.0 frames (0.56 s at 25 fps) |
| Longest track | 217 frames |
| Single-frame tracks | 16 (11.6 %) |

The aggregate is computed across the four per-clip runs (§12). `detections.json` stores a per-clip
`tracks` block but not this pooled row, so the mean and median are derived from the per-clip values.

#### Track length is a label-free reliability signal

Splitting the detections by how long their track survived:

| Track length | Detections | Mean confidence | Median confidence |
| --- | --- | --- | --- |
| single frame | 16 | 0.425 | 0.396 |
| 2–5 frames | 101 | 0.513 | 0.449 |
| 6–25 frames | 530 | 0.515 | 0.474 |
| 26+ frames | 4,454 | 0.625 | 0.630 |

Confidence rises monotonically with persistence — 0.425 to 0.625 in the mean — so the detections the model
is least sure about are the ones that fail to survive. On footage that cannot be annotated, track length
is therefore a free filter. It also revises the per-frame reading: 5,101 detections sounded like
instability, but 87 % of them (4,454) belong to tracks of 26 frames or longer, so the detector is more
temporally stable than the raw count suggests. These buckets are derived by hand from the `confidence`
and `track_id` fields; no command emits them.

#### Per-class persistence varies 4.5×

Median track length by class — Glass 32 frames, Paper 18, Plastic 10, Metal 7 — pooled by hand from the
per-clip `per_class` blocks. Metal tracks are the shortest by a wide margin; §8.5 already records white
film being labelled `Metal`, and short-lived Metal predictions are a plausible expression of the same
error, though the connection is not established here.

#### Two methodological notes

**Track ids do not survive a cut.** A montage of concatenated clips gives ByteTrack no boundary to reset
at, so ids leak across scenes: the montage run reported a longest track of 322 frames inside scenes only
~200 frames long. The figures above come from per-clip runs, aggregated afterwards; the montage figure is
recorded only as the caution it is.

**Tracked and untracked counts are not directly comparable.** Tracking adds its own thresholds, so the
tracked runs report 5,101 detections against 5,377 for the same clips without `--track`, 5.1 % fewer.
That is the tracker filtering rather than a change in the detector.

---

## 9. Reproducing

```bash
uv sync                                           # install dependencies
nvidia-smi                                        # confirm the GPU is visible

uv run litterbug validate                         # dataset contracts and split counts
uv run litterbug train --task segment --dry-run    # show resolved config, no GPU cost
uv run litterbug train --task segment              # fine-tune

uv run litterbug val --weights runs/<run>/weights/best.pt --split val
uv run litterbug val --weights runs/<run>/weights/best.pt --split test   # once, at the end
uv run litterbug predict --weights runs/<run>/weights/best.pt --source <image|dir|video>
uv run litterbug predict --weights runs/<run>/weights/best.pt --source <video> --track  # ByteTrack [4]

uv run litterbug analyze --weights runs/<run>/weights/best.pt --split val   # per-instance errors
uv run litterbug analyze --weights runs/<run>/weights/best.pt --split val --iou-mode box
uv run litterbug examples --weights runs/<run>/weights/best.pt --split val  # commented FP/FN figures
uv run litterbug eda                                                        # dataset EDA

uv run pytest                                     # contracts + config
```

`python -m litterbug.cli <command>` is equivalent to `uv run litterbug <command>`.

Hyperparameters are declared once, in `src/litterbug/common/config.py`, and reproduced in §4.3 for
reading convenience. Every run prints its resolved configuration on startup; to see one without
training, use `--dry-run`.

---

## 10. Limitations and threats to validity

Limitations that belong to one result are stated there — the matching convention and confidence floor in
§7.1, the mask-versus-box pair sets in §7.6, the occlusion proxies in §7.5, the equivalence claim and its
parameter mismatch in §6.1, the smoke-test caveat in §4.2, the environment constraint in §4.4, the
unlabelled footage in §8.6 and the tracking caveats in §8.7. What remains are limits that apply across the
whole report.

- **Single seed.** One run per architecture and split, so differences below ~0.01 mAP are not meaningful.
  That floor is the same size as the segmentation-versus-detection difference (§6.1) and as the
  `valid`/`test` gap (§5.4); the first is therefore reported as no detectable difference and the second
  as a coarse estimate, neither as a measurement at that resolution.
- **Scope of the data used.** Only `1_Model_Training_Data` was used, so no figure or metric here includes
  the record's second component, a 366-image conveyor-belt evaluation set with 600 objects (§3). The
  provider cautions that set "should not be used alone to support broad claims about unrestricted
  real-world robustness" — a caution this report inherits, since no broader-conditions evaluation was
  performed.
- **The `test` split was used once, at the end.** Figures from it (§1, §5.4, §6, §7.8) are not ones any
  decision was made on; every other figure comes from `valid`.
- **The training data contains no lighting variation.** §3.1 measures a p10–p90 brightness spread of about
  ten levels on a 0–255 scale, with no image below 60 or above 180. The model has therefore never seen a
  differently lit scene — a concrete and testable reason to expect the transfer failures of §8, rather
  than an appeal to "domain shift" in general.
- **The model cannot decline to detect.** Four classes and no background class means anything unfamiliar
  must be assigned to one of them, confidently. §8.4 shows this happening — a background wall, a worker's
  glove and a conveyor chute each attract a prediction — which is a property of the label set rather than
  of the footage, and it bounds behaviour on any scene the model was not trained for.
- **Reporting conventions.** Where no command writes a number to an artifact, the report says so at the
  point of use and gives the derivation (§4.1, §4.2, §4.4, §7.1, §8.3, §8.7). Everything else is
  reproduced from the files in §12.

---

## 11. References

> **[1]** JITNGERNMADAN, P. et al. **BUU-WOD**: Burapha University Waste Occlusion Dataset for
> high-density waste segmentation. [S.l.]: Kaggle, 2026. Dataset. Licence: CC BY 4.0. Disponível em:
> <https://www.kaggle.com/datasets/visionlab1buu/buu-waste-occlusion-dataset>. Acesso em: 12 set. 2026.

> **[2]** JITNGERNMADAN, P. et al. A sensing-to-sorting multimodal robotic pipeline for waste
> classification using low-cost RGB-D vision and spectral sensors with adaptive Bayesian fusion.
> **IEEE Access**, 2026. DOI: 10.1109/ACCESS.2026.3717685. IEEE Xplore document 11626869. Disponível em:
> <https://doi.org/10.1109/ACCESS.2026.3717685>. Acesso em: 12 set. 2026.

> **[3]** ULTRALYTICS. **Ultralytics YOLO**: version 8.4.148. [S.l.]: Ultralytics, 2026. Software.
> Licence: AGPL-3.0. Disponível em: <https://github.com/ultralytics/ultralytics>. Acesso em: 12 set. 2026.

> **[4]** ZHANG, Y. et al. **ByteTrack**: multi-object tracking by associating every detection box.
> arXiv:2110.06864 [cs.CV], 2021. Version 3, revised 7 Apr. 2022. DOI: 10.48550/arXiv.2110.06864.
> Disponível em: <https://arxiv.org/abs/2110.06864>. Acesso em: 12 set. 2026.

> **[5]** ULTRALYTICS. **yolo26s.pt**; **yolo26s-seg.pt**: COCO-pretrained weights for Ultralytics YOLO26.
> [S.l.]: Ultralytics, 2026. Distributed with [3] under the same licence. Disponível em:
> <https://github.com/ultralytics/ultralytics>. Acesso em: 12 set. 2026. §4.1 reports how many parameter
> tensors transferred from them.

> **[6]** 230274526. [S.l.]: Dreamstime, [s.d.]. 1 online video. Used for the montage. Local file:
> `230274526.webm`.

> **[7]** 232194788. [S.l.]: Dreamstime, [s.d.]. 1 online video. Used for the montage. Local file:
> `232194788.webm`.

> **[8]** 232591973. [S.l.]: Dreamstime, [s.d.]. 1 online video. Used for the montage. Local file:
> `232591973.webm`.

> **[9]** 261034938. [S.l.]: Dreamstime, [s.d.]. 1 online video. Used for the montage. Local file:
> `261034938.webm`.

> **[10]** MIXED waste travels along a conveyor belt inside a recycling plant. [S.l.]: Shutterstock,
> [s.d.]. 1 online video. Clip 4058202301. Used for the close view. Local file:
> `stock-footage-mixed-waste-travels-along-a-conveyor-belt-inside-a-recycling-plant-plastic-bags-bottles-paper.webm`.

> **[11]** MANUAL sorting of waste glass in a mixed waste processing facility. [S.l.]: Shutterstock,
> [s.d.]. 1 online video. Used as a facility counterexample. Local file:
> `stock-footage-konin-poland-september-manual-sorting-of-waste-glass-in-a-mixed-waste-processing-facility.webm`.

> **[12]** MIXED waste drops from a conveyor belt inside a recycling plant. [S.l.]: Shutterstock, [s.d.].
> 1 online video. Used as a facility counterexample. Local file:
> `stock-footage-mixed-waste-drops-from-a-conveyor-belt-inside-a-recycling-plant-plastic-bags-wrappers-paper-and.webm`.

> **[13]** WORKER in special gloves sorting glass garbage on a conveyor belt at a waste sorting plant.
> [S.l.]: Shutterstock, [s.d.]. 1 online video. Used as a facility counterexample. Local file:
> `stock-footage-worker-in-special-gloves-sorting-glass-garbage-on-a-conveyor-belt-at-waste-sorting-plant.webm`.

---

## 12. Appendix — run record

| Item | Value |
| --- | --- |
| Code revision described | `5744ee960bcca200958901f7aee754461fdba194` (2026-09-12) |
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
| Example counts (§7.7, §7.8) | `report/figures/examples-{val,test}/error_examples.json` |
| Figure licence and change statement | `report/figures/README.md` |
| Video montage | `runs/media/demo_montage.mp4` (828 frames, 33.12 s) |
| Tracked demo | `runs/media/demo_tracked.mp4` (828 frames, 33.12 s, track ids) |
| Clean montage input | `runs/media/raw_montage.webm` |
| Track statistics | `runs/predict/track-<clip>/detections.json` (`tracks` block) |
| Per-clip detections | `runs/predict/<clip>/detections.json` |

Environment: Python 3.12.3, torch 2.14.0+cu130, CUDA 13.0, Ultralytics 8.4.148 [3], RTX 3060 Laptop
(5,804 MiB), Ubuntu 24.04 dev container.

Not written to any artifact, and therefore quoted from run stdout: the architecture summary (§4.1), the
Day-1 gate metrics (§4.2) and peak VRAM (§4.4). Derived by hand: the detection-size percentages (§8.3)
and the tracking confidence and per-class persistence buckets (§8.7). Derived by sweeping the `valid`
split under §7.1's own convention: the F1-optimal confidence floor, 0.555.
