# litterbug — Trash Detection & Instance Segmentation on a Conveyor Belt

**Integrantes:** <!-- FILL: names of all group members -->

Instance segmentation and detection of waste objects on a conveyor belt, built on the BUU Waste
Occlusion dataset [1] with Ultralytics YOLO26 [3]. Code revision `5744ee96` (2026-09-12).

---

## 1. Problem and setting

Waste arrives on a conveyor belt in dense, overlapping batches. Annotations trace only the visible
boundary of each object, and objects below roughly 30 % visibility are excluded by the provider's
protocol [1]. Two consequences shape the evaluation:

- Overlap is a labelling convention, not an error: a partially occluded object is correctly annotated
  data.
- Success is not only finding an object but tracing where it stops being visible. That is a harder
  target than detection, and it is what this project measures.

Segmentation is the product. Detection is trained as a comparison baseline on the same splits with the
same schedule, so any difference is attributable to the head rather than to the recipe.

---

## 2. Dataset

**Source.** BUU-WOD (Burapha University Waste Occlusion Dataset), VisionLab, Burapha University,
licensed CC BY 4.0 [1], [2]. No image or annotation was modified. The public record has two components
and this work uses only the first: `1_Model_Training_Data/` (2,000 images at 640×640, YOLO segmentation
polygons), not `2_Experiment_Sample/` (366 images at 640×480, COCO polygons and boxes, 600 objects).
None of those 600 objects appears in any figure or metric here, and the provider states that set "should
not be used alone to support broad claims about unrestricted real-world robustness" — a caution this
report inherits, since no broader-conditions evaluation was performed.

Split counts, as delivered and as reproduced by `uv run litterbug validate`:

| Split | Images | Instances | Instances/image |
| --- | --- | --- | --- |
| train | 1,400 | 17,400 | 12.43 |
| valid | 400 | 4,941 | 12.35 |
| test | 200 | 2,507 | 12.54 |

Classes are `0 Glass`, `1 Metal`, `2 Paper`, `3 Plastic`, in fixed order. The order is a contract with
the annotations and is declared once, in `src/litterbug/common/constants.py`.

**Split policy.** `valid` is the iteration split. `test` was evaluated once, after training and every
model decision had stopped, so none of the reported figures is one a decision was made on.

### 2.1 Exploratory analysis

Produced by `uv run litterbug eda`, which reads the annotations and samples the images rather than
transcribing figures from an earlier run (`runs/dataset-eda/dataset_eda.json`). The dataset is balanced:
the largest-to-smallest class ratio is 1.07 on train and 1.17 on test, and Paper is the largest class in
every split — which independently rules out class imbalance as an explanation for Paper's poor recall,
since Paper is the most numerous class rather than the scarcest (§5.3). No image is unlabelled, no label
row is malformed, and every sampled image is 640×640, three-channel and single-resolution.

**Lighting is invariant.** Mean brightness over a 200-image sample per split has a p10–p90 spread of
about ten levels on a 0–255 scale (median ≈76), and no image falls below 60 or above 180. The dataset is
dim and almost perfectly uniform, so the model has never seen a scene lit differently from this one — a
concrete, testable reason to expect the transfer failures of §6 rather than an appeal to "domain shift"
in general.

---

## 3. Method

`yolo26s-seg` from the Ultralytics YOLO26 segmentation family, "s" size, fine-tuned from the
COCO-pretrained weights in [5]; 830 of 844 parameter tensors transferred. It has 309 layers (136 fused),
11,437,172 parameters, 37.3 GFLOPs and a `Segment26` head, and is declared once in `MODELS`
(`src/litterbug/common/constants.py`), so changing architecture is a one-line edit.

A single epoch ran before committing to the architecture, to check that the segmentation head produced
non-degenerate masks rather than whole-image blobs. It did. Its metrics are not evidence: the gate
reported mask mAP50-95 of 0.159 at epoch 1 against 0.564 at epoch 1 of the 100-epoch run on the same
model, data and seed, because `warmup_epochs=3` and `close_mosaic=10` are absolute epoch counts and both
degenerate in a one-epoch schedule. A smoke test validates the plumbing, never the quality.

The **baseline** is a `yolo26s` detection model trained on the same splits with the same 100-epoch
schedule, seed and `s` size, so the head and the training objective are the only differences.

The schedule was frozen before training and not tuned; it is declared in
`src/litterbug/common/config.py`. 100 epochs (completed, no early stop), patience 20, image size 640,
batch 8, seed 42, AMP and cosine LR, `copy_paste` 0.1, `overlap_mask` True, `mask_ratio` 4,
`close_mosaic` 10, and AdamW at lr 0.00125 with momentum 0.9. That optimizer is the value Ultralytics
resolved, not the value in `args.yaml`: that file records the pre-selection defaults (`optimizer: auto`,
`lr0: 0.01`, `momentum: 0.937`), while the resolved settings appear only in the run log and the `lr/pg0`
column of `results.csv`. Batch size is derived from VRAM by fixed thresholds rather than chosen, and
resolves to 8 on this device. `copy_paste` is enabled because this dataset's difficulty is occlusion.

**Cost.** RTX 3060 Laptop, 5.7 GiB VRAM. Segmentation completed 100 epochs in ≈2.7 h of GPU time, at 86.6
s/epoch measured across epochs 40–100. The run was interrupted and resumed once, which changes how the
timings read: the internal timer resets on resume, so `results.csv` ends at 8,629 s (2.40 h) covering the
second session alone, and 2.7 h is both sessions summed (1,158 + 8,629 = 9,787 s). That is the figure
used against the detection run, which ran 1.80 h uninterrupted and peaked at roughly 2.5 GB against 3.9
GB — so no resource limit can explain §4.4. Data loading ran with `workers=0` because `/dev/shm` is 63
MiB in this container; this affects throughput, not results, but it means 86.6 s/epoch is not a
well-tuned benchmark.

All randomness is seeded (`torch.manual_seed`, `np.random.seed`, `random.seed`, `PYTHONHASHSEED`,
Ultralytics `seed=42`). Every run prints its resolved configuration before touching the GPU and writes
`args.yaml`; `--dry-run` resolves and prints it without spending GPU time. What is *not* pinned, and
therefore limits exact reproduction: `dataset/`, `weights/` and `runs/` are excluded from version
control, the dataset has no recorded checksum, and `pyproject.toml` leaves dependencies unpinned —
versions are those in `uv.lock` [3].

---

## 4. Results

### 4.1 Held-out test split

200 images, 2,507 instances, evaluated once after every model decision had already been made:

| Metric | Box | Mask |
| --- | --- | --- |
| mAP50-95 | 0.837 | 0.796 |
| mAP50 | 0.954 | 0.955 |
| Precision | 0.939 | 0.944 |
| Recall | 0.901 | 0.905 |

Mean IoU over matched pairs is 0.901 for masks against 0.924 for boxes (§5.4). Inference costs 8.6
ms/image on this GPU (≈117 FPS), excluding pre- and post-processing.

The same figures run one to two points higher on `valid`, and that gap is the cost of reporting on data
the model never saw:

| Metric | `valid` | `test` | Change |
| --- | --- | --- | --- |
| Box mAP50-95 | 0.857 | 0.837 | −0.019 |
| Mask mAP50-95 | 0.814 | 0.796 | −0.018 |
| mAP50 | 0.963 | 0.954 | −0.009 |
| Recall (box) | 0.923 | 0.901 | −0.021 |

The gap is about two points of mAP50-95 — the same size as the ~0.01 seed-noise floor established in
§4.4 — so the model is not overfitted to `valid`, though the estimate is too coarse to read finely.
Where the drop sits is more informative than its size: recall on the three larger size quartiles is flat
between splits to within half a point, while the smallest falls by 0.069 (0.8947 → 0.8262). The model has
not degraded uniformly on unseen data; it has lost small objects specifically, the same failure §5.2
identifies as dominant on `valid`, now amplified.

Training converged rather than being cut off by the schedule: the best checkpoint is epoch 89 (mask
mAP50-95 0.81451), and epoch 100 is 0.0016 below it, so patience never triggered. Over half the total
quality arrives in the first 10 epochs (box mAP50-95 0.589 → 0.740).

### 4.2 Per-class results (`valid`)

| Class | Box mAP50-95 | Mask mAP50-95 | Box − Mask |
| --- | --- | --- | --- |
| Glass | 0.892 | 0.844 | 0.048 |
| Metal | 0.881 | 0.832 | 0.049 |
| Plastic | 0.861 | 0.819 | 0.042 |
| Paper | 0.791 | 0.763 | 0.028 |

Paper is the weakest class in both heads: 0.070 below the next-weakest (Plastic, 0.861 box) and 0.101
below the strongest (Glass, 0.892). That gap exceeds the box/mask difference, and neither Glass nor
Plastic is a weak class here. Early-epoch rankings should not be trusted: at the 1-epoch gate Glass was
the worst class, consistent with an unconverged head rather than any property of Glass.

### 4.3 Box head versus mask head, within the segmentation model

Mask mAP50-95 is 0.042 below box mAP50-95 on `valid` (0.814 against 0.857), and the ranking of classes is
identical in both heads: the mask head adds outline precision at modest cost and does not reorder class
difficulty. An independent metric agrees — mean IoU over matched pairs is 0.9074 for masks against 0.9291
for boxes, a 0.022 gap in the same direction and of the same order (§5.4). Two metrics that measure
different things agree in direction and magnitude. This is a within-model comparison, however; it is not a
segmentation-versus-detection result, which requires the baseline of §4.4.

### 4.4 Detection baseline

| Split | Detection only | Segmentation (box head) | Difference |
| --- | --- | --- | --- |
| `valid` | 0.8583 | 0.8566 | 0.0017 |
| `test` | 0.8376 | 0.8372 | 0.0004 |

Both differences are far below the ~0.01 floor that one seed per model can resolve, so the two box heads
are indistinguishable on either split, and the held-out split repeats the `valid` result rather than
narrowing it. Per-class differences on `valid` agree, with mixed signs and every value inside ±0.01.
Adding the mask head at the same `s` scale therefore cost no measurable box accuracy: segmentation yields
masks at 0.796 mAP50-95 on unseen data while its box head stays indistinguishable from a model trained on
boxes alone.

Two qualifications bound that claim. The models are not parameter-matched: segmentation adds 1,486,212
parameters (+14.9 %) and 14.5 GFLOPs (+63.6 %) over detection (11,437,172 against 9,950,960 parameters;
37.3 against 22.8 GFLOPs). "Same size" means the same `s` scale, not equal capacity, so the defensible
form is "no measurable detection cost at 14.9 % more parameters", not "a free mask head". And precision
and recall traded against each other — segmentation is slightly more precise (0.948 against 0.939) and
slightly less sensitive (0.923 against 0.931) on `valid` — so a single mAP number hides two
countervailing behaviours.

---

## 5. Error analysis

### 5.1 How the numbers were derived

Ultralytics computes per-instance matches internally but does not expose them, so the confusion matrix
normally exists only as a PNG; this analysis re-derives the matching from scratch. Predictions are
assigned by descending confidence, greedily and one-to-one, at mask IoU ≥ 0.5. Matching is class-agnostic
by design: a Paper prediction landing on a Plastic object must register as a confusion, not as a miss
plus a false positive.

**The confidence floor is a free parameter, and this is where it was set.** Sweeping it over `valid`
under exactly that convention puts the F1 optimum at **0.555** (F1 0.9559, precision 0.9690, recall
0.9431). This analysis runs lower, at 0.34 — F1 0.9452, precision 0.9282, recall 0.9628 — giving up 0.011
of F1 for 0.020 of recall. The trade is deliberate: the questions below concern which objects the model
fails to *find*, and a higher floor would start manufacturing misses rather than measuring them. The
bundled `BoxF1_curve.png` and `MaskF1_curve.png` peak in the same region, 0.49–0.56. At that floor,
4,757 of 5,125 predictions matched 4,941 ground-truth instances.

Metrics are given to three decimals, except IoU and gaps below 0.01, which take four; deltas are
recomputed from artifact values rather than rounded table figures, which is why the box/mask gap in §4.3
is 0.042 and not the 0.043 that subtracting two rounded headline numbers would give.

**Confusion between classes is negligible.** Across 4,941 instances the largest off-diagonal cell is 31
(Paper predicted as Metal), and Glass↔Plastic confusion accounts for 12 errors (0.24 %). No class pair is
a meaningful error source: the error budget is dominated by misses.

### 5.2 Size is the dominant driver of misses (`valid`)

| Size quartile (ground-truth mask px) | Instances | Recall |
| --- | --- | --- |
| q1 — 15 to 3,600 | 1,234 | 0.895 |
| q2 — 3,600 to 5,713 | 1,235 | 0.979 |
| q3 — 5,713 to 9,030 | 1,236 | 0.985 |
| q4 — 9,030 to 59,264 | 1,236 | 0.992 |

Recall differs by ten points between the smallest and largest quartile, with almost all of the loss
concentrated in q1. Small instances are the largest single source of error in this model, larger than
every class confusion combined.

### 5.3 Paper is intrinsically hard, not merely small (`valid`)

Paper has the weakest recall (0.885). The obvious explanation is that paper objects are small; the data
says otherwise. Recall by class within the same size quartiles:

| Class | Instances | Median size (px) | q1 | q2 | q3 | q4 |
| --- | --- | --- | --- | --- | --- | --- |
| Glass | 1,203 | 4,982 | 0.953 | 0.990 | 0.997 | 1.000 |
| Metal | 1,198 | 4,570 | 0.936 | 0.989 | 0.996 | 0.994 |
| Plastic | 1,253 | 7,914 | 0.874 | 0.970 | 0.982 | 0.998 |
| Paper | 1,287 | 6,116 | 0.798 | 0.957 | 0.967 | 0.980 |

Paper's instances are not the smallest — its median (6,116 px) exceeds Glass (4,982) and Metal (4,570) —
so the weakness is not a size artefact. And Paper is the worst class in *every* quartile: in q1 it trails
Glass by 15.5 points, Metal by 13.8 and Plastic by 7.6, and in q2–q4 by 1.3–3.3. Size degrades every
class, but it does not explain Paper. Its errors also skew to misses rather than confusion (96 missed
against 52 confused), and its largest confusion is with Metal (31), not Plastic (19). Class imbalance is
ruled out separately in §2.1.

### 5.4 Mask quality — IoU (`valid`)

Recall and precision say whether an object was found; IoU says whether it was traced correctly. Across
the 4,757 matched pairs, mean IoU is 0.9074 (median 0.9278, p10 0.8391, p90 0.9597) and 95.7 % of pairs
score ≥ 0.75. Per class, over true positives only, Paper is worst in both modes (0.8953 mask against
0.9120 box) — a second metric singling out the same class as §5.3. IoU also degrades with size, so small
instances are penalised twice over: missed more often (§5.2) and outlined less accurately when found. The
mask head's cost is concentrated there too, 0.032 IoU on the smallest quartile against 0.014 on the
largest. One caveat: the two runs match under different criteria (mask IoU ≥ 0.5 against box IoU ≥ 0.5),
so the pair sets differ — 4,757 against 4,747, with recall and precision marginally apart for the same
reason. Immaterial to the conclusion, but the columns are not computed over the same pairs.

### 5.5 False positives are two different failures

Matching is one-to-one, so a second prediction on an already-detected object cannot match and is scored
as a false positive even though it covers a real object. Those are duplicates, not hallucinations, and
they are a different problem from predicting on bare belt.

| Category | Count | Share |
| --- | --- | --- |
| Matched | 4,757 | — |
| Missed instances | 184 | — |
| Confused (matched, wrong class) | 135 | — |
| False positives — duplicates | 156 | 42 % of FPs |
| False positives — over background | 212 | 58 % of FPs |

42 % of the model's false positives are duplicate detections, so precision is not one behaviour described
by one number: 0.9282 is 212 background errors plus 156 redundant masks, which have different causes and
would need different remedies. Suppressing duplicates alone, with recall held fixed, would move precision
to 0.9573 — an upper bound on what that could buy rather than a result, since no simple threshold
achieves it cleanly. The `analyze` output cannot make this distinction; only re-examining overlap against
all instances separates the two.

### 5.6 The held-out split replicates the analysis

Recall falls 1.7 points (0.9628 → 0.9454) and mask IoU 0.6 (0.9074 → 0.9010); precision is fractionally
higher (0.9301). Paper remains the weakest class on `test` in both metrics (recall 0.8432, IoU 0.8876),
so the claim of §5.3 holds on data the model never saw — which matters, because it rested on the
iteration split alone. Glass improves (0.9709 → 0.9846), so the class differences are not uniform
shrinkage toward the mean. The false-positive decomposition replicates almost exactly — 156 duplicates
and 212 over background from 368 on `valid`, 75 and 103 from 178 on `test`, 42 % / 58 % on both — so
duplicate detections are a stable property of this model rather than an artefact of the split.

---

## 6. Out-of-distribution generalisation (video)

Eight third-party stock clips were evaluated, four close belt views and four facility views. Nothing was
applied beyond concatenation, and the slowed variants use frame duplication rather than interpolation, so
no synthetic frame entered the model. Deliverable: `runs/media/demo_montage.mp4` — 828 frames at 580×326,
25 fps, 33.12 s at native speed, recording 5,377 detections.

**The model transfers to a matching viewpoint, and not otherwise.** On the close views detections land on
the waste: green plastic bags → `Plastic 0.87`, corrugated cardboard → `Paper 0.86`, glass bottles in a
kerbside bin → `Glass 0.71`. On the facility views they land on the scene instead: `Metal 0.70` across a
background wall over a belt of glass, `Metal 0.48` on a worker's yellow glove, `Paper 0.55` on large ochre
background regions. Same weights, same model — the variable is the view, not the video. Motion blur and
compression are secondary; the model fails when the scene departs from the training composition, which
contains only unobstructed views of one belt.

The mechanism belongs to the label set rather than to video: with four classes and no way to say "not
waste", the model's implicit background model is "the BUU belt", so anything unfamiliar is forced into
one of the four, confidently. It would recur in any installation with guarding, hands or unfamiliar
conveyor furniture, and the invariant lighting of §2.1 is a concrete reason to expect it.

**The drop cannot be quantified.** Every clip is unlabelled and mAP requires ground truth, so any
percentage would be fabricated. Tracking needs no labels and was measured instead: ByteTrack [4]
collapses 5,101 per-frame detections into 138 tracks, and mean confidence rises monotonically with track
length — 0.425 for single-frame detections against 0.625 for tracks of 26 frames or more, which hold
87 % of all detections. Track length is a free reliability signal on footage that cannot be annotated.

---

## 7. Limitations and next steps

Limitations that belong to one result are stated there — the matching convention and confidence floor
(§5.1), the mask-versus-box pair sets (§5.4), the equivalence claim and its parameter mismatch (§4.4),
the smoke-test caveat (§3), the environment constraint (§3). What remains applies across the whole
report.

- **Single seed.** One run per architecture and split, so differences below ~0.01 mAP are not meaningful.
  That floor is the same size as the segmentation-versus-detection difference (§4.4) and as the
  `valid`/`test` gap (§4.1), so the first is reported as no detectable difference and the second as a
  coarse estimate, neither as a measurement at that resolution.
- **Scope of the data used.** Only `1_Model_Training_Data/` was used, so no figure or metric here
  includes the record's second component (§2). The provider's caution against broad robustness claims is
  inherited, since no broader-conditions evaluation was performed.
- **The `test` split was used once, at the end.** Figures from it (§4.1, §4.4, §5.6) are not ones any
  decision was made on; every other figure comes from `valid`.
- **True occlusion is not measurable from this data.** Annotations trace only visible boundaries, and
  objects below ~30 % visibility were excluded rather than annotated with their hidden geometry. Of two
  proxies, mask area is strong and monotonic (q1 recall 0.895 against q4 0.992) while solidity is weak
  and non-monotonic (0.924 against 0.963) — consistent with it tracking material rather than occlusion,
  since crumpled paper and plastic film score low whether or not they are occluded. No
  occlusion-stratified claim is made anywhere in this report.
- **The training data contains no lighting variation** (§2.1), so the model has never seen a differently
  lit scene — a testable reason to expect the transfer failures of §6, rather than an appeal to "domain
  shift" in general.
- **The model cannot decline to detect.** Four classes and no background class means anything unfamiliar
  must be assigned to one of them, confidently. §6 shows this happening — a background wall, a worker's
  glove and a conveyor chute each attract a prediction — which is a property of the label set rather than
  of the footage, and it bounds behaviour on any scene the model was not trained for.

**Next steps.**

1. **Establish why Paper is hard.** Size (§5.2), class imbalance (§2.1) and Plastic confusion are all
   ruled out. The remaining candidates — low-contrast material appearance, annotation density, genuine
   occlusion — are not distinguished by anything measured here, and would need either the pre-filtering
   full-object annotations or a targeted sampling of Paper instances.
2. **Measure occlusion properly**, which requires the full-object annotations removed before release, or
   a proxy validated against data where occlusion is known.
3. **Quantify the video drop.** Hand-annotate 20–30 sampled frames and compute precision and recall
   against them, or obtain labelled footage from a comparable viewpoint. Track persistence (§6) is not a
   substitute.
4. **Replicate with multiple seeds** to separate real differences from the ~0.01 floor, which currently
   limits both §4.4 and §4.1.
5. **Test composition transfer on labelled data** — the record's second component, or a facility-view
   clip annotated in the four classes — to convert §6's qualitative finding into a number.

---

## 8. Reproducing

```bash
uv sync                                             # install dependencies
uv run litterbug validate                           # dataset contracts and split counts
uv run litterbug train --task segment --dry-run      # show resolved config, no GPU cost
uv run litterbug train --task segment                # fine-tune
uv run litterbug val --weights runs/<run>/weights/best.pt --split test   # once, at the end
uv run litterbug analyze --weights runs/<run>/weights/best.pt --split val
uv run litterbug predict --weights runs/<run>/weights/best.pt --source <video> --track
uv run litterbug eda
uv run pytest                                        # contracts + config
```

**Dataset.** BUU-WOD, Kaggle `visionlab1buu`, CC BY 4.0 —
<https://www.kaggle.com/datasets/visionlab1buu/buu-waste-occlusion-dataset> [1].

Hyperparameters are declared once, in `src/litterbug/common/config.py`, and every run prints its resolved
configuration on startup; `--dry-run` shows one without training. Dependencies are those in `uv.lock`.
Machine-readable metrics, error analyses and EDA output are listed in the repository README.

---

## 9. References

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
> <https://github.com/ultralytics/ultralytics>. Acesso em: 12 set. 2026.

> **[6]** MIXED waste travels along a conveyor belt inside a recycling plant. [S.l.]: Shutterstock,
> [s.d.]. 1 online video. Clip 4058202301. Used for the close view.

> **[7]** MANUAL sorting of waste glass in a mixed waste processing facility. [S.l.]: Shutterstock,
> [s.d.]. 1 online video. Used as a facility counterexample.

> **[8]** MIXED waste drops from a conveyor belt inside a recycling plant. [S.l.]: Shutterstock, [s.d.].
> 1 online video. Used as a facility counterexample.

> **[9]** WORKER in special gloves sorting glass garbage on a conveyor belt at a waste sorting plant.
> [S.l.]: Shutterstock, [s.d.]. 1 online video. Used as a facility counterexample.

> **[10]** 230274526; 232194788; 232591973; 261034938. [S.l.]: Dreamstime, [s.d.]. 4 online videos.
> Close-view clips composing `runs/media/demo_montage.mp4`.
