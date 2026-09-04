# Weakest classes per split

60-epoch bf16 `none` runs (UniMatch V2, DINOv2-S, Pascal), one seed per split, launched
2026-09-02, assembled 2026-09-03. Per-class values are **EMA IoU at the epoch of the run's
best EMA mIoU** (the "best EMA only" metric). Runs: `92_60ep_none_seed12000`,
`183_60ep_none_seed12001`, `366_60ep_none_seed12002`, `732_60ep_none_seed12003`,
`1464_60ep_none_seed12004` under `exp/pascal/unimatch_v2_1gpu_bf16/dinov2_small/`.

## Bottom five classes at each split

| Split | Best EMA | Epoch | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|---|---|
| 92 | 82.42 | 58 | chair 28.5 | sofa 34.2 | dining table 66.7 | potted plant 68.8 | tv/monitor 71.4 |
| 183 | 84.01 | 53 | chair 36.8 | sofa 37.4 | potted plant 67.7 | dining table 72.8 | tv/monitor 74.6 |
| 366 | 86.07 | 58 | chair 48.1 | sofa 63.5 | potted plant 65.9 | dining table 77.2 | tv/monitor 79.3 |
| 732 | 87.02 | 50 | chair 53.6 | sofa 64.2 | potted plant 73.7 | dining table 77.0 | bicycle 79.4 |
| 1464 | 87.72 | 48 | chair 55.8 | sofa 68.0 | potted plant 73.7 | dining table 77.7 | bicycle 80.1 |

## Trajectory of the classes that carry the split gap

| Class | 92 | 183 | 366 | 732 | 1464 | 92 to 1464 |
|---|---|---|---|---|---|---|
| chair | 28.5 | 36.8 | 48.1 | 53.6 | 55.8 | +27.3 |
| sofa | 34.2 | 37.4 | 63.5 | 64.2 | 68.0 | +33.8 |
| tv/monitor | 71.4 | 74.6 | 79.3 | 80.9 | 85.4 | +14.0 |
| dining table | 66.7 | 72.8 | 77.2 | 77.0 | 77.7 | +11.0 |
| potted plant | 68.8 | 67.7 | 65.9 | 73.7 | 73.7 | +4.9 |
| bicycle | 80.2 | 80.9 | 79.7 | 79.4 | 80.1 | −0.1 |

Reading:

- The mIoU curve flattens hard: 92 to 366 gains 3.7 points, 366 to 1464 gains 1.7.
  Almost the whole 92-to-1464 gap sits in four classes: chair, sofa, tv/monitor and
  dining table. Every other class is within about two points across all five splits.
- Potted plant is flat around 66 to 74 and bicycle around 80 at every split; they are weak
  but do not respond to more labels.
- No run collapsed: cat and cow, the fingerprint classes of the split-92 failures, read
  96 to 97 everywhere.
- The nine OSR-Adam $\beta = 0.99$ learning-rate cells at split 366 (seeds 12005 to 12013)
  have the same three weakest classes as the baseline: chair 46.6 to 52.4, sofa 56.1 to
  67.6, then potted plant or dining table 66.4 to 71.2. See
  `OSR-Adam-Beta-099-Split-366.md`.

Source: per-run W&B pull `wandb_60ep.json` (job 1139d423 scratch, 2026-09-03 11:33), field
`cls_at_best`. Verified 2026-09-04 against the `Class [..] IoU: .., EMA: ..` lines of each
run's train log at the best-EMA epoch: identical to two decimals at all five splits. Two
values sit on a rounding boundary and appear 0.1 lower in tables parsed from the logs:
sofa at 92 is 34.15 and chair at 366 is 48.05.

## How the weak classes fail: merged concept versus fading into background

From the 21×21 pixel confusion matrices of `best_ema.pth` (EMA teacher, Pascal val) for the
split-92 and split-1464 `none` runs, computed 2026-09-03 with `confusion_matrix.py`. The
matrices reproduce the logged mIoU exactly (82.42 and 87.72) and are saved as
`confusion_best_ema.npz` next to each checkpoint. Written up in the 2026-09-03 session
(13:38Z and 13:54Z); copied here on 2026-09-04.

### Where the errors go, split 92

Each class's misses (false negatives) and false alarms (false positives), split by whether
they land on background or on another foreground class:

| Class | Misses to background | Misses to foreground | Top partner | False alarms from background | False alarms from foreground |
|---|---|---|---|---|---|
| sofa | 7% | 93% | chair, 90% of misses | 59% | 41% |
| chair | 17% | 83% | sofa, 68% of misses | 41% | 59% |
| dining table | 38% | 62% | chair, 25% | 93% | 7% |
| tv/monitor | 87% | 13% | person, 9% | 78% | 22% |
| potted plant | 93% | 7% | sofa, 3% | 93% | 7% |
| bicycle | 80% | 20% | person, 17% | 80% | 20% |
| motorbike | 48% | 52% | person, 46% | 35% | 65% |

Sofa and chair leak almost entirely into each other. Potted plant, tv/monitor and bicycle
leak almost entirely into background; the small foreground remainder is the rider or the
person sitting in front of the screen, which is occlusion, not confusion.

### The sofa/chair pair, split 92 versus 1464

| | Split 92 | Split 1464 |
|---|---|---|
| GT sofa predicted as chair | 50.0% | 9.5% |
| GT sofa predicted as sofa | 44.6% | 81.9% |
| GT sofa predicted as background | 3.8% | 7.5% |
| chair precision | 32.4% | 69.0% |
| of predicted-chair pixels, actually sofa | 37.7% | 14.6% |

At 92 the model has effectively one merged sofa/chair concept that outputs "chair": chair is
over-predicted at more than twice its ground-truth area, and the low chair IoU at 92 is mostly
a precision problem caused by sofa pixels. The sofa and chair failures at 92 are a single
failure.

### The merge test

A merged concept shows a *mutual* leak with complementary damage: chair's precision collapses
to 32% because it swallows sofa, sofa's recall collapses to 45% because chair swallowed it.
The decisive check is to score the pair as one class:

| | Split 92 | Split 1464 |
|---|---|---|
| IoU(sofa $\cup$ chair) as one class | 68.3 | 76.6 |
| IoU sofa, separate | 34 | 68 |
| IoU chair, separate | 28 | 56 |

The 92 model finds seating furniture nearly as well as the 1464 model does; the whole gap is
in splitting it. A background-boundary class has nothing to merge with: no foreground partner
exceeds a few percent, and its gain with labels is one-sided, recall for tv/monitor,
precision for dining table.

Dining table at 92 is a mixed case. Its over-prediction is against background (93% of false
alarms), but a quarter of its misses go to chair, which at 92 acts as the sink for all indoor
furniture: 38% of predicted chair is sofa and some is table.

### Bicycle versus motorbike: essentially none

Ground-truth bicycle predicted as motorbike is 0.06% at 92 and 0.02% at 1464; ground-truth
motorbike predicted as bicycle is 0.13% and 0.21%. Bicycle's misses are background at 7.8% of
its area and person at 1.7%, with the same two on the false-alarm side, giving recall 90 and
precision 88 at every split. Motorbike's leak at 92 was to car and bus, not bicycle. Two
labeled bicycle images are enough for DINOv2 features to keep the two apart.

### What the matrix does not tell you

"Background" errors could be a thin band along object edges or whole objects missed and
hallucinated. A pixel confusion matrix cannot separate the two, so "boundary problem" is an
interpretation, not a measurement. For bicycle and potted plant it is the standard
explanation: recall and precision are symmetric near 90 and 88, and the ceiling does not move
with labels, which fits an annotation-resolution floor around spokes and leaves. For
tv/monitor at 92 the asymmetry, recall 80 against precision 87, fixed by more labels, fits
whole dark screens being missed rather than edges. Settling it needs a trimap evaluation
(errors within a few pixels of a ground-truth boundary versus interior errors), which means
re-running inference since only the matrices were saved. Not run as of 2026-09-04.

### Consequence

Any method that meaningfully improves mIoU at a split has to move these classes without
hurting the others, and concretely has to separate chair from sofa better. The related
collapse-mechanism work on split 92 (`seed_attribution_experiment/notes.md`, the pseudo-label
audit and the OSR/VLR rows) draws the same line from the other side: substitution (cat
relabelled as dog) responds to OSR-class, absorption into background (bicycle, tv/monitor)
responds to VLR-class.

## Frozen DINOv2 as a head-free classifier: k-NN and class-mean probes

Measured 2026-09-03 (session 15:37Z to 17:10Z) to answer: do the pretrained features tell
sofa from chair when the fine-tuned split-92 model does not? Code: `feature_probe.py` (repo
root, untracked); saved confusion matrices: `analysis_outputs/feature_probe_layer11.npz`.
The head swap that motivated it is `head_swap.py` with
`analysis_outputs/head_swap_seed12000_92.npz`. The per-class table reproduces without GPU
work via

```
python feature_probe.py --report-npz analysis_outputs/feature_probe_layer11.npz --clf knn --classes
```

### Setup

- **Trunks.** Pretrained DINOv2-S; "merged" = trunk of `92_60ep_none_seed12000`
  `best_ema.pth` (sofa/chair merged, sofa 34.2 / chair 28.5); "separated" = trunk of
  `92_20ep_none_seed12000` `best_ema.pth` (same seed, compressed 20-epoch schedule, sofa 49.5
  / chair 39.8). Each trunk is frozen and every image is run through it; features are the
  layer-11 patch tokens (one layer, where the DPT head reads four).
- **Bank.** Every $14 \times 14$ patch of the labeled images of one split whose block is at
  least 75% one class contributes its feature vector and label. The 92 bank holds five sofa
  images' worth of patches, the 1464 bank 93.
- **k-NN.** For each val patch, the $k = 20$ bank patches with the highest cosine
  similarity vote for their class, weighted by similarity. Uses the *local* structure of the
  feature space.
- **Class mean.** Average the bank patches of each class into one vector; each val patch
  takes the class whose mean it is most similar to. Uses only the *global* geometry, and is
  the head-free analogue of a $1 \times 1$ conv head and of the class-gradient prototypes.
- Per-class similarity maps are upsampled bilinearly to pixel resolution and argmaxed, then
  scored exactly like the trainer's evaluation (ignore pixels dropped). Patch-level
  prediction carries a boundary penalty, so absolute IoUs run below the model's.

### Sofa and chair under the probes

| Trunk | Bank | k-NN sofa / chair | k-NN mIoU | GT sofa to chair, k-NN | GT sofa to chair, class mean |
|---|---|---|---|---|---|
| pretrained | 92 | 28.3 / 23.6 | 55.0 | 35% | 63% |
| pretrained | 1464 | 53.5 / 40.0 | 72.9 | 9% | 8% |
| merged (60-ep) | 92 | 34.9 / 27.0 | 76.2 | 48% | 49% |
| merged (60-ep) | 1464 | 55.1 / 39.0 | 80.5 | 18% | 38% |
| separated (20-ep) | 92 | 48.8 / 35.0 | 73.8 | 19% | 19% |
| separated (20-ep) | 1464 | 58.7 / 41.8 | 80.2 | 11% | 15% |

- **The pretrained features do separate the pair, given enough reference.** With the 1464
  bank, raw DINOv2 k-NN reaches sofa 53.5 and chair 40.0 with 9% leakage, matching the best
  fine-tuned run and far above the merged model's 34 / 28.5 at 50%.
- **But not from five sofa images.** With the 92 bank the raw features give sofa 28 and 35%
  leakage, and the class-mean classifier leaks 63%. Fine-tuning is what makes the features
  linearly usable at 92: class-mean mIoU with the 92 bank is 45.7 on the pretrained trunk,
  75.2 on the merged trunk and 73.1 on the separated one (49.1 / 75.2 / 73.4 with the 1464
  bank; `--clf ncm`). The merged basin pays for that with the pair.
- **The head adds nothing for the pair.** k-NN on each fine-tuned trunk with the 92 bank
  reproduces its full model: merged 34.9 / 27.0 at 48% against the model's 34.2 / 28.5 at
  50%; separated 48.8 / 35.0 at 19% against 49.5 / 39.8 at 13.5%. The head-swap result was not
  a head artefact.
- **The merged trunk still holds the information locally; its class means have collapsed.**
  With the 1464 bank, k-NN on the merged features separates the pair nearly as well as on the
  separated trunk (55 / 39 at 18%), while the class-mean classifier on the same features leaks
  38% against 8% pretrained and 15% separated. Every linear consumer downstream, the
  $1 \times 1$ head and the gradient prototypes alike, reads exactly the geometry that
  collapsed.

### Scarcity versus damage: what happens when the bank grows

| Trunk | class-mean leak, 92 to 1464 bank | k-NN leak, 92 to 1464 bank |
|---|---|---|
| pretrained | 63% to 8% | 35% to 9% |
| separated (20-ep) | 19% to 15% | 19% to 11% |
| merged (60-ep) | 49% to 38% | 48% to 18% |

A k-NN-versus-class-mean gap is generic (k-NN wins whenever a class is multimodal or its mean
is badly estimated), so the gap itself proves nothing; its response to more reference data
does. **Pretrained at 92 is a mean-estimation problem**: five sofas define a bad center in a
space with much non-semantic variance, and 93 sofas fix it completely (both classifiers
converge at 8 to 9%). **Merged at 1464 is a geometry problem**: with the mean well estimated,
class-mean leak stays at 38% because the two centers really are close. The merged trunk's
local structure is also degraded (k-NN leak 18%, twice the pretrained 9% and separated 11%),
and at the 92 bank it shows no gap at all (48% vs 49%): the five training sofas were fit as
special cases and val sofas land near chairs under either classifier. The first is damage,
the second is scarcity.

### Per-class k-NN IoU, six scenarios

Layer-11 features, $k = 20$, with the full fine-tuned models' best EMA at 92 and 1464
labels for reference. Sorted by the pretrained-1464 column.

| Class | pre-92 | pre-1464 | merged-92 | merged-1464 | sep-92 | sep-1464 | model 92 | model 1464 |
|---|---|---|---|---|---|---|---|---|
| background | 90.7 | 93.2 | 93.7 | 94.9 | 92.9 | 94.6 | 95.9 | 97.0 |
| bird | 76.8 | 86.0 | 86.3 | 88.5 | 84.8 | 86.5 | 94.7 | 95.4 |
| person | 82.9 | 85.7 | 86.2 | 88.4 | 86.6 | 88.2 | 90.0 | 92.3 |
| bus | 72.5 | 84.8 | 91.2 | 91.8 | 90.5 | 91.9 | 95.5 | 95.7 |
| train | 72.4 | 83.6 | 89.1 | 89.9 | 89.8 | 90.4 | 92.2 | 93.0 |
| aeroplane | 75.2 | 82.1 | 80.3 | 83.9 | 77.3 | 82.1 | 93.1 | 91.8 |
| cat | 45.9 | 81.8 | 93.3 | 93.0 | 91.7 | 92.4 | 96.6 | 96.8 |
| dog | 50.6 | 80.1 | 90.5 | 91.6 | 88.6 | 91.0 | 94.5 | 95.2 |
| car | 71.1 | 79.9 | 86.1 | 87.0 | 85.8 | 87.5 | 89.5 | 90.4 |
| motorbike | 61.2 | 78.8 | 82.6 | 86.0 | 74.9 | 85.0 | 89.0 | 93.0 |
| bottle | 58.7 | 74.9 | 80.8 | 81.8 | 80.4 | 80.6 | 83.8 | 87.8 |
| horse | 53.8 | 74.9 | 85.9 | 88.3 | 83.3 | 86.6 | 94.5 | 95.1 |
| boat | 66.2 | 72.8 | 77.2 | 78.5 | 72.9 | 76.9 | 83.2 | 86.2 |
| sheep | 51.2 | 72.3 | 85.1 | 86.7 | 80.4 | 85.8 | 92.5 | 94.9 |
| cow | 35.9 | 70.9 | 90.3 | 91.3 | 88.7 | 90.4 | 96.3 | 96.5 |
| tv/monitor | 51.0 | 70.2 | 68.6 | 75.2 | 66.6 | 74.8 | 71.4 | 85.4 |
| dining table | 47.0 | 59.5 | 63.3 | 73.7 | 67.7 | 73.2 | 66.7 | 77.7 |
| bicycle | 0.5 | 53.6 | 43.9 | 63.6 | 2.6 | 61.6 | 80.2 | 80.1 |
| sofa | 28.3 | 53.5 | 34.9 | 55.1 | 48.8 | 58.7 | 34.1 | 68.0 |
| potted plant | 40.5 | 52.4 | 64.5 | 63.3 | 60.7 | 63.3 | 68.8 | 73.7 |
| chair | 23.6 | 40.0 | 27.0 | 39.0 | 35.0 | 41.8 | 28.5 | 55.8 |
| **mIoU** | **55.0** | **72.9** | **76.2** | **80.5** | **73.8** | **80.2** | 82.4 | 87.7 |

| | mIoU over 21 classes | mIoU over 20 classes, bicycle dropped |
|---|---|---|
| pretrained k-NN, 92 bank | 55.0 | 57.8 |
| pretrained k-NN, 1464 bank | 72.9 | 73.9 |
| merged k-NN, 92 bank | 76.2 | 77.8 |
| separated k-NN, 92 bank | 73.8 | 77.4 |

- **Fine-tuning organizes the features for almost everything.** Raw DINOv2 with five to
  fifteen reference images fails on cat 45.9, cow 35.9, dog 50.6, sheep 51.2; either
  fine-tuned trunk puts them at 85 to 93 from the same 92 references. Growing the bank from 92
  to 1464 is worth +17.9 mIoU on the pretrained trunk but only +4.3 and +6.3 on the fine-tuned
  ones.
- **The merged trunk is the better trunk on 20 of 21 classes** against pretrained at the
  1464 bank (it loses only chair, by 1.1, and its sofa 55.1 beats pretrained 53.5). What the
  merged basin did is improve every class except the pair, collapse the pair's class means,
  and make the five training sofas useless as references (34.9 against the separated trunk's
  48.8 at the 92 bank).
- **Against the separated trunk it is a different trade-off, not a worse trunk.** At 92
  references it trails on sofa by 13.9 and chair by 8.0, and leads on bicycle, motorbike,
  aeroplane and potted plant by 4 to 41; overall 76.2 against 73.8, full models 82.4 against
  82.8.
- **The head adds nothing for the pair and a lot for thin classes.** Model-92 minus
  merged-92 is 0 on sofa and 1.5 on chair, against +36 on bicycle, +13 on aeroplane, +9 on
  horse. Bicycle's near-zero probe scores at the 92 bank (0.5 pretrained, 2.6 separated) are a
  purity artefact: thin frames leave almost no 75%-pure $14 \times 14$ blocks. Aeroplane is
  not that: its k-NN score barely moves with references and sits about 10 under the model at
  both bank sizes, which fits patch-resolution loss on wing edges, tails and landing gear that
  the four-layer DPT head recovers (horse and sheep show the same shape). Likely, not verified
  with a trimap.
- **tv/monitor and dining table are reference-limited, sofa and chair are trunk-limited.**
  With the 1464 bank the merged trunk's own features reach tv 75.2 and table 73.7, above what
  the 92-trained head extracts from them (71.4, 66.7). For sofa and chair the 1464 bank on the
  merged trunk gives 55 and 39, still well short of the 1464-trained model's 68 and 56. More
  references cannot get the 92 trunk there; only a different trunk can.

### Context: the merge lives in the trunk

Head swap between the merged and separated checkpoints (seed 12000, split 92, Pascal val):

| Trunk | Head | sofa | chair | table | mIoU | GT sofa to chair |
|---|---|---|---|---|---|---|
| merged | merged | 34.2 | 28.5 | 66.7 | 82.42 | 50.0% |
| merged | separated | 36.7 | 29.5 | 65.6 | 81.98 | 44.5% |
| separated | merged | 48.6 | 34.6 | 71.9 | 82.97 | 24.1% |
| separated | separated | 49.5 | 39.8 | 70.6 | 82.75 | 13.5% |

Giving the merged trunk the good head recovers 2.5 points of sofa; giving the good trunk the
merged head recovers 14.4 and cuts the leak from 50% to 24%. Same pattern as
`head_vs_trunk.md` found for the cat collapse.

Frozen-trunk training runs (`--lock-backbone`, seed 12000, split 92, epoch 19 of the
60-epoch schedule) then showed the merge needs a trainable trunk:

| Run | sofa | chair | cat | mIoU |
|---|---|---|---|---|
| trunk frozen, semi-supervised | 51.1 | 32.2 | 95.8 | 81.8 |
| trunk frozen, sup-only | 30.9 | 21.0 | 87.9 | 68.0 |
| trunk trainable, semi-supervised (60-ep run at ep 19) | 26.5 | 27.3 | 94.9 | 81.2 |
| trunk trainable, sup-only | 28.9 | 23.8 | 78.2 | 64.9 |

Sofa under the frozen trunk with unlabeled data rises monotonically 34, 40, 42, 43, 46, 48,
51 while the same seed with a trainable trunk falls to 26. The frozen trunk costs 6 mIoU at
epoch 0 (60.8 vs 66.8), is ahead by epoch 5 (79.5 vs 77.8) and finishes 20 epochs slightly
ahead. The frozen sup-only run stays flat near 31: labeled data alone cannot teach sofa even
when it cannot damage the trunk. One seed; the 60-epoch ceiling of a frozen trunk is
unmeasured.

### Pretrained k-NN as a labeler

| Labeler at split 92 | mIoU on val | Training steps spent |
|---|---|---|
| pretrained DINOv2 k-NN, 92-image bank | 55.0 | 0 |
| model (any arm) after epoch 0 | 66.8 to 67.6 | 655 |
| model after epoch 1 | 70 to 73 | 1310 |

The frozen k-NN is not better than the trained model; it is better than what the pipeline
has during epoch 0, when the mask ratio climbs from 0 to 0.56 on a teacher that started as
a random head a few hundred steps earlier. That window is also where the sofa/chair basin and
the cat collapse are decided. Its ceiling arrives within one epoch, its patches are blobby
(bicycle 0.5), and its per-class unevenness at the 92 bank (cat 45.9, cow 35.9, dog 50.6,
sheep 51.2) could seed the same confirmation-bias spiral from a different initial error, so
any use has to be confidence-gated per patch and per class, and gone by roughly epoch 1. The
decisive offline measurement, accuracy versus retained fraction of k-NN-92 labels against the
epoch-0/1/19 teacher on the unlabeled set (which carries ground truth as `mask_u_gt`), had
not been run as of 2026-09-03; the idea was carried forward on 2026-09-04 as the
`--abstention` referee gate.

### Caveats

One seed pair (12000), one layer (11) where the DPT reads four, $k = 20$, purity 0.75,
patch-level boundary penalty on every absolute IoU.
