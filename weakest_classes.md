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
