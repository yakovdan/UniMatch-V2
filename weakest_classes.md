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

Source: per-run W&B pull `wandb_60ep.json` (job 1139d423 scratch, 2026-09-03 11:33), fields
`cls_at_best` and `worst`; numbers cross-checked against the train logs' evaluation lines.
