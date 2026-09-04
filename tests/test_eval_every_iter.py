"""Tests for --eval-every-iter: the flag/env plumbing, and that supervised.evaluate on a val
prefix loader draws no RNG from the training streams (it is run between optimizer steps, where a
draw would shift the Complementary Dropout stream and break every seed pairing). Needs a GPU and
data/PASCAL for the evaluation test."""
import logging
import os

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
import supervised
from supervised import evaluate
from unimatch_v2_1gpu import apply_env_overrides, check_unlock_after, parser

REQUIRED = ['--config', 'c.yaml', '--labeled-id-path', 'l.txt', '--unlabeled-id-path', 'u.txt', '--save-path', 's']
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
needs_data = pytest.mark.skipif(not os.path.isdir('data/PASCAL/SegmentationClass'), reason='data/PASCAL not present')


def test_flags_env_and_validation():
    args = parser.parse_args(REQUIRED)
    assert args.eval_every_iter is False and args.eval_every_iter_images == 0
    args = parser.parse_args(REQUIRED + ['--eval-every-iter', '--eval-every-iter-images', '20'])
    check_unlock_after(args, {'lock_backbone': False})
    assert args.eval_every_iter and args.eval_every_iter_images == 20
    with pytest.raises(ValueError):
        check_unlock_after(parser.parse_args(REQUIRED + ['--eval-every-iter-images', '-1']), {'lock_backbone': False})
    args = parser.parse_args(REQUIRED)
    os.environ.update({'EVAL_EVERY_ITER': '1', 'EVAL_EVERY_ITER_IMAGES': '7'})
    try:
        apply_env_overrides(args, logging.getLogger('t'))
    finally:
        for k in ('EVAL_EVERY_ITER', 'EVAL_EVERY_ITER_IMAGES'):
            del os.environ[k]
    assert args.eval_every_iter is True and args.eval_every_iter_images == 7


@needs_cuda
@needs_data
def test_evaluate_on_a_val_prefix_draws_no_rng_and_needs_train_restored(monkeypatch):
    monkeypatch.setattr(supervised.dist, 'all_reduce', lambda *a, **k: None)  # single process, no group here
    torch.manual_seed(0)
    model = DPT(encoder_size='small', nclass=21, features=64, out_channels=[48, 96, 192, 384]).cuda().train()
    valset = SemiDataset('pascal', 'data/PASCAL', 'val')
    gen = torch.Generator()
    gen.manual_seed(4)
    loader = DataLoader(Subset(valset, range(3)), batch_size=1, num_workers=1, generator=gen)
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    cfg = {'dataset': 'pascal', 'nclass': 21, 'crop_size': 518}
    mIoU, per_class = evaluate(model, loader, 'original', cfg, multiplier=14)
    assert torch.equal(torch.get_rng_state(), cpu) and torch.equal(torch.cuda.get_rng_state(), cuda)
    assert len(per_class) == 21 and 0.0 <= float(mIoU) <= 100.0
    assert not model.training  # evaluate() switches to eval and does not switch back: the trainer restores train()
    model.train()
    assert model.training
