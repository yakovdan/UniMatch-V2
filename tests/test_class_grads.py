"""Tests for accumulate_class_grads, the per-class supervised gradient accumulation
extracted out of the training loop of unimatch_v2_1gpu.py.

``reference_block`` is the inline code as it stood before the extraction, so the
equivalence test pins the refactor to the old behaviour. The function itself is
device-agnostic, so most tests run on the CPU, where the backward is deterministic
and results can be compared exactly. One test runs the real DPT head on the GPU
under bf16, the way the trainer uses it.
"""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from unimatch_v2_1gpu import accumulate_class_grads

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')

NCLASS = 5
B, CIN, H, W = 2, 6, 7, 9


def reference_block(pred_x, mask_x, proto, anchor_params, proto_acc, proto_cnt):
    """The training-loop block before the extraction, verbatim."""
    ce_px = F.cross_entropy(pred_x, mask_x, ignore_index=255, reduction='none')
    g_c = None
    for c in mask_x.unique().tolist():
        if c == 255 or c >= proto.shape[0]:
            continue
        g_c = torch.autograd.grad(ce_px[mask_x == c].mean(), anchor_params,
                                  retain_graph=True, allow_unused=True)
        row, off = proto_acc[c], 0
        for p, g in zip(anchor_params, g_c):
            n = p.numel()
            if g is not None:
                row[off:off + n].add_(g.reshape(-1))
            off += n
        proto_cnt[c] += 1
    del ce_px, g_c


def toy_head(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(CIN, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, NCLASS, 1))


def toy_batch(seed, classes=range(NCLASS), ignore_rows=1):
    gen = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, CIN, H, W, generator=gen)
    classes = torch.tensor(list(classes))
    mask = classes[torch.randint(0, len(classes), (B, H, W), generator=gen)]
    mask[:, :ignore_rows] = 255
    return feats, mask


def accumulators(params, n_cls=NCLASS):
    d = sum(p.numel() for p in params)
    return torch.zeros(n_cls, d), torch.zeros(n_cls)


def flat_grad(loss, params):
    """Gradient of `loss` w.r.t. `params`, flattened in list order, zeros for unused."""
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p, g in zip(params, gs)])


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_matches_reference(seed):
    feats, mask = toy_batch(seed)
    outs = []
    for impl in ('new', 'old'):
        head = toy_head(seed)
        params = list(head.parameters())
        acc, cnt = accumulators(params)
        pred = head(feats)
        if impl == 'new':
            accumulate_class_grads(pred, mask, params, acc, cnt)
        else:
            reference_block(pred, mask, torch.zeros_like(acc), params, acc, cnt)
        outs.append((acc, cnt))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


def test_pixel_counts_are_accumulated_without_touching_the_gradients():
    feats, mask = toy_batch(3, classes=[0, 1, 4])
    head = toy_head(3)
    params = list(head.parameters())
    acc, cnt = accumulators(params)
    acc_ref, cnt_ref = accumulators(params)
    px = torch.zeros(NCLASS, dtype=torch.long)
    pred = head(feats)
    accumulate_class_grads(pred, mask, params, acc, cnt, px)
    accumulate_class_grads(pred, mask, params, acc_ref, cnt_ref)  # no proto_px: unchanged path
    assert torch.equal(acc, acc_ref) and torch.equal(cnt, cnt_ref)
    for c in range(NCLASS):
        assert px[c] == (mask == c).sum()  # 0 for absent classes; 255 never counted
    # a second micro-batch adds on top
    feats2, mask2 = toy_batch(4, classes=[1, 2])
    accumulate_class_grads(head(feats2), mask2, params, acc, cnt, px)
    assert px[1] == (mask == 1).sum() + (mask2 == 1).sum()
    assert px[2] == (mask2 == 2).sum() and px[4] == (mask == 4).sum()


def test_each_row_is_the_gradient_of_the_mean_ce_over_that_class():
    feats, mask = toy_batch(0, classes=[0, 2, 3])  # classes 1 and 4 absent
    head = toy_head(0)
    params = list(head.parameters())
    acc, cnt = accumulators(params)
    pred = head(feats)
    accumulate_class_grads(pred, mask, params, acc, cnt)
    ce_px = F.cross_entropy(pred, mask, ignore_index=255, reduction='none')
    for c in range(NCLASS):
        if c in (0, 2, 3):
            assert cnt[c] == 1
            torch.testing.assert_close(acc[c], flat_grad(ce_px[mask == c].mean(), params))
        else:
            assert cnt[c] == 0 and not acc[c].any()


def test_ignore_pixels_and_ids_beyond_the_accumulator_are_skipped():
    # logits with more channels than the accumulator has rows: the extra id is a valid
    # CE target but has no prototype, so it must be skipped rather than indexed
    torch.manual_seed(0)
    head = nn.Sequential(nn.Conv2d(CIN, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, NCLASS + 3, 1))
    feats, mask = toy_batch(0, classes=[1, 2])
    mask[0, -1] = NCLASS + 2
    params = list(head.parameters())
    acc, cnt = accumulators(params)
    pred = head(feats)
    accumulate_class_grads(pred, mask, params, acc, cnt)  # no IndexError
    assert cnt.tolist() == [0, 1, 1, 0, 0]
    ce_px = F.cross_entropy(pred, mask, ignore_index=255, reduction='none')
    for c in (1, 2):
        # the 255 strip is in no class's mean: ce_px there is 0 and mask == c excludes it
        torch.testing.assert_close(acc[c], flat_grad(ce_px[mask == c].mean(), params))


def test_accumulates_across_micro_batches():
    head = toy_head(0)
    params = list(head.parameters())
    acc, cnt = accumulators(params)
    singles = []
    for seed in (1, 2):
        feats, mask = toy_batch(seed)
        acc_i, cnt_i = accumulators(params)
        accumulate_class_grads(head(feats), mask, params, acc_i, cnt_i)
        accumulate_class_grads(head(feats), mask, params, acc, cnt)
        singles.append((acc_i, cnt_i))
    torch.testing.assert_close(acc, singles[0][0] + singles[1][0])
    assert torch.equal(cnt, singles[0][1] + singles[1][1])


def test_the_graph_survives_for_the_full_loss_backward():
    feats, mask = toy_batch(0)
    head = toy_head(0)
    params = list(head.parameters())
    acc, cnt = accumulators(params)
    pred = head(feats)
    loss = F.cross_entropy(pred, mask, ignore_index=255)
    accumulate_class_grads(pred, mask, params, acc, cnt)
    loss.backward()  # would raise if a per-class pass had freed the graph
    head_ref = toy_head(0)
    pred_ref = head_ref(feats)
    F.cross_entropy(pred_ref, mask, ignore_index=255).backward()
    for p, q in zip(params, head_ref.parameters()):
        # autograd.grad never touches .grad, so the full-loss gradient is untouched
        torch.testing.assert_close(p.grad, q.grad)


def test_unused_anchor_params_contribute_zeros_and_keep_the_layout():
    feats, mask = toy_batch(0)
    head = toy_head(0)
    used = list(head.parameters())
    unused = nn.Parameter(torch.randn(3, 4))
    params = [used[0], unused, *used[1:]]  # unused in the middle shifts the later offsets
    acc, cnt = accumulators(params)
    pred = head(feats)
    accumulate_class_grads(pred, mask, params, acc, cnt)
    ce_px = F.cross_entropy(pred, mask, ignore_index=255, reduction='none')
    for c in mask.unique().tolist():
        if c == 255:
            continue
        torch.testing.assert_close(acc[c], flat_grad(ce_px[mask == c].mean(), params))
        off = used[0].numel()
        assert not acc[c, off:off + unused.numel()].any()


@needs_cuda
@pytest.mark.parametrize('bf16', [False, True])
def test_with_the_real_dpt_head(bf16):
    """The trainer's use: head-only anchors on a DPT forward, under the same autocast
    as the labeled forward, the full-loss backward still possible afterwards."""
    from model.semseg.dpt import DPT
    torch.manual_seed(0)
    model = DPT(encoder_size='small', features=64, out_channels=[48, 96, 192, 384], nclass=21).cuda().train()
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    crop = 14 * 8
    gen = torch.Generator().manual_seed(0)
    img = torch.randn(2, 3, crop, crop, generator=gen).cuda()
    mask = torch.randint(0, 21, (2, crop, crop), generator=gen).cuda()
    mask[:, :8] = 255
    d = sum(p.numel() for p in head_params)
    acc, cnt = torch.zeros(21, d, device='cuda'), torch.zeros(21, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        pred = model(img)
        loss = F.cross_entropy(pred, mask, ignore_index=255)
    accumulate_class_grads(pred, mask, head_params, acc, cnt)
    present = [c for c in mask.unique().tolist() if c != 255]
    assert cnt.sum() == len(present)
    assert all(cnt[c] == 1 and acc[c].abs().sum() > 0 for c in present)
    assert torch.isfinite(acc).all()
    loss.backward()
