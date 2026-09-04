"""Tests for util/referee.py (the k-NN referee and the abstention loss) and for the
--abstention path through unimatch_v2_1gpu.py (load_micro_batch, unsup_forward_backward,
train_micro_batch, validation and env override).

The referee needs a DINOv2 trunk: a randomly initialised dinov2_small is used where only shapes
and invariances matter, the pretrained checkpoint and data/PASCAL where they exist (those tests
skip otherwise). Anything that moves tensors to CUDA is GPU-marked like the other test files.
"""
import logging
import math
import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from model.backbone.dinov2 import DINOv2
from unimatch_v2_1gpu import (MicroBatch, MicroBatchLosses, apply_env_overrides, check_unlock_after,
                              load_micro_batch, parser, train_micro_batch, unsup_forward_backward)
from util.referee import (K, PATCH, PURITY, KNNReferee, abstention_loss_px, build_bank, knn_scores,
                          load_pretrained_backbone, patch_features, patch_labels, plain_dataset)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
CKPT = './pretrained/dinov2_small.pth'
needs_ckpt = pytest.mark.skipif(not os.path.exists(CKPT), reason='pretrained/dinov2_small.pth not present')
needs_data = pytest.mark.skipif(not os.path.isdir('data/PASCAL/SegmentationClass'), reason='data/PASCAL not present')
REQUIRED = ['--config', 'c.yaml', '--labeled-id-path', 'l.txt', '--unlabeled-id-path', 'u.txt', '--save-path', 's']

NCLASS = 5


# ============================================================================
# abstention_loss_px
# ============================================================================

def logits_and_labels(seed, b=2, c=NCLASS, h=3, w=4, force_disagree=True):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, c, h, w, generator=g, requires_grad=True)
    ya = torch.randint(0, c, (b, h, w), generator=g)
    yb = torch.randint(0, c, (b, h, w), generator=g)
    if force_disagree:
        yb = torch.where(yb == ya, (ya + 1) % c, yb)
    return z, ya, yb


def test_abstention_value_is_minus_log_candidate_mass():
    z, ya, yb = logits_and_labels(0)
    p = z.softmax(1)
    pa, pb = p.gather(1, ya[:, None]).squeeze(1), p.gather(1, yb[:, None]).squeeze(1)
    torch.testing.assert_close(abstention_loss_px(z, ya, yb), -torch.log(pa + pb))


def test_abstention_gradient_is_p_off_the_candidates_and_zero_on_them():
    z, ya, yb = logits_and_labels(1)
    abstention_loss_px(z, ya, yb).sum().backward()
    p = z.detach().softmax(1)
    sel = torch.zeros_like(p, dtype=torch.bool).scatter_(1, ya[:, None], True).scatter_(1, yb[:, None], True)
    torch.testing.assert_close(z.grad[~sel], p[~sel])
    assert torch.equal(z.grad[sel], torch.zeros_like(z.grad[sel]))  # exactly zero, not merely small


def test_abstention_keeps_exactly_the_shared_part_of_the_two_ce_gradients():
    # CE(y_a) and CE(y_b) have gradient p - e_y: identical off {y_a, y_b}. The abstention gradient
    # equals both of them there and is zero where they disagree.
    z, ya, yb = logits_and_labels(2)
    g_abs = torch.autograd.grad(abstention_loss_px(z, ya, yb).sum(), z)[0]
    g_a = torch.autograd.grad(F.cross_entropy(z, ya, reduction='sum'), z)[0]
    g_b = torch.autograd.grad(F.cross_entropy(z, yb, reduction='sum'), z)[0]
    sel = torch.zeros_like(z, dtype=torch.bool).scatter_(1, ya[:, None], True).scatter_(1, yb[:, None], True)
    torch.testing.assert_close(g_abs[~sel], g_a[~sel])
    torch.testing.assert_close(g_abs[~sel], g_b[~sel])
    assert (g_a[sel] != g_b[sel]).all()  # the two CE gradients really do disagree there


def test_abstention_gradient_leaves_the_candidate_log_ratio_untouched():
    z, ya, yb = logits_and_labels(3)
    z.grad = None
    abstention_loss_px(z, ya, yb).sum().backward()
    ga = z.grad.gather(1, ya[:, None]).squeeze(1)
    gb = z.grad.gather(1, yb[:, None]).squeeze(1)
    assert torch.equal(ga - gb, torch.zeros_like(ga))
    # and the total push-down equals the probability mass outside the candidate set
    p = z.detach().softmax(1)
    pS = p.gather(1, ya[:, None]).squeeze(1) + p.gather(1, yb[:, None]).squeeze(1)
    torch.testing.assert_close(z.grad.sum(1), 1 - pS)


def test_abstention_is_symmetric_in_the_two_labelers():
    z, ya, yb = logits_and_labels(4)
    torch.testing.assert_close(abstention_loss_px(z, ya, yb), abstention_loss_px(z, yb, ya))
    g1 = torch.autograd.grad(abstention_loss_px(z, ya, yb).sum(), z)[0]
    g2 = torch.autograd.grad(abstention_loss_px(z, yb, ya).sum(), z)[0]
    torch.testing.assert_close(g1, g2)


def test_abstention_with_equal_labels_is_not_cross_entropy():
    # documents why the caller must route agreed pixels to CE: with |S| = 1 the label's own
    # logit gets no gradient at all, whereas CE pushes it up by 1 - p_y
    z, ya, _ = logits_and_labels(5)
    g_abs = torch.autograd.grad(abstention_loss_px(z, ya, ya).sum(), z)[0]
    g_ce = torch.autograd.grad(F.cross_entropy(z, ya, reduction='sum'), z)[0]
    on = g_abs.gather(1, ya[:, None])
    assert torch.equal(on, torch.zeros_like(on))
    assert not torch.allclose(g_abs, g_ce)
    torch.testing.assert_close(abstention_loss_px(z, ya, ya), F.cross_entropy(z, ya, reduction='none'))  # values do agree


def test_abstention_tolerates_ignore_labels():
    z, ya, yb = logits_and_labels(6)
    ya = ya.clone(); ya[0, 0, 0] = 255; yb = yb.clone(); yb[1, 2, 3] = 255
    out = abstention_loss_px(z, ya, yb)
    assert out.shape == ya.shape and torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_abstention_is_per_pixel():
    z, ya, yb = logits_and_labels(7)
    full = abstention_loss_px(z, ya, yb)
    one = abstention_loss_px(z[1:, :, 2:3, 1:2], ya[1:, 2:3, 1:2], yb[1:, 2:3, 1:2])
    torch.testing.assert_close(full[1, 2, 1], one[0, 0, 0])


@needs_cuda
def test_abstention_under_bf16_autocast_matches_fp32():
    z, ya, yb = logits_and_labels(8)
    z, ya, yb = z.detach().cuda().requires_grad_(True), ya.cuda(), yb.cuda()
    ref = abstention_loss_px(z, ya, yb)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = abstention_loss_px(z.to(torch.bfloat16), ya, yb)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# ============================================================================
# patch_labels
# ============================================================================

def block_mask(blocks, h=2, w=2):
    """(1, h*14, w*14) mask from a list of per-block fill functions."""
    m = torch.zeros(1, h * PATCH, w * PATCH, dtype=torch.long)
    for i, fill in enumerate(blocks):
        r, c = divmod(i, w)
        fill(m[0, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH])
    return m


def test_patch_labels_pure_mixed_and_ignore_blocks():
    def pure(v): return lambda blk: blk.fill_(v)
    def half(a, b): return lambda blk: (blk.fill_(a), blk[:PATCH // 2].fill_(b))
    def mostly_ignore(v): return lambda blk: (blk.fill_(255), blk[:2].fill_(v))
    m = block_mask([pure(3), half(1, 2), mostly_ignore(4), pure(0)])
    lab = patch_labels(m, tuple(m.shape[-2:]), NCLASS, PURITY)
    assert lab.tolist() == [3, 255, 255, 0]


def test_patch_labels_purity_threshold_is_inclusive():
    def frac(v, other, k):
        # fill the block with v, then its first k pixels (row-major) with `other`; the block is a
        # non-contiguous view of the mask, so write through row slices rather than a flat view
        def fill(blk):
            blk.fill_(v)
            rows, rem = divmod(k, PATCH)
            blk[:rows].fill_(other)
            blk[rows, :rem].fill_(other)
        return fill
    n = PATCH * PATCH
    k_exact = n - int(round(PURITY * n))           # leaves exactly PURITY of the block as class 1
    m = block_mask([frac(1, 2, k_exact), frac(1, 2, k_exact + 1), frac(1, 255, k_exact), frac(1, 255, k_exact + 1)])
    lab = patch_labels(m, tuple(m.shape[-2:]), NCLASS, PURITY)
    assert lab.tolist() == [1, 255, 1, 255]


def test_patch_labels_rescales_to_the_requested_grid():
    m = block_mask([lambda b: b.fill_(2)] * 4, h=2, w=2)              # 28 x 28
    lab = patch_labels(m, (PATCH * 4, PATCH * 4), NCLASS, PURITY)      # resized to 56 x 56 -> 4x4 patches
    assert lab.shape == (16,) and (lab == 2).all()


# ============================================================================
# knn_scores
# ============================================================================

@needs_cuda
def test_knn_scores_identical_query_wins_and_shapes():
    bank = F.normalize(torch.randn(50, 16, device='cuda'), dim=1).half()
    y = torch.randint(0, NCLASS, (50,), device='cuda'); y[7] = 3
    scores = knn_scores(bank[7:8], bank, y, k=1, nclass=NCLASS)
    assert scores.shape == (1, NCLASS) and scores.argmax().item() == 3 and (scores >= 0).all()


@needs_cuda
def test_knn_scores_chunking_is_exact():
    torch.manual_seed(0)
    bank = F.normalize(torch.randn(1000, 16, device='cuda'), dim=1).half()
    y = torch.randint(0, NCLASS, (1000,), device='cuda')
    q = F.normalize(torch.randn(30, 16, device='cuda'), dim=1).half()
    torch.testing.assert_close(knn_scores(q, bank, y, K, NCLASS, chunk=64), knn_scores(q, bank, y, K, NCLASS, chunk=10 ** 6))


@needs_cuda
def test_knn_scores_k_larger_than_bank_and_similarity_weighting():
    a = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device='cuda'), dim=1).half()
    y = torch.tensor([0, 0, 1], device='cuda')
    q = F.normalize(torch.tensor([[math.cos(0.6), math.sin(0.6)]], device='cuda'), dim=1).half()
    # cos to the two class-0 patches: cos(0.6)=0.825 each -> 1.65; to the class-1 patch: sin(0.6)=0.565
    s = knn_scores(q, a, y, k=10, nclass=2)
    assert s.argmax().item() == 0
    torch.testing.assert_close(s[0], torch.tensor([2 * math.cos(0.6), math.sin(0.6)], device='cuda'), rtol=1e-2, atol=1e-2)


@needs_cuda
def test_knn_scores_negative_similarities_do_not_vote():
    a = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], device='cuda').half()
    y = torch.tensor([0, 1], device='cuda')
    s = knn_scores(torch.tensor([[1.0, 0.0]], device='cuda').half(), a, y, k=2, nclass=2)
    assert s[0, 1].item() == 0.0 and s[0, 0].item() > 0


# ============================================================================
# patch_features / KNNReferee on a random trunk
# ============================================================================

@pytest.fixture(scope='module')
def random_trunk():
    torch.manual_seed(0)
    bb = DINOv2(model_name='small').cuda().eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    return bb


@needs_cuda
def test_patch_features_shape_norm_and_grid(random_trunk):
    x = torch.randn(2, 3, 56, 84, device='cuda')
    f, grid, size = patch_features(random_trunk, x, len(random_trunk.blocks) - 1)
    assert f.shape == (2, 4 * 6, random_trunk.embed_dim) and f.dtype == torch.half
    assert grid == (4, 6) and size == (56, 84)
    torch.testing.assert_close(f.float().norm(dim=-1), torch.ones(2, 24, device='cuda'), rtol=2e-3, atol=2e-3)


@needs_cuda
def test_patch_features_rounds_odd_sizes_to_the_patch_grid(random_trunk):
    f, grid, size = patch_features(random_trunk, torch.randn(1, 3, 60, 75, device='cuda'), 11)
    assert size == (56, 70) and grid == (4, 5) and f.shape[1] == 20   # 60/14=4.29->4, 75/14=5.36->5


@needs_cuda
def test_referee_predict_shape_dtype_range_and_determinism(random_trunk):
    torch.manual_seed(1)
    bank = F.normalize(torch.randn(300, random_trunk.embed_dim, device='cuda'), dim=1).half()
    y = torch.randint(0, NCLASS, (300,), device='cuda')
    ref = KNNReferee(random_trunk, bank, y, NCLASS)
    assert ref.layer == 11 and ref.k == K
    x = torch.randn(2, 3, 70, 84, device='cuda')
    out = ref.predict(x)
    assert out.shape == (2, 70, 84) and out.dtype == torch.long
    assert out.min() >= 0 and out.max() < NCLASS
    assert torch.equal(out, ref.predict(x))


@needs_cuda
def test_referee_predict_restores_non_multiple_of_14_sizes(random_trunk):
    bank = F.normalize(torch.randn(100, random_trunk.embed_dim, device='cuda'), dim=1).half()
    ref = KNNReferee(random_trunk, bank, torch.zeros(100, dtype=torch.long, device='cuda'), NCLASS)
    assert ref.predict(torch.randn(1, 3, 61, 79, device='cuda')).shape == (1, 61, 79)


@needs_cuda
def test_referee_reproduces_labels_of_its_own_bank_image(random_trunk):
    # bank = this image's own patches labeled by a coarse map; predicting the image must return
    # that map (every query's nearest neighbour is itself, similarity 1)
    x = torch.randn(1, 3, 84, 84, device='cuda')            # 6x6 patches
    f, (h, w), _ = patch_features(random_trunk, x, 11)
    lab_grid = torch.arange(h * w, device='cuda') % NCLASS
    ref = KNNReferee(random_trunk, f[0], lab_grid, NCLASS, k=1)
    out = ref.predict(x)[0]
    per_block = out.view(h, PATCH, w, PATCH).permute(0, 2, 1, 3).reshape(h, w, -1)
    majority = per_block.mode(dim=-1).values                 # (h, w)
    # bilinear upsampling with align_corners=True stretches the outer ring of patches over the
    # image border and blends neighbours across block edges, so the exact check is on the
    # interior blocks and the outer ring only has to be mostly right
    assert torch.equal(majority[1:-1, 1:-1], lab_grid.view(h, w)[1:-1, 1:-1])
    assert (majority == lab_grid.view(h, w)).float().mean() > 0.75


# ============================================================================
# plain_dataset / build_bank / from_pretrained on real assets
# ============================================================================

def test_plain_dataset_replaces_the_val_id_list(tmp_path):
    ids = ['JPEGImages/a.jpg SegmentationClass/a.png', 'JPEGImages/b.jpg SegmentationClass/b.png']
    ds = plain_dataset('pascal', 'nowhere', ids)
    assert ds.mode == 'val' and ds.ids == ids and len(ds) == 2
    f = tmp_path / 'ids.txt'; f.write_text('\n'.join(ids) + '\n\n')
    assert plain_dataset('pascal', 'nowhere', str(f)).ids == ids


@needs_cuda
@needs_ckpt
def test_load_pretrained_backbone_is_frozen_eval_and_matches_checkpoint():
    bb = load_pretrained_backbone('dinov2_small', CKPT)
    assert not bb.training and all(not p.requires_grad for p in bb.parameters()) and len(bb.blocks) == 12
    sd = torch.load(CKPT, map_location='cpu')
    torch.testing.assert_close(bb.blocks[0].attn.qkv.weight.cpu(), sd['blocks.0.attn.qkv.weight'])


@needs_cuda
@needs_ckpt
@needs_data
def test_build_bank_and_from_pretrained_on_two_labeled_images():
    ids = [l for l in open('splits/pascal/92/labeled.txt').read().splitlines() if l.strip()][:2]
    bb = load_pretrained_backbone('dinov2_small', CKPT)
    f, y = build_bank(bb, ids, 'data/PASCAL', 'pascal', 11, 21, purity=0.5)
    f2, y2 = build_bank(bb, ids, 'data/PASCAL', 'pascal', 11, 21, purity=1.0)
    assert f.shape[0] == y.shape[0] > f2.shape[0] > 0 and f.dtype == torch.half
    assert y.min() >= 0 and y.max() < 21 and y.dtype == torch.long
    torch.testing.assert_close(f.float().norm(dim=1), torch.ones_like(y, dtype=torch.float), rtol=2e-3, atol=2e-3)
    ref = KNNReferee.from_pretrained('dinov2_small', CKPT, ids, 'data/PASCAL', 'pascal', 21, logger=logging.getLogger('t'))
    out = ref.predict(torch.randn(1, 3, 56, 56, device='cuda'))
    assert out.shape == (1, 56, 56) and out.max() < 21


# ============================================================================
# the --abstention path through the trainer
# ============================================================================

B, H, W = 2, 16, 24
pytest_cuda = needs_cuda


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, NCLASS, 1))

    def forward(self, x, comp_drop=False):
        return self.body(x)


class LogitsStudent(nn.Module):
    """Returns a learnable logits tensor for the 2B strong views, so per-pixel logit gradients
    can be read directly."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.z = nn.Parameter(torch.randn(2 * B, NCLASS, H, W))

    def forward(self, x, comp_drop=False):
        return self.z


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, NCLASS, 1)

    def forward(self, x):
        return self.proj(x)


class FakeReferee:
    """Stands in for KNNReferee: returns a fixed label map and records what it was asked about."""
    def __init__(self, labels):
        self.labels, self.calls = labels, []

    def predict(self, x):
        self.calls.append(x)
        return self.labels


def synth_mb(seed, ref=None, conf=None, box=None, gt=None, ignore=None):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    mask_u_w = torch.randint(0, NCLASS, (B, H, W), generator=g)
    conf_u_w = torch.rand(B, H, W, generator=g) if conf is None else conf
    ign = torch.zeros(B, H, W, dtype=torch.long) if ignore is None else ignore
    box1 = torch.zeros(B, H, W) if box is None else box
    gt = torch.randint(0, NCLASS, (B, H, W), generator=g) if gt is None else gt
    t = lambda x: x.cuda()
    return MicroBatch(t(r(B, 3, H, W)), t(torch.randint(0, NCLASS, (B, H, W), generator=g)), t(r(B, 3, H, W)), t(r(B, 3, H, W)),
                      t(mask_u_w), t(conf_u_w), t(ign), t(box1), t(box1.clone()), t(gt), None if ref is None else t(ref(mask_u_w)))


def grads_of(model):
    return [p.grad.clone() for p in model.parameters()]


def test_microbatch_and_losses_defaults():
    mb = MicroBatch(torch.zeros(1), torch.zeros(1))
    assert mb.ref_u_w is None and mb.mask_u_gt is None
    ten = MicroBatch(*([torch.zeros(1)] * 10))
    assert ten.ref_u_w is None and ten.mask_u_gt is not None
    # the losses record is unchanged: four fields, so the pre-refactor tuple comparisons still hold
    r = MicroBatchLosses(1.0, 2.0, 3.0, 0.5)
    assert tuple(r) == (1.0, 2.0, 3.0, 0.5) and len(r._fields) == 4


@pytest_cuda
def test_full_agreement_is_identical_to_the_ungated_loss():
    crit = nn.CrossEntropyLoss(reduction='none').cuda()
    outs = []
    for gated in (False, True):
        torch.manual_seed(0); model = ToyStudent().cuda().train()
        mb = synth_mb(0, ref=(lambda m: m.clone()) if gated else None)
        stats = {}
        loss, ratio = unsup_forward_backward(model, mb, crit, 0.5, 4, False, False, stats=stats)
        outs.append((loss.item(), ratio, grads_of(model), stats))
    assert outs[0][0] == pytest.approx(outs[1][0]) and outs[0][1] == outs[1][1]
    for g0, g1 in zip(outs[0][2], outs[1][2]):
        torch.testing.assert_close(g0, g1)
    assert outs[0][3] == {} and outs[1][3]['abstain_ratio'] == 0.0


@pytest_cuda
def test_stats_are_written_only_when_a_referee_ran():
    crit = nn.CrossEntropyLoss(reduction='none').cuda(); model = ToyStudent().cuda()
    stats = {}
    unsup_forward_backward(model, synth_mb(1), crit, 0.5, 4, False, False, stats=stats)
    assert stats == {}
    stats = {}
    unsup_forward_backward(model, synth_mb(1, ref=lambda m: (m + 1) % NCLASS), crit, 0.5, 4, False, False, stats=stats)
    assert set(stats) == {'abstain_ratio', 'teacher_acc', 'referee_acc', 'agree_acc'}
    # sup_only: zero stub, nothing written
    stats = {}
    loss, ratio = unsup_forward_backward(model, MicroBatch(torch.zeros(1, device='cuda'), torch.zeros(1, device='cuda')),
                                         crit, 0.5, 4, True, False, stats=stats)
    assert loss.item() == 0.0 and ratio == 0.0 and stats == {}


@pytest_cuda
def test_abstain_ratio_extremes_and_valid_set():
    crit = nn.CrossEntropyLoss(reduction='none').cuda(); model = ToyStudent().cuda()
    conf = torch.ones(B, H, W)
    stats = {}
    unsup_forward_backward(model, synth_mb(2, ref=lambda m: (m + 1) % NCLASS, conf=conf), crit, 0.5, 4, False, False, stats=stats)
    assert stats['abstain_ratio'] == pytest.approx(1.0)
    stats = {}
    unsup_forward_backward(model, synth_mb(2, ref=lambda m: m.clone(), conf=conf), crit, 0.5, 4, False, False, stats=stats)
    assert stats['abstain_ratio'] == 0.0
    # disagreement only where the teacher is UNconfident -> those pixels are not counted -> ratio 0
    conf = torch.zeros(B, H, W); conf[:, :H // 2] = 1.0
    def ref(m):
        r = m.clone(); r[:, H // 2:] = (m[:, H // 2:] + 1) % NCLASS; return r
    stats = {}
    unsup_forward_backward(model, synth_mb(2, ref=ref, conf=conf), crit, 0.5, 4, False, False, stats=stats)
    assert stats['abstain_ratio'] == 0.0


@pytest_cuda
def test_accuracy_diagnostics_against_the_unlabeled_gt():
    crit = nn.CrossEntropyLoss(reduction='none').cuda(); model = ToyStudent().cuda()
    conf = torch.ones(B, H, W)
    g = torch.Generator().manual_seed(3)
    mask_u_w = torch.randint(0, NCLASS, (B, H, W), generator=g)
    mb0 = synth_mb(3, conf=conf)
    # gt == teacher everywhere, referee == teacher on the left half only
    def ref(m):
        r = m.clone(); r[:, :, W // 2:] = (m[:, :, W // 2:] + 1) % NCLASS; return r
    mb = synth_mb(3, ref=ref, conf=conf, gt=mb0.mask_u_w.cpu())
    stats = {}
    unsup_forward_backward(model, mb, crit, 0.5, 4, False, False, stats=stats)
    assert stats['teacher_acc'] == pytest.approx(1.0)
    assert stats['referee_acc'] == pytest.approx(0.5)
    assert stats['agree_acc'] == pytest.approx(1.0)
    assert stats['abstain_ratio'] == pytest.approx(0.5)
    # gt ignore (255) pixels are excluded from the accuracies
    gt = mb0.mask_u_w.cpu().clone(); gt[:, :H // 2] = 255
    mb = synth_mb(3, ref=lambda m: m.clone(), conf=conf, gt=gt)
    stats = {}
    unsup_forward_backward(model, mb, crit, 0.5, 4, False, False, stats=stats)
    assert stats['teacher_acc'] == pytest.approx(1.0) and stats['referee_acc'] == pytest.approx(1.0)


@pytest_cuda
def test_disputed_pixels_get_no_gradient_on_either_candidate():
    crit = nn.CrossEntropyLoss(reduction='none').cuda()
    model = LogitsStudent().cuda()
    conf = torch.ones(B, H, W)
    mb = synth_mb(4, ref=lambda m: (m + 2) % NCLASS, conf=conf)
    unsup_forward_backward(model, mb, crit, 0.5, 1, False, False)
    grad = model.z.grad  # (2B, C, H, W): views s1 then s2, no CutMix (boxes zero)
    for view in range(2):
        gv = grad[view * B:(view + 1) * B]
        on_t = gv.gather(1, mb.mask_u_w[:, None]); on_r = gv.gather(1, mb.ref_u_w[:, None])
        assert torch.equal(on_t, torch.zeros_like(on_t)) and torch.equal(on_r, torch.zeros_like(on_r))
        # every other class was pushed down (positive gradient) at every counted pixel
        sel = torch.zeros_like(gv, dtype=torch.bool).scatter_(1, mb.mask_u_w[:, None], True).scatter_(1, mb.ref_u_w[:, None], True)
        assert (gv[~sel] > 0).all()


@pytest_cuda
def test_agreed_pixels_get_the_ce_gradient():
    crit = nn.CrossEntropyLoss(reduction='none').cuda()
    model = LogitsStudent().cuda()
    conf = torch.ones(B, H, W)
    mb = synth_mb(5, ref=lambda m: m.clone(), conf=conf)
    unsup_forward_backward(model, mb, crit, 0.5, 1, False, False)
    z = model.z.detach()
    n_nonpad = float(H * W * B)
    # loss = mean over views of sum_px CE / n_nonpad, scaled by 1/(2*accum) with accum=1 -> 1/2
    expected = (z.softmax(1) - F.one_hot(torch.cat([mb.mask_u_w, mb.mask_u_w]), NCLASS).permute(0, 3, 1, 2).float()) / n_nonpad / 2.0 / 2.0
    torch.testing.assert_close(model.z.grad, expected, rtol=1e-4, atol=1e-6)


@pytest_cuda
def test_referee_map_is_cutmixed_with_the_same_box():
    # a box covering everything swaps every image with its mirror in the batch; the gated loss
    # must equal the loss with pre-swapped maps and no box
    crit = nn.CrossEntropyLoss(reduction='none').cuda()
    conf = torch.ones(B, H, W)
    ref = lambda m: (m + 1) % NCLASS
    full = torch.ones(B, H, W)
    torch.manual_seed(0); m1 = ToyStudent().cuda()
    mb_box = synth_mb(6, ref=ref, conf=conf, box=full)
    l_box, _ = unsup_forward_backward(m1, mb_box, crit, 0.5, 4, False, False)
    torch.manual_seed(0); m2 = ToyStudent().cuda()
    base = synth_mb(6, ref=ref, conf=conf)
    swapped = base._replace(mask_u_w=base.mask_u_w.flip(0), conf_u_w=base.conf_u_w.flip(0), ignore_mask=base.ignore_mask.flip(0),
                            ref_u_w=base.ref_u_w.flip(0), img_u_s1=mb_box.img_u_s1, img_u_s2=mb_box.img_u_s2)
    l_plain, _ = unsup_forward_backward(m2, swapped, crit, 0.5, 4, False, False)
    assert l_box.item() == pytest.approx(l_plain.item(), rel=1e-5)
    for g1, g2 in zip(grads_of(m1), grads_of(m2)):
        torch.testing.assert_close(g1, g2)


@pytest_cuda
def test_padding_and_ignore_pixels_stay_out_of_the_gated_loss():
    crit = nn.CrossEntropyLoss(reduction='none').cuda()
    conf = torch.ones(B, H, W)
    ign = torch.zeros(B, H, W, dtype=torch.long); ign[:, :, W // 2:] = 255
    torch.manual_seed(0); m1 = ToyStudent().cuda()
    mb = synth_mb(7, ref=lambda m: (m + 1) % NCLASS, conf=conf, ignore=ign)
    l1, _ = unsup_forward_backward(m1, mb, crit, 0.5, 4, False, False)
    # corrupting the labels in the padded half changes nothing: a different class for the
    # teacher (its argmax is always in range, so criterion_u has no ignore index) and the
    # ignore value 255 for the referee (which abstention_loss_px clamps)
    torch.manual_seed(0); m2 = ToyStudent().cuda()
    bad = mb._replace(mask_u_w=mb.mask_u_w.clone(), ref_u_w=mb.ref_u_w.clone())
    bad.mask_u_w[:, :, W // 2:] = (mb.mask_u_w[:, :, W // 2:] + 3) % NCLASS; bad.ref_u_w[:, :, W // 2:] = 255
    l2, _ = unsup_forward_backward(m2, bad, crit, 0.5, 4, False, False)
    assert l1.item() == pytest.approx(l2.item())
    for g1, g2 in zip(grads_of(m1), grads_of(m2)):
        torch.testing.assert_close(g1, g2)


def unl_batch(seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    lab = lambda: torch.randint(0, NCLASS, (B, H, W), generator=g)
    return ((r(B, 3, H, W), lab()), (r(B, 3, H, W), r(B, 3, H, W), r(B, 3, H, W), torch.zeros(B, H, W, dtype=torch.long),
                                     torch.zeros(B, H, W), torch.zeros(B, H, W), lab()))


@pytest_cuda
def test_load_micro_batch_calls_the_referee_on_the_weak_view():
    teacher = ToyTeacher().cuda().eval()
    batch = unl_batch(8)
    labels = torch.randint(0, NCLASS, (B, H, W), device='cuda')
    ref = FakeReferee(labels)
    mb = load_micro_batch(iter([batch]), teacher, sup_only=False, bf16=False, referee=ref)
    assert torch.equal(mb.ref_u_w, labels) and len(ref.calls) == 1
    torch.testing.assert_close(ref.calls[0], batch[1][0].cuda())      # the unmixed weak view
    plain = load_micro_batch(iter([batch]), teacher, sup_only=False, bf16=False)
    assert plain.ref_u_w is None
    assert load_micro_batch(iter([batch[0]]), teacher, sup_only=True, bf16=False, referee=ref).ref_u_w is None
    assert len(ref.calls) == 1  # sup-only never consults the referee


@pytest_cuda
def test_train_micro_batch_threads_the_referee_and_reports_its_stats():
    crit_l = nn.CrossEntropyLoss(ignore_index=255).cuda(); crit_u = nn.CrossEntropyLoss(reduction='none').cuda()
    outs = []
    for use_ref in (False, True):
        torch.manual_seed(0); model = ToyStudent().cuda().train(); teacher = ToyTeacher().cuda().eval()
        params = list(model.parameters())
        flat = torch.zeros(sum(p.numel() for p in params), device='cuda')
        views, off = [], 0
        for p in params:
            views.append(flat[off:off + p.numel()].view_as(p)); off += p.numel()
        batch = unl_batch(9)
        with torch.no_grad():
            agree = teacher(batch[1][0].cuda()).argmax(1)   # referee == teacher -> full agreement
        stats = {}
        r = train_micro_batch(iter([batch]), model, teacher, crit_l, crit_u, params, views, 4, 0.0, False, False,
                              referee=FakeReferee(agree) if use_ref else None, stats=stats)
        outs.append((r, stats, [p.grad.clone() for p in params]))
    (r0, s0, g0s), (r1, s1, g1s) = outs
    assert isinstance(r1, MicroBatchLosses) and len(r1) == 4
    assert r0.loss == pytest.approx(r1.loss) and r0.loss_s == pytest.approx(r1.loss_s)
    assert s0 == {}                                       # no referee: nothing written
    assert s1['abstain_ratio'] == 0.0 and not math.isnan(s1['teacher_acc'])
    assert s1['agree_acc'] == pytest.approx(s1['teacher_acc'])   # all pixels agreed
    for g0, g1 in zip(g0s, g1s):
        torch.testing.assert_close(g0, g1)


# ============================================================================
# argument plumbing
# ============================================================================

def test_abstention_flag_default_env_override_and_validation(monkeypatch):
    args = parser.parse_args(REQUIRED)
    assert args.abstention is False
    monkeypatch.setenv('ABSTENTION', 'true')
    apply_env_overrides(args, logging.getLogger('t'))
    assert args.abstention is True
    check_unlock_after(args, {'lock_backbone': False})   # no lock needed: abstention is independent of it
    check_unlock_after(args, {'lock_backbone': True})    # ...and works with a permanently locked backbone
    args.unlock_after = 500
    check_unlock_after(args, {'lock_backbone': False})   # and alongside a delayed unlock
    args.unlock_accumulate = True
    check_unlock_after(args, {'lock_backbone': False})   # accumulate is optional alongside it
    args.sup_only = True
    with pytest.raises(ValueError):                 # nothing to gate
        check_unlock_after(args, {'lock_backbone': False})
    args.sup_only = False; args.unlock_after = 500
    with pytest.raises(ValueError):                 # unlock-after with lock_backbone is still refused
        check_unlock_after(args, {'lock_backbone': True})
    monkeypatch.setenv('ABSTENTION', 'off')
    apply_env_overrides(args, logging.getLogger('t'))
    assert args.abstention is False
    monkeypatch.setenv('ABSTENTION', 'maybe')
    with pytest.raises(ValueError):
        apply_env_overrides(args, logging.getLogger('t'))
