"""Tests for the micro-batch preparation extracted out of the training loop of
unimatch_v2_1gpu.py: ``load_micro_batch``, ``cutmix_`` and ``MicroBatch``.

``reference_block`` below is the inline code as it stood before the extraction,
so the equivalence tests pin the refactor to the old behaviour instead of to a
re-derivation of it. The function under test calls ``.cuda()``, so everything
except the ``cutmix_`` unit tests needs a GPU.
"""
import copy

import pytest
import torch
from torch import nn

from unimatch_v2_1gpu import MicroBatch, cutmix_, load_micro_batch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='load_micro_batch moves tensors to CUDA')

NCLASS = 21
B, H, W = 4, 32, 48  # micro-batch of 4, like cfg['batch_size']; small crops keep the tests fast


def reference_block(loader, model_ema, sup_only, bf16):
    """The training-loop block before the extraction, verbatim apart from packing
    its locals into a MicroBatch at the end."""
    if sup_only:
        img_x, mask_x = next(loader)
    else:
        (img_x, mask_x), (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2, mask_u_gt) = next(loader)
        img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
        ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
        mask_u_gt = mask_u_gt.cuda()

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            pred_u_w = model_ema(img_u_w).detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

        img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
        img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

    img_x, mask_x = img_x.cuda(), mask_x.cuda()
    if sup_only:
        return MicroBatch(img_x, mask_x)
    return MicroBatch(img_x, mask_x, img_u_s1, img_u_s2, mask_u_w, conf_u_w,
                      ignore_mask, cutmix_box1, cutmix_box2, mask_u_gt)


# ----------------------------------------------------------------------------
# synthetic batches shaped like SemiDataset's output (dataset/semi.py)
# ----------------------------------------------------------------------------

def random_boxes(gen, p_empty=0.25):
    """(B, H, W) float 0/1 masks, one rectangle per image, like obtain_cutmix_box in
    dataset/transform.py -- including the images that draw no box at all."""
    box = torch.zeros(B, H, W)
    for i in range(B):
        if torch.rand((), generator=gen) < p_empty:
            continue
        h = int(torch.randint(H // 4, H // 2 + 1, (), generator=gen))
        w = int(torch.randint(W // 4, W // 2 + 1, (), generator=gen))
        y = int(torch.randint(0, H - h + 1, (), generator=gen))
        x = int(torch.randint(0, W - w + 1, (), generator=gen))
        box[i, y:y + h, x:x + w] = 1
    return box


def labeled_batch(gen):
    img = torch.randn(B, 3, H, W, generator=gen)
    mask = torch.randint(0, NCLASS, (B, H, W), generator=gen)
    mask[:, :3] = 255  # an ignore strip, as PASCAL borders produce
    return img, mask


def unlabeled_batch(gen):
    img_u_w, img_u_s1, img_u_s2 = (torch.randn(B, 3, H, W, generator=gen) for _ in range(3))
    ignore_mask = torch.zeros(B, H, W, dtype=torch.long)
    ignore_mask[:, -4:] = 255  # padding rows, like dataset/semi.py's mask == 254 -> 255
    mask_u_gt = torch.randint(0, NCLASS, (B, H, W), generator=gen)
    return img_u_w, img_u_s1, img_u_s2, ignore_mask, random_boxes(gen), random_boxes(gen), mask_u_gt


def semi_loader(seed, n_batches=1):
    """Iterator over (labeled, unlabeled) pairs, like iter(zip(trainloader_l, trainloader_u))."""
    gen = torch.Generator().manual_seed(seed)
    return iter([(labeled_batch(gen), unlabeled_batch(gen)) for _ in range(n_batches)])


def sup_loader(seed, n_batches=1):
    gen = torch.Generator().manual_seed(seed)
    return iter([labeled_batch(gen) for _ in range(n_batches)])


def toy_teacher(seed):
    """A deterministic stand-in for model_ema: per-pixel logits from a 1x1 conv."""
    torch.manual_seed(seed)
    return nn.Conv2d(3, NCLASS, kernel_size=1).cuda().eval()


class SpyTeacher(nn.Module):
    """Records the autograd / autocast state and the input it was called with."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, NCLASS, kernel_size=1)
        self.calls = []

    def forward(self, x):
        self.calls.append({'grad_enabled': torch.is_grad_enabled(),
                           'autocast': torch.is_autocast_enabled('cuda'),
                           'input': x.detach().clone()})
        return self.proj(x)


def assert_same_micro_batch(a, b):
    assert a._fields == b._fields
    for name, x, y in zip(a._fields, a, b):
        if x is None or y is None:
            assert x is None and y is None, name
            continue
        assert x.device == y.device and x.dtype == y.dtype and x.shape == y.shape, name
        assert torch.equal(x, y), name


# ----------------------------------------------------------------------------
# cutmix_ (device-agnostic)
# ----------------------------------------------------------------------------

def test_cutmix_matches_the_old_one_liner():
    gen = torch.Generator().manual_seed(0)
    img, box = torch.randn(B, 3, H, W, generator=gen), random_boxes(gen, p_empty=0.0)
    new = img.clone()
    cutmix_(new, box)
    old = img.clone()
    old[box.unsqueeze(1).expand(old.shape) == 1] = old.flip(0)[box.unsqueeze(1).expand(old.shape) == 1]
    assert torch.equal(new, old)


def test_cutmix_takes_the_partner_inside_the_box_and_nothing_outside():
    gen = torch.Generator().manual_seed(1)
    img, box = torch.randn(B, 3, H, W, generator=gen), random_boxes(gen, p_empty=0.0)
    out = img.clone()
    cutmix_(out, box)
    inside = box.bool().unsqueeze(1).expand_as(img)
    for i in range(B):
        partner = B - 1 - i  # flip(0) pairs image i with image B-1-i
        assert torch.equal(out[i][inside[i]], img[partner][inside[i]])
        assert torch.equal(out[i][~inside[i]], img[i][~inside[i]])


def test_cutmix_is_in_place_and_handles_empty_and_full_boxes():
    img = torch.randn(B, 3, H, W)
    out = img.clone()
    cutmix_(out, torch.zeros(B, H, W))
    assert torch.equal(out, img)
    result = cutmix_(out, torch.ones(B, H, W))
    assert result is None  # in-place, nothing returned
    assert torch.equal(out, img.flip(0))


# ----------------------------------------------------------------------------
# load_micro_batch vs the pre-refactor block
# ----------------------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_sup_only_matches_reference(seed):
    got = load_micro_batch(sup_loader(seed), None, sup_only=True, bf16=False)
    want = reference_block(sup_loader(seed), None, sup_only=True, bf16=False)
    assert_same_micro_batch(got, want)
    assert got.img_x.is_cuda and got.mask_x.is_cuda
    assert all(getattr(got, f) is None for f in MicroBatch._fields[2:])


@needs_cuda
@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_semi_matches_reference(seed, bf16):
    # the strong views are CutMixed in place, so each side gets its own copy of the batch
    batch = next(semi_loader(seed))
    got = load_micro_batch(iter([copy.deepcopy(batch)]), toy_teacher(seed), sup_only=False, bf16=bf16)
    want = reference_block(iter([copy.deepcopy(batch)]), toy_teacher(seed), sup_only=False, bf16=bf16)
    assert_same_micro_batch(got, want)
    assert all(getattr(got, f).is_cuda for f in MicroBatch._fields)


# ----------------------------------------------------------------------------
# the properties the loop relies on
# ----------------------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize('bf16', [False, True])
def test_teacher_runs_once_without_grad_under_the_requested_autocast(bf16):
    teacher = SpyTeacher().cuda().eval()
    batch = next(semi_loader(0))
    mb = load_micro_batch(iter([copy.deepcopy(batch)]), teacher, sup_only=False, bf16=bf16)
    assert len(teacher.calls) == 1
    call = teacher.calls[0]
    assert call['grad_enabled'] is False
    assert call['autocast'] is bf16
    # the teacher sees the clean weak view, never a CutMixed one
    assert torch.equal(call['input'], batch[1][0].cuda())
    assert not mb.mask_u_w.requires_grad and not mb.conf_u_w.requires_grad


@needs_cuda
def test_pseudo_labels_are_the_teacher_argmax_and_confidence():
    teacher = toy_teacher(0)
    batch = next(semi_loader(0))
    mb = load_micro_batch(iter([copy.deepcopy(batch)]), teacher, sup_only=False, bf16=False)
    with torch.no_grad():
        logits = teacher(batch[1][0].cuda())
    assert mb.mask_u_w.dtype == torch.long and mb.mask_u_w.shape == (B, H, W)
    assert torch.equal(mb.mask_u_w, logits.argmax(1))
    assert torch.equal(mb.conf_u_w, logits.softmax(1).max(1)[0])


@needs_cuda
def test_strong_views_are_cutmixed_and_the_rest_passes_through():
    batch = next(semi_loader(0))
    (img_x, mask_x), (img_u_w, img_u_s1, img_u_s2, ignore_mask, box1, box2, mask_u_gt) = batch
    mb = load_micro_batch(iter([copy.deepcopy(batch)]), toy_teacher(0), sup_only=False, bf16=False)
    want_s1, want_s2 = img_u_s1.clone(), img_u_s2.clone()
    cutmix_(want_s1, box1)
    cutmix_(want_s2, box2)
    assert torch.equal(mb.img_u_s1.cpu(), want_s1)
    assert torch.equal(mb.img_u_s2.cpu(), want_s2)
    for got, want in [(mb.img_x, img_x), (mb.mask_x, mask_x), (mb.ignore_mask, ignore_mask),
                      (mb.cutmix_box1, box1), (mb.cutmix_box2, box2), (mb.mask_u_gt, mask_u_gt)]:
        assert got.is_cuda and got.dtype == want.dtype
        assert torch.equal(got.cpu(), want)


@needs_cuda
def test_each_call_consumes_exactly_one_batch():
    loader = semi_loader(0, n_batches=2)
    first = load_micro_batch(loader, toy_teacher(0), sup_only=False, bf16=False)
    second = load_micro_batch(loader, toy_teacher(0), sup_only=False, bf16=False)
    assert not torch.equal(first.img_x, second.img_x)
    with pytest.raises(StopIteration):
        load_micro_batch(loader, toy_teacher(0), sup_only=False, bf16=False)


@needs_cuda
@pytest.mark.parametrize('bf16', [False, True])
def test_with_the_real_dpt_teacher(bf16):
    """Shapes and dtypes with the actual model: the DPT head ends in a bilinear
    interpolate, which CUDA autocast runs in fp32, so the pseudo-label confidence is
    fp32 even under --bf16."""
    from model.semseg.dpt import DPT
    torch.manual_seed(0)
    teacher = DPT(encoder_size='small', features=64, out_channels=[48, 96, 192, 384], nclass=NCLASS).cuda().eval()
    crop = 14 * 8  # DINOv2 patch size 14; small multiple keeps the forward cheap
    gen = torch.Generator().manual_seed(0)
    img_x, mask_x = torch.randn(2, 3, crop, crop, generator=gen), torch.randint(0, NCLASS, (2, crop, crop), generator=gen)
    img_u = [torch.randn(2, 3, crop, crop, generator=gen) for _ in range(3)]
    box = torch.zeros(2, crop, crop)
    box[:, :crop // 2, :crop // 2] = 1
    batch = ((img_x, mask_x), (*img_u, torch.zeros(2, crop, crop, dtype=torch.long), box, box.clone(),
                               torch.randint(0, NCLASS, (2, crop, crop), generator=gen)))
    mb = load_micro_batch(iter([batch]), teacher, sup_only=False, bf16=bf16)
    assert mb.mask_u_w.shape == mb.conf_u_w.shape == (2, crop, crop)
    assert mb.mask_u_w.dtype == torch.long
    assert mb.conf_u_w.dtype == torch.float32
    assert bool(((mb.conf_u_w > 0) & (mb.conf_u_w <= 1)).all())
    assert bool((mb.mask_u_w >= 0).all()) and bool((mb.mask_u_w < NCLASS).all())
