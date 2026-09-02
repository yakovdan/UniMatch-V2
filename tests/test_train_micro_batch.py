"""Tests for train_micro_batch, the per-micro-batch forward/backward step extracted
out of the training loop of unimatch_v2_1gpu.py.

``reference_block`` is the loop body as it stood before the extraction, verbatim,
so the equivalence tests pin the refactor to the old behaviour. The rest checks
the properties the optimizer step relies on: what lands in .grad and in the
supervised flat buffer, how the accum scaling works, what the student and the
teacher are asked to do, and what the function reports back. The function moves
tensors to CUDA, so everything here needs a GPU; cuDNN is pinned to its
deterministic algorithms so repeated backwards can be compared exactly.
"""
import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from unimatch_v2_1gpu import (MicroBatchLosses, accumulate_class_grads, load_micro_batch,
                              train_micro_batch, unsup_forward_backward)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='train_micro_batch moves tensors to CUDA')

NCLASS = 5
B, H, W = 4, 16, 24
ACCUM = 4


@pytest.fixture(autouse=True)
def deterministic_cudnn():
    prev = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev


def reference_block(loader, model, model_ema, criterion_l, criterion_u, params, grad_x_acc,
                    accum, conf_thresh, sup_only, bf16, proto, anchor_params, proto_acc, proto_cnt):
    """The micro-batch body of the training loop before the extraction, verbatim,
    returning what the loop's bookkeeping consumed."""
    mb = load_micro_batch(loader, model_ema, sup_only, bf16)
    img_x, mask_x = mb.img_x, mb.mask_x

    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        pred_x = model(img_x)
        loss_x = criterion_l(pred_x, mask_x)
    if proto is not None:
        accumulate_class_grads(pred_x, mask_x, anchor_params, proto_acc, proto_cnt)
    grads = torch.autograd.grad(loss_x / (2.0 * accum), params, allow_unused=True)
    for p, g_x, g in zip(params, grad_x_acc, grads):
        if g is None:
            continue
        g_x += g
        p.grad = g.contiguous() if p.grad is None else p.grad.add_(g)
    del grads

    loss_u_s, mask_ratio = unsup_forward_backward(model, mb, criterion_u, conf_thresh, accum, sup_only, bf16)

    loss = (loss_x.detach() + loss_u_s.detach()) / 2.0
    return loss.item(), loss_x.item(), loss_u_s.item(), mask_ratio


# ----------------------------------------------------------------------------
# a toy student / teacher and synthetic batches shaped like the real ones
# ----------------------------------------------------------------------------

class ToyStudent(nn.Module):
    """Stands in for DPT: a small conv net whose forward takes DPT's comp_drop flag.
    Every call is recorded; the flag itself is ignored so results stay deterministic."""

    def __init__(self, spare_param=False):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, NCLASS, 1))
        if spare_param:
            self.spare = nn.Parameter(torch.randn(3))  # reached by no loss
        self.calls = []

    def forward(self, x, comp_drop=False):
        self.calls.append({'comp_drop': comp_drop, 'batch': x.shape[0],
                           'autocast': torch.is_autocast_enabled('cuda'), 'grad': torch.is_grad_enabled()})
        return self.body(x)


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, NCLASS, 1)
        self.calls = []

    def forward(self, x):
        self.calls.append({'grad': torch.is_grad_enabled(), 'autocast': torch.is_autocast_enabled('cuda')})
        return self.proj(x)


def flat_views(flat, params):
    views, off = [], 0
    for p in params:
        views.append(flat[off:off + p.numel()].view_as(p))
        off += p.numel()
    return views


class Setup:
    """A student, a teacher, the trainable parameter list and the supervised flat
    buffer with its per-parameter views, as main() builds them."""

    def __init__(self, seed, spare_param=False, with_proto=False):
        torch.manual_seed(seed)
        self.model = ToyStudent(spare_param).cuda().train()
        self.teacher = ToyTeacher().cuda().eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.flat_x = torch.zeros(sum(p.numel() for p in self.params), device='cuda')
        self.grad_x_acc = flat_views(self.flat_x, self.params)
        self.criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
        self.criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
        self.anchor_params = self.proto_acc = self.proto_cnt = self.proto = None
        if with_proto:
            self.anchor_params = self.params
            self.proto = torch.zeros(NCLASS, self.flat_x.numel(), device='cuda')
            self.proto_acc = torch.zeros_like(self.proto)
            self.proto_cnt = torch.zeros(NCLASS, device='cuda')

    def run(self, loader, sup_only=False, bf16=False, accum=ACCUM, conf_thresh=0.0):
        return train_micro_batch(loader, self.model, self.teacher, self.criterion_l, self.criterion_u,
                                 self.params, self.grad_x_acc, accum, conf_thresh, sup_only, bf16,
                                 anchor_params=self.anchor_params, proto_acc=self.proto_acc, proto_cnt=self.proto_cnt)

    def run_reference(self, loader, sup_only=False, bf16=False, accum=ACCUM, conf_thresh=0.0):
        return reference_block(loader, self.model, self.teacher, self.criterion_l, self.criterion_u,
                               self.params, self.grad_x_acc, accum, conf_thresh, sup_only, bf16,
                               self.proto, self.anchor_params, self.proto_acc, self.proto_cnt)

    def grads(self):
        return [p.grad for p in self.params]


def random_boxes(gen):
    box = torch.zeros(B, H, W)
    for i in range(B):
        if i == 2:
            continue  # one image without a box, as the dataset produces
        h, w = int(torch.randint(H // 4, H // 2 + 1, (), generator=gen)), int(torch.randint(W // 4, W // 2 + 1, (), generator=gen))
        y, x = int(torch.randint(0, H - h + 1, (), generator=gen)), int(torch.randint(0, W - w + 1, (), generator=gen))
        box[i, y:y + h, x:x + w] = 1
    return box


def labeled_batch(gen):
    img = torch.randn(B, 3, H, W, generator=gen)
    mask = torch.randint(0, NCLASS, (B, H, W), generator=gen)
    mask[:, :2] = 255
    return img, mask


def unlabeled_batch(gen):
    img_u_w, img_u_s1, img_u_s2 = (torch.randn(B, 3, H, W, generator=gen) for _ in range(3))
    ignore_mask = torch.zeros(B, H, W, dtype=torch.long)
    ignore_mask[:, -3:] = 255
    mask_u_gt = torch.randint(0, NCLASS, (B, H, W), generator=gen)
    return img_u_w, img_u_s1, img_u_s2, ignore_mask, random_boxes(gen), random_boxes(gen), mask_u_gt


def batches(seed, n=1, sup_only=False):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        lab = labeled_batch(gen)
        out.append(lab if sup_only else (lab, unlabeled_batch(gen)))
    return out


def loader_of(batch_list):
    return iter([copy.deepcopy(b) for b in batch_list])


def assert_equal_tensors(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b)


# ----------------------------------------------------------------------------
# equivalence with the pre-refactor loop body
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('with_proto', [False, True])
@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('sup_only', [False, True])
@pytest.mark.parametrize('seed', [0, 1])
def test_matches_reference(seed, sup_only, bf16, with_proto):
    data = batches(seed, n=2, sup_only=sup_only)
    new, old = Setup(seed, with_proto=with_proto), Setup(seed, with_proto=with_proto)
    ln, lo = loader_of(data), loader_of(data)
    for _ in range(2):  # a second call exercises the accumulate-into-.grad path
        torch.manual_seed(seed)
        rn = new.run(ln, sup_only, bf16)
        torch.manual_seed(seed)
        ro = old.run_reference(lo, sup_only, bf16)
        assert tuple(rn) == tuple(ro)
    assert_equal_tensors(new.flat_x, old.flat_x)
    for a, b in zip(new.grads(), old.grads()):
        assert_equal_tensors(a, b)
    if with_proto:
        assert_equal_tensors(new.proto_acc, old.proto_acc)
        assert_equal_tensors(new.proto_cnt, old.proto_cnt)


def test_returns_floats_in_a_named_tuple():
    s = Setup(0)
    r = s.run(loader_of(batches(0)))
    assert isinstance(r, MicroBatchLosses)
    assert all(isinstance(v, float) for v in r)
    assert r == MicroBatchLosses(r.loss, r.loss_x, r.loss_s, r.mask_ratio)


# ----------------------------------------------------------------------------
# what lands in .grad and in the supervised flat buffer
# ----------------------------------------------------------------------------

def test_supervised_gradient_is_the_scaled_gradient_of_loss_x():
    data = batches(0, sup_only=True)
    s = Setup(0)
    r = s.run(loader_of(data), sup_only=True)
    # the same loss on an identical model, plain backward
    ref = Setup(0)
    img, mask = data[0]
    loss_x = ref.criterion_l(ref.model(img.cuda()), mask.cuda())
    (loss_x / (2.0 * ACCUM)).backward()
    assert r.loss_x == pytest.approx(loss_x.item())
    for p, q, g_x in zip(s.params, ref.params, s.grad_x_acc):
        torch.testing.assert_close(p.grad, q.grad)
        assert torch.equal(p.grad, g_x)  # sup-only: nothing else reached .grad


def test_flat_buffer_views_alias_flat_x():
    s = Setup(0)
    s.run(loader_of(batches(0, sup_only=True)), sup_only=True)
    assert s.flat_x.abs().sum() > 0
    assert torch.equal(torch.cat([v.reshape(-1) for v in s.grad_x_acc]), s.flat_x)


def test_unsupervised_gradient_is_grad_minus_flat_x():
    data = batches(0)
    s = Setup(0)
    torch.manual_seed(0)
    s.run(loader_of(data))
    # the unsupervised half alone on an identical model, from the same prepared micro-batch
    ref = Setup(0)
    mb = load_micro_batch(loader_of(data), ref.teacher, sup_only=False, bf16=False)
    torch.manual_seed(0)
    loss_u_s, _ = unsup_forward_backward(ref.model, mb, ref.criterion_u, 0.0, ACCUM, False, False)
    assert loss_u_s.item() > 0
    for p, q, g_x in zip(s.params, ref.params, s.grad_x_acc):
        torch.testing.assert_close(p.grad - g_x, q.grad)


def test_sup_only_leaves_the_unsupervised_part_exactly_zero():
    s = Setup(0)
    r = s.run(loader_of(batches(0, sup_only=True)), sup_only=True)
    assert r.loss_s == 0.0 and r.mask_ratio == 0.0
    assert r.loss == pytest.approx(r.loss_x / 2.0)
    for p, g_x in zip(s.params, s.grad_x_acc):
        assert torch.equal(p.grad, g_x)


def test_calls_accumulate_rather_than_overwrite():
    data = batches(0, n=2)
    both = Setup(0)
    torch.manual_seed(0)
    both.run(loader_of(data))
    torch.manual_seed(1)
    both.run(loader_of(data[1:]))
    singles = []
    for i, seed in enumerate((0, 1)):
        one = Setup(0)
        torch.manual_seed(seed)
        one.run(loader_of(data[i:i + 1]))
        singles.append(one)
    torch.testing.assert_close(both.flat_x, singles[0].flat_x + singles[1].flat_x)
    for p, a, b in zip(both.params, singles[0].params, singles[1].params):
        torch.testing.assert_close(p.grad, a.grad + b.grad)


def test_second_call_adds_in_place_into_the_same_grad_tensors():
    data = batches(0, n=2)
    s = Setup(0)
    s.run(loader_of(data))
    first = [p.grad for p in s.params]
    assert all(g is not None and g.is_contiguous() for g in first)
    s.run(loader_of(data[1:]))
    assert all(p.grad is g for p, g in zip(s.params, first))


@pytest.mark.parametrize('accum', [1, 2, 8])
def test_gradients_scale_with_one_over_accum_and_losses_do_not(accum):
    data = batches(0)
    a, b = Setup(0), Setup(0)
    torch.manual_seed(0)
    ra = a.run(loader_of(data), accum=1)
    torch.manual_seed(0)
    rb = b.run(loader_of(data), accum=accum)
    assert ra == rb
    torch.testing.assert_close(b.flat_x * accum, a.flat_x)
    for p, q in zip(b.params, a.params):
        torch.testing.assert_close(p.grad * accum, q.grad)


def test_a_parameter_no_loss_reaches_keeps_grad_none_and_a_zero_slice():
    data = batches(0)
    s = Setup(0, spare_param=True)
    s.run(loader_of(data))
    idx = s.params.index(s.model.spare)
    assert s.model.spare.grad is None
    assert not s.grad_x_acc[idx].any()
    assert all(p.grad is not None for i, p in enumerate(s.params) if i != idx)
    # the offsets after the spare parameter are still right: the last slice is a real gradient
    assert s.grad_x_acc[-1].abs().sum() > 0
    torch.optim.AdamW(s.params, lr=1e-3).step()  # None grads are skipped, not an error


# ----------------------------------------------------------------------------
# what the student and the teacher are asked to do
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('bf16', [False, True])
def test_student_forwards_and_teacher_forward_in_semi_mode(bf16):
    s = Setup(0)
    s.run(loader_of(batches(0)), bf16=bf16)
    assert [c['comp_drop'] for c in s.model.calls] == [False, True]  # labeled view, then both strong views
    assert [c['batch'] for c in s.model.calls] == [B, 2 * B]
    assert all(c['grad'] and c['autocast'] is bf16 for c in s.model.calls)
    assert len(s.teacher.calls) == 1
    assert s.teacher.calls[0] == {'grad': False, 'autocast': bf16}


@pytest.mark.parametrize('bf16', [False, True])
def test_student_forward_only_in_sup_only_mode(bf16):
    s = Setup(0)
    s.run(loader_of(batches(0, sup_only=True)), sup_only=True, bf16=bf16)
    assert [c['comp_drop'] for c in s.model.calls] == [False]
    assert s.model.calls[0]['autocast'] is bf16
    assert s.teacher.calls == []


def test_each_call_consumes_exactly_one_batch():
    data = batches(0, n=2)
    s = Setup(0)
    loader = loader_of(data)
    r1 = s.run(loader)
    r2 = s.run(loader)
    assert r1.loss_x != r2.loss_x
    with pytest.raises(StopIteration):
        s.run(loader)


# ----------------------------------------------------------------------------
# what it reports back
# ----------------------------------------------------------------------------

def test_loss_is_the_mean_of_the_two_halves():
    s = Setup(0)
    r = s.run(loader_of(batches(0)))
    assert r.loss_x > 0 and r.loss_s > 0
    assert r.loss == pytest.approx((r.loss_x + r.loss_s) / 2.0, rel=1e-6)


def test_mask_ratio_follows_the_confidence_threshold():
    data = batches(0)
    every = Setup(0).run(loader_of(data), conf_thresh=0.0)
    assert every.mask_ratio == 1.0
    s = Setup(0)
    none = s.run(loader_of(data), conf_thresh=2.0)  # nothing is ever that confident
    assert none.mask_ratio == 0.0 and none.loss_s == 0.0
    assert every.loss_x == none.loss_x  # the labeled half does not depend on the threshold
    for p, g_x in zip(s.params, s.grad_x_acc):
        assert torch.equal(p.grad, g_x)  # a fully masked unsupervised loss backwards exact zeros


def test_mask_ratio_matches_a_direct_count():
    data = batches(0)
    s = Setup(0)
    thresh = 0.25
    r = s.run(loader_of(data), conf_thresh=thresh)
    mb = load_micro_batch(loader_of(data), Setup(0).teacher, sup_only=False, bf16=False)
    valid = mb.ignore_mask != 255
    assert r.mask_ratio == ((mb.conf_u_w >= thresh) & valid).sum().item() / valid.sum().item()
    assert 0.0 < r.mask_ratio < 1.0


@pytest.mark.parametrize('bf16', [False, True])
def test_results_are_finite_and_grads_stay_fp32(bf16):
    s = Setup(0)
    r = s.run(loader_of(batches(0)), bf16=bf16)
    assert all(v == v and abs(v) < float('inf') for v in r)
    assert s.flat_x.dtype == torch.float32 and torch.isfinite(s.flat_x).all()
    assert all(p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in s.params)


# ----------------------------------------------------------------------------
# the per-class prototype accumulators
# ----------------------------------------------------------------------------

def test_prototype_accumulators_follow_the_labeled_mask():
    data = batches(0)
    s = Setup(0, with_proto=True)
    s.run(loader_of(data))
    (img, mask), _ = data[0]
    present = sorted(c for c in mask.unique().tolist() if c != 255)
    assert [c for c in range(NCLASS) if s.proto_cnt[c] == 1] == present
    assert all(s.proto_acc[c].abs().sum() > 0 for c in present)
    # the rows are what accumulate_class_grads produces on the same forward
    ref = Setup(0, with_proto=True)
    pred = ref.model(img.cuda())
    accumulate_class_grads(pred, mask.cuda(), ref.params, ref.proto_acc, ref.proto_cnt)
    torch.testing.assert_close(s.proto_acc, ref.proto_acc)


def test_without_accumulators_nothing_per_class_happens():
    s = Setup(0, with_proto=False)
    calls_before = len(s.model.calls)
    s.run(loader_of(batches(0)))
    assert s.proto_acc is None and s.proto_cnt is None
    assert len(s.model.calls) == calls_before + 2


def test_prototypes_do_not_change_the_applied_gradient():
    data = batches(0)
    with_p, without = Setup(0, with_proto=True), Setup(0, with_proto=False)
    torch.manual_seed(0)
    r1 = with_p.run(loader_of(data))
    torch.manual_seed(0)
    r2 = without.run(loader_of(data))
    assert r1 == r2
    assert torch.equal(with_p.flat_x, without.flat_x)
    for p, q in zip(with_p.params, without.params):
        assert torch.equal(p.grad, q.grad)


# ----------------------------------------------------------------------------
# the real model
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('bf16', [False, True])
def test_with_the_real_dpt(bf16):
    from model.semseg.dpt import DPT
    torch.manual_seed(0)
    model = DPT(encoder_size='small', features=64, out_channels=[48, 96, 192, 384], nclass=21).cuda().train()
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    flat_x = torch.zeros(sum(p.numel() for p in params), device='cuda')
    grad_x_acc = flat_views(flat_x, params)
    head = [p for n, p in model.named_parameters() if 'backbone' not in n]
    proto_acc = torch.zeros(21, sum(p.numel() for p in head), device='cuda')
    proto_cnt = torch.zeros(21, device='cuda')
    crop = 14 * 8
    gen = torch.Generator().manual_seed(0)
    img_x, mask_x = torch.randn(2, 3, crop, crop, generator=gen), torch.randint(0, 21, (2, crop, crop), generator=gen)
    img_u = [torch.randn(2, 3, crop, crop, generator=gen) for _ in range(3)]
    box = torch.zeros(2, crop, crop)
    box[:, :crop // 2, :crop // 2] = 1
    batch = ((img_x, mask_x), (*img_u, torch.zeros(2, crop, crop, dtype=torch.long), box, box.clone(),
                               torch.randint(0, 21, (2, crop, crop), generator=gen)))
    r = train_micro_batch(iter([batch]), model, teacher, nn.CrossEntropyLoss(ignore_index=255).cuda(),
                          nn.CrossEntropyLoss(reduction='none').cuda(), params, grad_x_acc, ACCUM, 0.0, False, bf16,
                          anchor_params=head, proto_acc=proto_acc, proto_cnt=proto_cnt)
    assert all(v == v and abs(v) < float('inf') for v in r) and r.mask_ratio == 1.0
    # DPT has parameters no forward reaches (DINOv2's mask_token, the top refinenet's
    # first residual unit): a plain backward on a fresh copy says which, and the
    # function must leave exactly those at None
    plain = copy.deepcopy(teacher).train()
    for p in plain.parameters():
        p.requires_grad = True
    F.cross_entropy(plain(img_x.cuda()), mask_x.cuda()).backward()
    unreached = {n for n, p in plain.named_parameters() if p.grad is None}
    assert unreached and len(unreached) < 10
    for n, p in model.named_parameters():
        if n in unreached:
            assert p.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all()
    assert flat_x.abs().sum() > 0 and torch.isfinite(flat_x).all()
    assert proto_cnt.sum() == len([c for c in mask_x.unique().tolist() if c != 255])
    # backbone and head both received supervised and unsupervised gradient
    n_backbone = len([p for p in model.backbone.parameters() if p.requires_grad])
    for lo, hi in ((0, n_backbone), (n_backbone, len(params))):
        reached = [i for i in range(lo, hi) if params[i].grad is not None]
        gx = torch.cat([grad_x_acc[i].reshape(-1) for i in reached])
        gs = torch.cat([params[i].grad.reshape(-1) for i in reached]) - gx
        assert gx.abs().sum() > 0 and gs.abs().sum() > 0
