"""Tests for --frozen-grad: the Complementary Dropout mask reuse in DPT.forward, the unsup_loss
extraction out of unsup_forward_backward, and util/frozen_grad.py (identity with the student's
own backbone gradient while the student still sits at the checkpoint, no side effects on the
model / .grad / RNG, parameter grouping, the locked-backbone case, flag and env override), --pla
(FrozenGrad.align) and its matched controls (--pla-control random / flip).

The decisive check is the identity one: with the student's backbone loaded from the same
checkpoint as the frozen copy, the counterfactual gradient IS the student's backbone gradient,
so every group must read cosine 1 and norm ratio 1. A real DINOv2-S is used (random init saved
as the "checkpoint"), on small 56x56 inputs. Everything moves tensors to CUDA.
"""
import copy
import os

import pytest
import torch
from torch import nn

from model.semseg.dpt import DPT
from unimatch_v2_1gpu import (MicroBatch, apply_env_overrides, check_unlock_after, parser, train_micro_batch,
                              unsup_forward_backward, unsup_loss)
from util.frozen_grad import FrozenGrad, group_of

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')

NCLASS, B, H, W = 5, 4, 56, 56  # 56 = 4 patches of 14
ACCUM = 4
REQUIRED = ['--config', 'c.yaml', '--labeled-id-path', 'l.txt', '--unlabeled-id-path', 'u.txt', '--save-path', 's']


@pytest.fixture(autouse=True)
def deterministic_cudnn():
    prev = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev


def small_dpt(seed, ckpt_path=None):
    """A dinov2_small DPT at random init; its backbone weights are saved to ckpt_path so a
    FrozenGrad built from that file holds exactly the student's backbone."""
    torch.manual_seed(seed)
    model = DPT(encoder_size='small', nclass=NCLASS, features=64, out_channels=[48, 96, 192, 384])
    if ckpt_path is not None:
        torch.save(model.backbone.state_dict(), ckpt_path)
    return model.cuda().train()


def random_boxes(gen):
    box = torch.zeros(B, H, W)
    for i in range(B):
        if i == 2:
            continue
        h, w = int(torch.randint(H // 4, H // 2 + 1, (), generator=gen)), int(torch.randint(W // 4, W // 2 + 1, (), generator=gen))
        y, x = int(torch.randint(0, H - h + 1, (), generator=gen)), int(torch.randint(0, W - w + 1, (), generator=gen))
        box[i, y:y + h, x:x + w] = 1
    return box


def batch(seed):
    """One (labeled, unlabeled) loader item shaped like the real ones."""
    gen = torch.Generator().manual_seed(seed)
    img_x = torch.randn(B, 3, H, W, generator=gen)
    mask_x = torch.randint(0, NCLASS, (B, H, W), generator=gen)
    mask_x[:, :4] = 255
    img_u_w, img_u_s1, img_u_s2 = (torch.randn(B, 3, H, W, generator=gen) for _ in range(3))
    ignore = torch.zeros(B, H, W, dtype=torch.long)
    ignore[:, -5:] = 255
    gt = torch.randint(0, NCLASS, (B, H, W), generator=gen)
    return (img_x, mask_x), (img_u_w, img_u_s1, img_u_s2, ignore, random_boxes(gen), random_boxes(gen), gt)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(123)
        self.proj = nn.Conv2d(3, NCLASS, 1)

    def forward(self, x):
        return self.proj(x)


def flat_views(flat, params):
    views, off = [], 0
    for p in params:
        views.append(flat[off:off + p.numel()].view_as(p))
        off += p.numel()
    return views


class Run:
    """A student at the checkpoint, its FrozenGrad twin, and the flat supervised buffer as main()
    builds it; ``step`` runs one train_micro_batch with the counterfactual pass."""

    def __init__(self, tmp_path, seed=0):
        ckpt = os.path.join(str(tmp_path), 'bb.pth')
        self.model = small_dpt(seed, ckpt)
        self.frozen = FrozenGrad(self.model, 'dinov2_small', ckpt)
        self.teacher = Teacher().cuda().eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.n_backbone = len(list(self.model.backbone.parameters()))
        self.flat_x = torch.zeros(sum(p.numel() for p in self.params), device='cuda')
        self.grad_x_acc = flat_views(self.flat_x, self.params)
        self.criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
        self.criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    def step(self, seed=0, sup_only=False, bf16=False, conf_thresh=0.0, frozen=True):
        item = batch(seed)
        loader = iter([item[0] if sup_only else item])
        stats = {}
        r = train_micro_batch(loader, self.model, self.teacher, self.criterion_l, self.criterion_u,
                              self.params, self.grad_x_acc, ACCUM, conf_thresh, sup_only, bf16,
                              stats=stats, frozen=self.frozen if frozen else None)
        return r, stats

    def backbone_grads(self):
        gx = [g.clone() for g in self.grad_x_acc[:self.n_backbone]]
        gs = [(p.grad - g).clone() if p.grad is not None else torch.zeros_like(g)
              for p, g in zip(self.params[:self.n_backbone], self.grad_x_acc[:self.n_backbone])]
        return gx, gs


# ----------------------------------------------------------------------------
# DPT.forward: mask reuse
# ----------------------------------------------------------------------------

def test_forward_reuses_a_given_mask_without_drawing_rng():
    model = small_dpt(0)
    x = torch.randn(2 * B, 3, H, W, device='cuda')
    torch.manual_seed(1)
    out1 = model(x, comp_drop=True)
    m = model.last_dropout_mask
    assert m is not None and tuple(m.shape) == (2 * B, model.backbone.embed_dim)
    state = torch.get_rng_state()
    out2 = model(x, dropout_mask=m)
    assert torch.equal(torch.get_rng_state(), state)  # no draw
    assert torch.equal(out1, out2)
    assert model.last_dropout_mask is m
    # a fresh comp_drop forward draws a different mask (and records it)
    out3 = model(x, comp_drop=True)
    assert not torch.equal(model.last_dropout_mask, m) and not torch.equal(out3, out1)
    # no dropout: mask None, and forward == backbone + forward_head
    out4 = model(x)
    assert model.last_dropout_mask is None
    feats = model.backbone.get_intermediate_layers(x, model.intermediate_layer_idx['small'])
    assert torch.equal(out4, model.forward_head(feats, H // 14, W // 14))


def test_sample_dropout_mask_is_complementary_with_kept_pairs():
    model = small_dpt(0)
    torch.manual_seed(2)
    m = model.sample_dropout_mask(2 * B, 16)
    m1, m2 = m[:B], m[B:]
    assert set(m.unique().tolist()) <= {0.0, 1.0, 2.0}
    kept = (m1 == 1).all(1)
    assert kept.sum() == B // 2 and torch.equal(kept, (m2 == 1).all(1))
    assert torch.equal(m1[~kept] + m2[~kept], torch.full_like(m1[~kept], 2.0))
    assert not (m1[~kept] == 1).any()


# ----------------------------------------------------------------------------
# unsup_loss extraction
# ----------------------------------------------------------------------------

class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, NCLASS, 1))

    def forward(self, x, comp_drop=False):
        return self.body(x)


def synth_mb(seed, ref=None):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).cuda()
    mask_u_w = torch.randint(0, NCLASS, (B, H, W), generator=g)
    box = random_boxes(g)
    return MicroBatch(r(B, 3, H, W), torch.randint(0, NCLASS, (B, H, W), generator=g).cuda(), r(B, 3, H, W), r(B, 3, H, W),
                      mask_u_w.cuda(), torch.rand(B, H, W, generator=g).cuda(), torch.zeros(B, H, W, dtype=torch.long).cuda(),
                      box.cuda(), random_boxes(g).cuda(), torch.randint(0, NCLASS, (B, H, W), generator=g).cuda(),
                      None if ref is None else ref(mask_u_w).cuda())


@pytest.mark.parametrize('gated', [False, True])
@pytest.mark.parametrize('bf16', [False, True])
def test_unsup_loss_is_what_unsup_forward_backward_backwards(gated, bf16):
    crit = nn.CrossEntropyLoss(reduction='none').cuda()
    torch.manual_seed(0)
    model = ToyStudent().cuda().train()
    mb = synth_mb(3, ref=(lambda m: (m + 1) % NCLASS) if gated else None)
    loss_fb, _ = unsup_forward_backward(model, mb, crit, 0.5, ACCUM, False, bf16)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        p1, p2 = model(torch.cat((mb.img_u_s1, mb.img_u_s2)), comp_drop=True).chunk(2)
    loss, disputed = unsup_loss(p1, p2, mb, crit, 0.5, bf16)
    assert loss.item() == pytest.approx(loss_fb.item(), rel=1e-6)
    assert len(disputed) == (2 if gated else 0)
    # and its gradient, scaled like the trainer does, is what landed in .grad
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    (loss / (2.0 * ACCUM)).backward()
    for g, p in zip(grads, model.parameters()):
        torch.testing.assert_close(p.grad, g)


# ----------------------------------------------------------------------------
# FrozenGrad
# ----------------------------------------------------------------------------

def test_parameter_groups_cover_every_backbone_parameter_once():
    model = small_dpt(0)
    names = [n for n, _ in model.backbone.named_parameters()]
    groups = [group_of(n) for n in names]
    assert len(groups) == len(names) == 175
    assert sorted(set(groups)) == sorted(['block%02d' % i for i in range(12)] + ['embed', 'norm'])
    assert group_of('blocks.7.attn.qkv.weight') == 'block07'
    assert group_of('patch_embed.proj.bias') == group_of('cls_token') == group_of('mask_token') == 'embed'
    with pytest.raises(ValueError):
        group_of('head.weight')


@pytest.mark.parametrize('bf16', [False, True])
def test_frozen_gradient_equals_the_students_backbone_gradient_at_the_checkpoint(tmp_path, bf16):
    run = Run(tmp_path)
    assert run.frozen.group_names[0] == 'embed' and run.frozen.group_names[-1] == 'norm'
    r, stats = run.step(bf16=bf16)
    # the counterfactual losses are the student's own losses: same weights, same head, same mask
    assert stats['loss_x_frozen'] == pytest.approx(r.loss_x, rel=1e-5)
    assert stats['loss_s_frozen'] == pytest.approx(r.loss_s, rel=1e-5)
    gx, gs = run.backbone_grads()
    tol = dict(rtol=2e-2, atol=1e-6) if bf16 else dict(rtol=1e-4, atol=1e-8)
    for a, b in zip(gx, run.frozen.acc_x):
        torch.testing.assert_close(b, a, **tol)
    for a, b in zip(gs, run.frozen.acc_s):
        torch.testing.assert_close(b, a, **tol)
    st = run.frozen.step_stats(*run.backbone_grads())  # mask_token is reached by no loss: .grad None, zero on both sides
    ctol = 1e-3 if bf16 else 1e-5
    for k in ('_all',) + tuple('/' + g for g in run.frozen.group_names):
        assert st['grad/frozen_cos' + k] == pytest.approx(1.0, abs=ctol), k
        assert st['grad/frozen_cos_x' + k] == pytest.approx(1.0, abs=ctol), k
        assert st['grad/frozen_cos_s' + k] == pytest.approx(1.0, abs=ctol), k
        assert st['grad/frozen_norm_ratio' + k] == pytest.approx(1.0, rel=ctol), k
    assert st['grad/frozen_norm_all'] > 0 and st['grad/frozen_norm_x_all'] > 0 and st['grad/frozen_norm_s_all'] > 0
    assert len(st) == 7 + 4 * len(run.frozen.group_names)


def test_frozen_gradient_diverges_once_the_student_backbone_moves(tmp_path):
    run = Run(tmp_path)
    with torch.no_grad():
        for p in run.model.backbone.blocks[11].parameters():
            p.add_(0.5 * torch.randn_like(p))
    run.step()
    st = run.frozen.step_stats(*run.backbone_grads())
    assert st['grad/frozen_cos/block11'] < 0.999
    assert st['grad/frozen_cos_all'] < 0.999


def test_counterfactual_pass_touches_nothing_of_the_student(tmp_path):
    run = Run(tmp_path)
    # a first step (without the frozen pass) puts gradients in .grad; the pass must leave them alone
    run.step(frozen=False)
    weights = [p.detach().clone() for p in run.model.parameters()]
    grads = [None if p.grad is None else p.grad.clone() for p in run.model.parameters()]
    item = batch(5)
    mb_x, (_, s1, s2, *_) = item
    mask = run.model.sample_dropout_mask(2 * B, run.model.backbone.embed_dim)  # the test's own draw, before the snapshot
    crit = nn.CrossEntropyLoss(ignore_index=255).cuda()
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    lx, ls = run.frozen.micro_batch(
        run.model, mb_x[0].cuda(), lambda pred: crit(pred, mb_x[1].cuda()),
        torch.cat((s1, s2)).cuda(), lambda p1, p2: (p1.mean() + p2.mean()), mask, ACCUM, False)
    assert torch.isfinite(lx) and torch.isfinite(ls)
    assert torch.equal(torch.get_rng_state(), cpu_state) and torch.equal(torch.cuda.get_rng_state(), cuda_state)
    for p, w, g in zip(run.model.parameters(), weights, grads):
        assert torch.equal(p.detach(), w)
        assert (g is None and p.grad is None) or torch.equal(p.grad, g)
    assert all(p.grad is None for p in run.frozen.params)
    # frozen weights stay the checkpoint
    ck = torch.load(os.path.join(str(tmp_path), 'bb.pth'))
    for n, p in run.frozen.bb.named_parameters():
        assert torch.equal(p.detach().cpu(), ck[n])


def test_sup_only_skips_the_unsupervised_half(tmp_path):
    run = Run(tmp_path)
    r, stats = run.step(sup_only=True)
    assert stats['loss_x_frozen'] == pytest.approx(r.loss_x, rel=1e-5)
    assert stats['loss_s_frozen'] != stats['loss_s_frozen']  # NaN
    assert run.frozen.flat_s.abs().sum() == 0 and run.frozen.flat_x.abs().sum() > 0
    st = run.frozen.step_stats(*run.backbone_grads())
    assert st['grad/frozen_cos_x_all'] == pytest.approx(1.0, abs=1e-5)
    assert st['grad/frozen_cos_s_all'] != st['grad/frozen_cos_s_all']  # NaN: no unsupervised gradient on either side


def test_locked_backbone_gives_norms_and_nan_cosines(tmp_path):
    run = Run(tmp_path)
    run.step()
    st = run.frozen.step_stats()  # no trainable views at all: lock_backbone's n_backbone == 0
    assert st['grad/frozen_norm_all'] > 0
    assert all(st[k] != st[k] for k in st if 'cos' in k or 'ratio' in k)


def test_zero_resets_and_accumulation_is_additive(tmp_path):
    run = Run(tmp_path)
    run.step(seed=0)
    once = run.frozen.flat_x.clone()
    run.step(seed=0)
    torch.testing.assert_close(run.frozen.flat_x, 2 * once)
    run.frozen.zero_()
    assert run.frozen.flat_x.abs().sum() == 0 and run.frozen.flat_s.abs().sum() == 0


def test_flag_default_and_env_override():
    import logging
    args = parser.parse_args(REQUIRED)
    assert args.frozen_grad is False
    args = parser.parse_args(REQUIRED + ['--frozen-grad'])
    assert args.frozen_grad is True
    args = parser.parse_args(REQUIRED)
    os.environ['FROZEN_GRAD'] = '1'
    try:
        apply_env_overrides(args, logging.getLogger('t'))
    finally:
        del os.environ['FROZEN_GRAD']
    assert args.frozen_grad is True


# ----------------------------------------------------------------------------
# --pla: FrozenGrad.align, the pairwise layer-wise projection
# ----------------------------------------------------------------------------

def group_dots(fr, g_views, f_views):
    """Per-group (<g, f>, ||f||^2, ||g||^2) in float64."""
    out = {}
    for grp, idx in fr.group_idx.items():
        d = sum(float((g_views[i].double() * f_views[i].double()).sum()) for i in idx)
        ff = sum(float((f_views[i].double() ** 2).sum()) for i in idx)
        gg = sum(float((g_views[i].double() ** 2).sum()) for i in idx)
        out[grp] = (d, ff, gg)
    return out


def test_align_fires_nothing_at_the_checkpoint(tmp_path):
    run = Run(tmp_path)
    run.step()
    gx, gs = run.backbone_grads()
    grads = [None if p.grad is None else p.grad.clone() for p in run.params[:run.n_backbone]]
    st = run.frozen.align(run.grad_x_acc[:run.n_backbone],
                          [p.grad - g if p.grad is not None else torch.zeros_like(g)
                           for p, g in zip(run.params[:run.n_backbone], run.grad_x_acc[:run.n_backbone])],
                          run.params[:run.n_backbone])
    assert st['grad/pla_fired_x_frac'] == 0.0 and st['grad/pla_fired_s_frac'] == 0.0
    assert st['grad/pla_norm_kept_x'] == 1.0 and st['grad/pla_norm_kept_s'] == 1.0
    for a, b in zip(gx, run.grad_x_acc[:run.n_backbone]):
        assert torch.equal(a, b)
    for p, g in zip(run.params[:run.n_backbone], grads):
        assert (g is None and p.grad is None) or torch.equal(p.grad, g)
    assert set(k.rsplit('/', 1)[0] for k in st if '/pla_fired_x/' in k) == {'grad/pla_fired_x'}
    assert len(st) == 2 * len(run.frozen.group_names) + 4


def synthetic_halves(run, seed=7, noise=0.1):
    """Random g / f buffers with a known sign pattern -- odd groups oppose in x, even groups oppose in s,
    |cos(g, f)| ~ 1 / sqrt(1 + noise^2) -- and .grad = g_x + g_s as the trainer leaves it before align.
    Returns (gx_views, gs_views); gx_views are the run's flat supervised buffer."""
    fr = run.frozen
    gen = torch.Generator(device='cuda').manual_seed(seed)
    nb = run.n_backbone
    gx_views, gs_views = run.grad_x_acc[:nb], flat_views(torch.zeros_like(run.flat_x), run.params)[:nb]
    for views in (gx_views, gs_views, fr.acc_x, fr.acc_s):
        for v in views:
            v.copy_(torch.randn(v.shape, generator=gen, device='cuda'))
    for k, grp in enumerate(fr.group_names):
        for i in fr.group_idx[grp]:
            if k % 2 == 1:
                gx_views[i].copy_(-fr.acc_x[i] + noise * torch.randn(gx_views[i].shape, generator=gen, device='cuda'))
            else:
                gx_views[i].copy_(fr.acc_x[i] + noise * torch.randn(gx_views[i].shape, generator=gen, device='cuda'))
            if k % 2 == 0:
                gs_views[i].copy_(-fr.acc_s[i] + noise * torch.randn(gs_views[i].shape, generator=gen, device='cuda'))
            else:
                gs_views[i].copy_(fr.acc_s[i] + noise * torch.randn(gs_views[i].shape, generator=gen, device='cuda'))
    for p, gx, gs in zip(run.params[:nb], gx_views, gs_views):
        p.grad = gx + gs
    return gx_views, gs_views


@pytest.mark.parametrize('target', ['both', 'sup', 'unsup'])
def test_align_projects_exactly_the_opposing_groups(tmp_path, target):
    run = Run(tmp_path)
    fr = run.frozen
    nb = run.n_backbone
    gx_views, gs_views = synthetic_halves(run)
    before_x, before_s = group_dots(fr, gx_views, fr.acc_x), group_dots(fr, gs_views, fr.acc_s)
    gx0 = [v.clone() for v in gx_views]; gs0 = [v.clone() for v in gs_views]
    st = fr.align(gx_views, gs_views, run.params[:nb], target)
    after_x, after_s = group_dots(fr, gx_views, fr.acc_x), group_dots(fr, gs_views, fr.acc_s)
    for tag, before, after, views0, views, active in (('x', before_x, after_x, gx0, gx_views, target in ('both', 'sup')),
                                                       ('s', before_s, after_s, gs0, gs_views, target in ('both', 'unsup'))):
        if not active:
            assert not any(k.startswith('grad/pla_fired_%s' % tag) for k in st)
            for a, b in zip(views0, views):
                assert torch.equal(a, b)
            continue
        n_fired = 0
        for grp in fr.group_names:
            d, ff, gg = before[grp]
            d2, _, gg2 = after[grp]
            fired = st['grad/pla_fired_%s/%s' % (tag, grp)]
            assert fired == float(d < 0)
            n_fired += int(fired)
            if d < 0:
                assert abs(d2) < 1e-6 * (ff * gg) ** 0.5           # exactly orthogonal after the projection
                assert gg2 == pytest.approx(gg - d * d / ff, rel=1e-5)  # ||g'||^2 = ||g||^2 (1 - cos^2)
            else:
                assert d2 == pytest.approx(d, rel=1e-9) and gg2 == pytest.approx(gg, rel=1e-9)
                for i in fr.group_idx[grp]:
                    assert torch.equal(views0[i], views[i])
        assert st['grad/pla_fired_%s_frac' % tag] == pytest.approx(n_fired / len(fr.group_names))
        assert 0 < n_fired < len(fr.group_names)
        tot0 = sum(b[2] for b in before.values()); tot1 = sum(a[2] for a in after.values())
        assert st['grad/pla_norm_kept_%s' % tag] == pytest.approx((tot1 / tot0) ** 0.5, rel=1e-5)
    # .grad reassembled from the (aligned) halves
    for p, gx, gs in zip(run.params[:nb], gx_views, gs_views):
        torch.testing.assert_close(p.grad, gx + gs)


def test_align_leaves_a_zero_counterfactual_half_alone(tmp_path):
    run = Run(tmp_path)
    run.step(sup_only=True)  # frozen unsupervised half is exactly zero
    gx, gs = run.backbone_grads()
    st = run.frozen.align(run.grad_x_acc[:run.n_backbone], gs, run.params[:run.n_backbone])
    assert st['grad/pla_fired_s_frac'] == 0.0
    assert st['grad/pla_norm_kept_s'] != st['grad/pla_norm_kept_s']  # NaN: nothing to keep


def test_align_skips_parameters_without_a_gradient(tmp_path):
    run = Run(tmp_path)
    run.step()
    nb = run.n_backbone
    for p in run.params[:nb]:
        p.grad = None  # the --unlock-after lock: backbone out of the graph
    gs = [torch.zeros_like(g) for g in run.grad_x_acc[:nb]]
    run.frozen.align(run.grad_x_acc[:nb], gs, run.params[:nb])
    assert all(p.grad is None for p in run.params[:nb])


def test_pla_flags_env_and_validation():
    import logging
    log = logging.getLogger('t')
    args = parser.parse_args(REQUIRED)
    assert args.pla is None and args.pla_target == 'both'  # absent = off
    args = parser.parse_args(REQUIRED + ['--pla'])
    assert args.pla == 0  # bare --pla = the whole run
    check_unlock_after(args, {'lock_backbone': False})
    assert args.frozen_grad is False  # not implied: the trainer runs the pass while --pla is active on its own
    assert parser.parse_args(REQUIRED + ['--pla', '3']).pla == 3  # K epochs
    assert parser.parse_args(REQUIRED + ['--pla', '0']).pla == 0
    assert parser.parse_args(REQUIRED + ['--pla', '--sup-only']).pla == 0  # a following flag is not read as K
    with pytest.raises(ValueError):
        check_unlock_after(parser.parse_args(REQUIRED + ['--pla']), {'lock_backbone': True})
    with pytest.raises(ValueError):
        check_unlock_after(parser.parse_args(REQUIRED + ['--pla', '-1']), {'lock_backbone': False})
    with pytest.raises(ValueError):
        check_unlock_after(parser.parse_args(REQUIRED + ['--pla', '--sup-only', '--pla-target', 'unsup']), {'lock_backbone': False})
    check_unlock_after(parser.parse_args(REQUIRED + ['--pla', '--sup-only', '--pla-target', 'sup']), {'lock_backbone': False})
    args = parser.parse_args(REQUIRED)
    os.environ.update({'PLA': '2', 'PLA_TARGET': 'unsup'})
    try:
        apply_env_overrides(args, log)
    finally:
        for k in ('PLA', 'PLA_TARGET'):
            del os.environ[k]
    assert args.pla == 2 and args.pla_target == 'unsup'
    os.environ['PLA'] = 'true'  # the old on/off spelling is refused rather than misread as a count
    try:
        with pytest.raises(ValueError):
            apply_env_overrides(parser.parse_args(REQUIRED), log)
    finally:
        del os.environ['PLA']
    os.environ['PLA_TARGET'] = 'all'
    try:
        with pytest.raises(ValueError):
            apply_env_overrides(parser.parse_args(REQUIRED), log)
    finally:
        del os.environ['PLA_TARGET']


# ----------------------------------------------------------------------------
# --pla-control: matched controls of the projection
# ----------------------------------------------------------------------------

def clones(views):
    return [v.clone() for v in views]


def group_numel(fr, grp):
    return sum(fr.numels[i] for i in fr.group_idx[grp])


@pytest.mark.parametrize('noise', [0.1, 3.0])  # |cos(g, f)| ~ 0.995 and ~ 0.3
def test_random_control_matches_the_projection_except_in_direction(tmp_path, noise):
    run = Run(tmp_path)
    fr, nb = run.frozen, run.n_backbone
    gx_views, gs_views = synthetic_halves(run, noise=noise)
    gx0, gs0 = clones(gx_views), clones(gs_views)
    px, ps = clones(gx_views), clones(gs_views)
    st_pla = fr.align(px, ps, run.params[:nb], 'both', 'none')  # the projection itself, on copies
    for p, gx, gs in zip(run.params[:nb], gx_views, gs_views):
        p.grad = gx + gs  # as the trainer leaves it before align (the call above rewrote it)
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    st = fr.align(gx_views, gs_views, run.params[:nb], 'both', 'random')
    # seed pairing: the private generator paid for the random directions, not the global streams
    assert torch.equal(torch.get_rng_state(), cpu_state) and torch.equal(torch.cuda.get_rng_state(), cuda_state)
    for tag, views0, views, pviews, f_views in (('x', gx0, gx_views, px, fr.acc_x), ('s', gs0, gs_views, ps, fr.acc_s)):
        before, after, pafter = group_dots(fr, views0, f_views), group_dots(fr, views, f_views), group_dots(fr, pviews, f_views)
        for grp in fr.group_names:
            d, ff, gg = before[grp]
            d2, _, gg2 = after[grp]
            fired = st['grad/pla_fired_%s/%s' % (tag, grp)]
            assert fired == st_pla['grad/pla_fired_%s/%s' % (tag, grp)] == float(d < 0)  # the same firing set
            key = 'grad/pla_ctrl_cos_gr/%s/%s' % (tag, grp)
            if not fired:
                assert key not in st
                for i in fr.group_idx[grp]:
                    assert torch.equal(views0[i], views[i])
                continue
            # the projection's final norm, exactly ...
            assert gg2 == pytest.approx(pafter[grp][2], rel=1e-5) and gg2 == pytest.approx(gg - d * d / ff, rel=1e-5)
            # ... but the direction is uninformative: still opposing f, where the projection is orthogonal to it
            assert d2 < 0 and abs(pafter[grp][0]) < 1e-6 * (ff * gg) ** 0.5
            # displacement length ||g|| |cos(g, f)|: g' = s (g - m r) with r nearly orthogonal to g gives
            # cos(g', g) = 1 / sqrt(1 + cos^2(g, f)) up to O(1 / sqrt(numel)), whatever the rescale s
            c2 = d * d / (ff * gg)
            dot = sum(float((views0[i].double() * views[i].double()).sum()) for i in fr.group_idx[grp])
            assert dot / (gg * gg2) ** 0.5 == pytest.approx(1 / (1 + c2) ** 0.5, abs=min(0.3, 10 / group_numel(fr, grp) ** 0.5))
            # the diagnostic: a random direction in numel dimensions
            assert abs(st[key]) < 6 / group_numel(fr, grp) ** 0.5
        assert st['grad/pla_fired_%s_frac' % tag] == st_pla['grad/pla_fired_%s_frac' % tag]
        assert st['grad/pla_norm_kept_%s' % tag] == pytest.approx(st_pla['grad/pla_norm_kept_%s' % tag], rel=1e-5)
    for p, gx, gs in zip(run.params[:nb], gx_views, gs_views):
        torch.testing.assert_close(p.grad, gx + gs)


def test_flip_control_removes_the_aligned_component(tmp_path):
    run = Run(tmp_path)
    fr, nb = run.frozen, run.n_backbone
    gx_views, gs_views = synthetic_halves(run)
    gx0, gs0 = clones(gx_views), clones(gs_views)
    st = fr.align(gx_views, gs_views, run.params[:nb], 'both', 'flip')
    assert not any(k.startswith('grad/pla_ctrl_cos_gr/') for k in st)
    for tag, views0, views, f_views in (('x', gx0, gx_views, fr.acc_x), ('s', gs0, gs_views, fr.acc_s)):
        before, after = group_dots(fr, views0, f_views), group_dots(fr, views, f_views)
        for grp in fr.group_names:
            d, ff, gg = before[grp]
            d2, _, gg2 = after[grp]
            assert st['grad/pla_fired_%s/%s' % (tag, grp)] == float(d > 0)  # the complementary firing set
            if d > 0:
                assert abs(d2) < 1e-6 * (ff * gg) ** 0.5 and gg2 == pytest.approx(gg - d * d / ff, rel=1e-5)
            else:
                for i in fr.group_idx[grp]:
                    assert torch.equal(views0[i], views[i])
    for p, gx, gs in zip(run.params[:nb], gx_views, gs_views):
        torch.testing.assert_close(p.grad, gx + gs)


def test_controls_at_the_checkpoint_random_is_idle_and_flip_fires_everywhere(tmp_path):
    run = Run(tmp_path)
    run.step()
    nb = run.n_backbone
    gx, gs = run.backbone_grads()
    st = run.frozen.align(clones(gx), clones(gs), run.params[:nb], 'both', 'random')
    assert st['grad/pla_fired_x_frac'] == 0.0 and st['grad/pla_fired_s_frac'] == 0.0  # cos = 1: nothing opposes
    st = run.frozen.align(clones(gx), clones(gs), run.params[:nb], 'both', 'flip')
    assert st['grad/pla_fired_x_frac'] == 1.0 and st['grad/pla_fired_s_frac'] == 1.0  # ... and everything is aligned,
    assert st['grad/pla_norm_kept_x'] < 0.05 and st['grad/pla_norm_kept_s'] < 0.05  # so flip leaves ~no gradient


def test_random_control_never_moves_a_tensor_the_counterfactual_never_reaches(tmp_path):
    run = Run(tmp_path)
    fr, nb = run.frozen, run.n_backbone
    ckpt = os.path.join(str(tmp_path), 'bb.pth')
    gx_views, gs_views = synthetic_halves(run)
    mt = fr.names.index('mask_token')  # in the embed group, reached by no loss: f = 0 and g = 0 there in a real step
    assert fr.groups[mt] == 'embed' and fr.group_idx['embed'][0] != mt  # not alone in its group
    ref = FrozenGrad(run.model, 'dinov2_small', ckpt, control_seed=0)  # twin with f nonzero on mask_token
    ref.flat_x.copy_(fr.flat_x)
    ref.flat_s.copy_(fr.flat_s)
    rx, rs_ = clones(gx_views), clones(gs_views)
    ref.align(rx, rs_, run.params[:nb], 'both', 'random')
    fr.acc_x[mt].zero_(); fr.acc_s[mt].zero_()
    gx_views[mt].zero_(); gs_views[mt].zero_()
    gx0, gs0 = clones(gx_views), clones(gs_views)
    before_x, before_s = group_dots(fr, gx0, fr.acc_x), group_dots(fr, gs0, fr.acc_s)
    st = fr.align(gx_views, gs_views, run.params[:nb], 'both', 'random')
    assert st['grad/pla_fired_x/embed'] + st['grad/pla_fired_s/embed'] == 1.0  # the sign pattern: one half fires
    assert torch.equal(gx_views[mt], torch.zeros_like(gx_views[mt])) and torch.equal(gs_views[mt], torch.zeros_like(gs_views[mt]))
    for tag, before, views, f_views in (('x', before_x, gx_views, fr.acc_x), ('s', before_s, gs_views, fr.acc_s)):
        if st['grad/pla_fired_%s/embed' % tag]:
            d, ff, gg = before['embed']
            assert group_dots(fr, views, f_views)['embed'][2] == pytest.approx(gg - d * d / ff, rel=1e-5)  # norm still exact
            assert any(not torch.equal(views[i], (gx0 if tag == 'x' else gs0)[i]) for i in fr.group_idx['embed'] if i != mt)
    # the draw happened anyway: the generator advanced exactly as in the twin whose f reaches mask_token
    assert torch.equal(fr.gen.get_state(), ref.gen.get_state())
    assert not torch.equal(rx[mt], torch.zeros_like(rx[mt])) or not torch.equal(rs_[mt], torch.zeros_like(rs_[mt]))


def test_control_seed_reproduces_and_distinguishes_the_random_directions(tmp_path):
    run = Run(tmp_path)
    fr, nb = run.frozen, run.n_backbone
    gx_views, gs_views = synthetic_halves(run)
    ckpt = os.path.join(str(tmp_path), 'bb.pth')
    assert fr.gen.initial_seed() == 0  # the default control seed
    outs = []
    for seed in (0, 0, 1):
        twin = FrozenGrad(run.model, 'dinov2_small', ckpt, control_seed=seed)
        twin.flat_x.copy_(fr.flat_x)
        twin.flat_s.copy_(fr.flat_s)
        gx, gs = clones(gx_views), clones(gs_views)
        st = twin.align(gx, gs, run.params[:nb], 'both', 'random')
        outs.append((gx + gs, st))
    (a, sa), (b, sb), (c, sc) = outs
    assert all(torch.equal(u, v) for u, v in zip(a, b)) and sa == sb  # same seed: bitwise the same run
    assert not all(torch.equal(u, v) for u, v in zip(a, c))  # another seed: other directions ...
    assert list(sa) == list(sc) and sa != sc  # ... on the same firing set (same keys), other cos(g, r)


def test_pla_control_flags_env_and_validation():
    import logging
    log = logging.getLogger('t')
    args = parser.parse_args(REQUIRED)
    assert args.pla_control == 'none' and args.pla_control_seed == 0
    args = parser.parse_args(REQUIRED + ['--pla', '--pla-control', 'random', '--pla-control-seed', '3'])
    assert args.pla_control == 'random' and args.pla_control_seed == 3
    check_unlock_after(args, {'lock_backbone': False})
    check_unlock_after(parser.parse_args(REQUIRED + ['--pla', '5', '--pla-control', 'flip']), {'lock_backbone': False})
    with pytest.raises(ValueError):  # a control of --pla: meaningless without it
        check_unlock_after(parser.parse_args(REQUIRED + ['--pla-control', 'random']), {'lock_backbone': False})
    with pytest.raises(SystemExit):  # not a choice
        parser.parse_args(REQUIRED + ['--pla', '--pla-control', 'shuffle'])
    args = parser.parse_args(REQUIRED + ['--pla'])
    os.environ.update({'PLA_CONTROL': 'flip', 'PLA_CONTROL_SEED': '11'})
    try:
        apply_env_overrides(args, log)
    finally:
        for k in ('PLA_CONTROL', 'PLA_CONTROL_SEED'):
            del os.environ[k]
    assert args.pla_control == 'flip' and args.pla_control_seed == 11
    os.environ['PLA_CONTROL'] = 'shuffle'
    try:
        with pytest.raises(ValueError):
            apply_env_overrides(parser.parse_args(REQUIRED), log)
    finally:
        del os.environ['PLA_CONTROL']
