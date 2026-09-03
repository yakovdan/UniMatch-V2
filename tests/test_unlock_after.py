"""Tests for --unlock-after: while ``DPT.backbone_frozen`` is set, the backbone runs under no_grad,
its parameters receive no gradient, and the trainer's AdamW therefore leaves them untouched
(no update, no weight decay, no state). Also the argument plumbing: env override and validation."""
import logging

import pytest
import torch
from torch import nn

from model.semseg.dpt import DPT
from unimatch_v2_1gpu import apply_env_overrides, check_unlock_after, parser
from util.optim import AdamW

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')

SMALL = dict(encoder_size='small', features=64, out_channels=[48, 96, 192, 384], nclass=21)
REQUIRED = ['--config', 'c.yaml', '--labeled-id-path', 'l.txt', '--unlabeled-id-path', 'u.txt', '--save-path', 's']


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return DPT(**SMALL).cuda()


@needs_cuda
def test_frozen_backbone_receives_no_gradient(model):
    model.train()
    x = torch.randn(2, 3, 112, 112, device='cuda')  # 8x8 patches
    def grad_bearing(module):
        return {n for n, p in module.named_parameters() if p.grad is not None}

    model.backbone_frozen = True
    model.zero_grad(set_to_none=True)
    model(x).float().mean().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    head_frozen = grad_bearing(model.head)
    assert model.head.scratch.output_conv[-1].weight.grad is not None

    model.backbone_frozen = False
    model.zero_grad(set_to_none=True)
    model(x).float().mean().backward()
    # the head trains identically either way (the DPT head has a dead branch,
    # refinenet4.resConfUnit1, that never receives a gradient in any mode)
    assert grad_bearing(model.head) == head_frozen
    assert len(head_frozen) >= len(list(model.head.parameters())) - 4
    # unfrozen, the trainable path reaches the transformer weights; only the unused mask
    # token stays None
    assert {n for n, p in model.backbone.named_parameters() if p.grad is None} == {'mask_token'}


@needs_cuda
def test_flag_does_not_change_the_forward(model):
    x = torch.randn(2, 3, 112, 112, device='cuda')
    model.eval()
    with torch.no_grad():
        model.backbone_frozen = True; frozen = model(x)
        model.backbone_frozen = False; free = model(x)
    torch.testing.assert_close(frozen, free)
    model.train()
    model.backbone_frozen = True; frozen = model(x).detach()
    model.backbone_frozen = False; free = model(x).detach()
    torch.testing.assert_close(frozen, free, rtol=1e-4, atol=1e-4)


def test_adamw_leaves_params_without_grad_untouched():
    locked, free = nn.Parameter(torch.randn(8)), nn.Parameter(torch.randn(8))
    opt = AdamW([locked, free], lr=0.1, weight_decay=0.5)
    before = locked.detach().clone()
    free.grad = torch.ones_like(free)   # locked.grad stays None, as under --unlock-after
    opt.step()
    assert torch.equal(locked, before)  # no update AND no weight decay
    assert not torch.equal(free, free.detach().clone().zero_())  # the other one moved
    assert len(opt.state[locked]) == 0 and opt.state[free]['step'] == 1


def test_adamw_accumulate_only_warms_moments_without_moving_params():
    torch.manual_seed(0)
    beta1, beta2, lr, wd = 0.9, 0.999, 0.1, 0.5
    p = nn.Parameter(torch.randn(8)); p0 = p.detach().clone()
    opt = AdamW([p], lr=lr, betas=(beta1, beta2), weight_decay=wd)
    grads = [torch.randn(8) for _ in range(3)]
    m = torch.zeros(8); v = torch.zeros(8)
    # two accumulate-only steps: state fills in, parameter stays bitwise put (no decay either)
    opt.param_groups[0]['accumulate_only'] = True
    for g in grads[:2]:
        p.grad = g.clone(); opt.step()
        m = beta1 * m + (1 - beta1) * g; v = beta2 * v + (1 - beta2) * g * g
    assert torch.equal(p, p0)
    st = opt.state[p]
    assert st['step'] == 2
    torch.testing.assert_close(st['exp_avg'], m); torch.testing.assert_close(st['exp_avg_sq'], v)
    # unlock: the next step is a normal AdamW step with bias correction at t=3, i.e. exactly what
    # an optimizer that had seen all three gradients would do
    opt.param_groups[0]['accumulate_only'] = False
    p.grad = grads[2].clone(); opt.step()
    m = beta1 * m + (1 - beta1) * grads[2]; v = beta2 * v + (1 - beta2) * grads[2] ** 2
    expected = p0 * (1 - lr * wd) - lr / (1 - beta1 ** 3) * m / (v.sqrt() / (1 - beta2 ** 3) ** 0.5 + 1e-8)
    torch.testing.assert_close(p.detach(), expected)
    # the flag round-trips through the optimizer state dict and defaults to off
    assert AdamW([nn.Parameter(torch.zeros(1))]).param_groups[0]['accumulate_only'] is False
    sd = opt.state_dict(); assert sd['param_groups'][0]['accumulate_only'] is False


def test_env_override_and_validation(monkeypatch):
    args = parser.parse_args(REQUIRED)
    assert args.unlock_after == 0
    monkeypatch.setenv('UNLOCK_AFTER', '7')
    apply_env_overrides(args, logging.getLogger('test'))
    assert args.unlock_after == 7
    check_unlock_after(args, {'lock_backbone': False})  # fine
    with pytest.raises(ValueError):
        check_unlock_after(args, {'lock_backbone': True})
    args.unlock_after = -1
    with pytest.raises(ValueError):
        check_unlock_after(args, {'lock_backbone': False})
    args.unlock_after = 0
    check_unlock_after(args, {'lock_backbone': True})  # off: any config is fine
    # --unlock-accumulate needs an active lock; env override reaches it too
    args.unlock_accumulate = True
    with pytest.raises(ValueError):
        check_unlock_after(args, {'lock_backbone': False})
    monkeypatch.setenv('UNLOCK_ACCUMULATE', '0')
    apply_env_overrides(args, logging.getLogger('test'))
    assert args.unlock_accumulate is False
