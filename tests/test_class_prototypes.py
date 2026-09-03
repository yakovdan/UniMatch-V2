"""Tests for update_class_prototypes: the Adam-style, bias-corrected per-class EMA and
the out-of-sample cosine it returns alongside each step's observations."""
import math

import pytest
import torch

from unimatch_v2_1gpu import update_class_prototypes

NCLASS, D = 4, 6


def fresh(beta_dtype=torch.float32):
    proto = torch.zeros(NCLASS, D, dtype=beta_dtype)
    acc = torch.zeros(NCLASS, D, dtype=beta_dtype)
    cnt = torch.zeros(NCLASS)
    upd = torch.zeros(NCLASS, dtype=torch.long)
    return proto, acc, cnt, upd


def observe(acc, cnt, c, g, n_micro=1):
    """Class c seen in `n_micro` micro-batches, each contributing gradient g."""
    acc[c] += n_micro * g
    cnt[c] += n_micro


def test_no_observation_returns_nones_and_leaves_state():
    proto, acc, cnt, upd = fresh()
    proto[1] = 1.0
    out = update_class_prototypes(proto, acc, cnt, upd, beta=0.9)
    assert out == (None, None, None)
    assert torch.equal(proto[1], torch.ones(D)) and upd.sum() == 0


def test_first_observation_replaces_zero_row_and_has_nan_cosine():
    proto, acc, cnt, upd = fresh()
    g = torch.arange(D, dtype=torch.float32)
    observe(acc, cnt, 2, g, n_micro=3)
    obs, obs_cls, cos_prev = update_class_prototypes(proto, acc, cnt, upd, beta=0.99)
    assert obs_cls.tolist() == [2]
    torch.testing.assert_close(obs[0], g)          # mean over the 3 micro-batches
    torch.testing.assert_close(proto[2], g)        # r_1 = 0: bias-corrected estimate IS the observation
    assert upd[2] == 1 and math.isnan(cos_prev[0].item())


@pytest.mark.parametrize('beta', [0.9, 0.99])
def test_cosine_is_against_the_prototype_before_the_update(beta):
    proto, acc, cnt, upd = fresh()
    torch.manual_seed(0)
    g1, g2 = torch.randn(D), torch.randn(D)
    observe(acc, cnt, 0, g1)
    update_class_prototypes(proto, acc, cnt, upd, beta=beta)
    acc.zero_(), cnt.zero_()
    prev = proto[0].clone()
    observe(acc, cnt, 0, g2)
    obs, obs_cls, cos_prev = update_class_prototypes(proto, acc, cnt, upd, beta=beta)
    expected = torch.dot(g2, prev) / (g2.norm() * prev.norm())
    torch.testing.assert_close(cos_prev[0], expected)
    assert not torch.allclose(proto[0], prev)      # the update itself still happened
    # second update: r_2 = beta(1-beta)/(1-beta^2) = beta/(1+beta)
    r2 = beta / (1.0 + beta)
    torch.testing.assert_close(proto[0], r2 * g1 + (1.0 - r2) * g2)


def test_per_class_bookkeeping_is_independent():
    proto, acc, cnt, upd = fresh()
    g = torch.ones(D)
    observe(acc, cnt, 0, g)
    observe(acc, cnt, 3, 2 * g)
    _, obs_cls, cos_prev = update_class_prototypes(proto, acc, cnt, upd, beta=0.9)
    assert obs_cls.tolist() == [0, 3] and all(math.isnan(v) for v in cos_prev.tolist())
    acc.zero_(), cnt.zero_()
    observe(acc, cnt, 3, -g)                        # only class 3 this step, opposite direction
    _, obs_cls, cos_prev = update_class_prototypes(proto, acc, cnt, upd, beta=0.9)
    assert obs_cls.tolist() == [3]
    torch.testing.assert_close(cos_prev[0], torch.tensor(-1.0))
    assert upd.tolist() == [1, 0, 0, 2]
    torch.testing.assert_close(proto[0], g)         # untouched: frozen, not decayed
