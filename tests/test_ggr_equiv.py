"""Bitwise equivalence of ``util/ggr.rectify`` against the original inline GGR block.

The reference is ``tests/fixtures/ggr_block_8a2eabd.txt``: the block exactly as it
stood in the training loop of ``unimatch_v2_1gpu.py`` at commit 8a2eabd. It is
exec'd in a namespace of synthetic training-loop variables; the extracted
function runs on clones of the identical inputs. Every mutated buffer (the flat
unsupervised gradient, each ``p.grad``, the basis ``U``, the prototypes), the
returned ring-buffer counters, the full stats dict and the logged lines must
match exactly -- ``torch.equal``, not ``allclose``.

Each scenario runs a multi-step sequence so the recent-anchor ring buffer is
exercised empty, partially filled, full and through re-orthogonalisation, and
the periodic (``iters % N == 0``) logging paths fire. Class-anchor scenarios
cover no class seen, some seen, background only, a linearly dependent
prototype and an exact-zero prototype (which trips the QR ``keep`` filter).

Run with ``pytest tests/`` (CUDA scenarios skip when no GPU is visible), or as a
script, ``python tests/test_ggr_equiv.py``, for a per-scenario table.
"""
import argparse
import logging
import math
import os
import sys
from typing import Any

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from util import ggr  # noqa: E402
from util.classes import CLASSES as REAL_CLASSES  # noqa: E402
from util.csr_cone import cone_rectify  # noqa: E402

FIXTURE = os.path.join(ROOT, 'tests', 'fixtures', 'ggr_block_8a2eabd.txt')
ORIG_CODE = compile(open(FIXTURE).read(), FIXTURE, 'exec')
# the constants the block read from the training script's module scope at 8a2eabd
ORIG_CONSTANTS = dict(GGR_MIN_RESIDUAL=1e-8, GGR_ORTH_LOG_EVERY=200, GGR_PROJ_LOG_EVERY=1)

# two size profiles: a toy one that runs everywhere in a few seconds, and one with
# the real Pascal class list and a head-sized D so the CUDA kernels are the ones
# training actually hits (cusolver QR on (D, 21), cuBLAS GEMV on a ~100k vector)
PROFILES: dict[str, dict[str, Any]] = {
    'small': dict(names=['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],
                  shapes=[(4, 3), (5,), (3, 4), (6,), (2, 2)],
                  freq=[0.5, 0.05, 0.2, 0.0, 0.1, 0.15]),
    'large': dict(names=list(REAL_CLASSES['pascal']),
                  shapes=[(64, 96), (64,), (96, 384), (96,), (384, 96), (96,), (21, 96), (21,)],
                  freq=[0.7] + [0.3 / 20 * (1 + 0.5 * math.sin(i)) for i in range(20)]),
}
N_BACKBONE = 2  # params[:2] play the backbone, the rest the head


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def make_logger(name):
    lg = logging.getLogger(name)
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.INFO)
    h = ListHandler()
    lg.addHandler(h)
    return lg, h


def flat_views(flat, params):
    views, off = [], 0
    for p in params:
        n = p.numel()
        views.append(flat[off:off + n].view_as(p))
        off += n
    return views


SEEN_PATTERNS = {
    'none': lambda n: [False] * n,
    'some': lambda n: [i % 2 == 0 for i in range(n)],
    'bg_only': lambda n: [True] + [False] * (n - 1),
    'dup': lambda n: [True] * n,   # last prototype = 3 x an earlier one (fp32 residual ~1e-6: kept)
    'zero': lambda n: [True] * n,  # one seen prototype exactly zero -> R diagonal 0 -> dropped by keep
}


class State:
    """One implementation's copy of every mutable training-loop object."""

    def __init__(self, profile, scope, mode, anchor, d, device, seed):
        prof = PROFILES[profile]
        self.names = prof['names']
        self.n_cls = len(self.names)
        self.device = device
        g = torch.Generator().manual_seed(seed)
        self.params = [torch.randn(*s, generator=g).to(device).requires_grad_(True) for s in prof['shapes']]
        for i, p in enumerate(self.params):
            p.grad = torch.randn(*p.shape, generator=g).to(device) if i != 1 else None  # one param without grad
        numels = [p.numel() for p in self.params]
        self.D_total = sum(numels)
        D_backbone = sum(numels[:N_BACKBONE])
        if scope == 'backbone':
            self.scope = (0, D_backbone, 0, N_BACKBONE)
        elif scope == 'head':
            self.scope = (D_backbone, self.D_total, N_BACKBONE, len(self.params))
        else:
            self.scope = (0, self.D_total, 0, len(self.params))
        self.D_scope = self.scope[1] - self.scope[0]
        self.flat_x = torch.zeros(self.D_total, device=device)
        self.flat_s = torch.zeros(self.D_total, device=device)
        self.grad_x_acc = flat_views(self.flat_x, self.params)
        self.grad_s_acc = flat_views(self.flat_s, self.params)
        self.class_anchor = mode != 'none' and anchor == 'class'
        self.U, self.u_k, self.u_ptr, self.u_iters, self.ggr_scratch = None, 0, 0, None, None
        self.proto = self.proto_w = self.proto_order = None
        self.seen, self.n_seen = None, 0
        if self.class_anchor:
            self.proto = torch.zeros(self.n_cls, self.D_scope, device=device)
            freq = torch.tensor(prof['freq'])
            inv = torch.where(freq > 0, 1.0 / freq.clamp_min(1e-12), torch.zeros(()))
            self.proto_w = (inv / inv.sum()).to(device)
            self.proto_order = torch.argsort(torch.where(freq > 0, freq, torch.tensor(float('inf')))).to(device)
            if mode in ('osr', 'csr'):
                self.U = torch.zeros(self.n_cls, self.D_scope, device=device)
        elif mode in ('osr', 'csr'):
            self.U = torch.zeros(d, self.D_scope, device=device)
            self.u_iters = [None] * d
            self.ggr_scratch = torch.empty(self.D_scope, device=device)

    def load_step(self, gen, seen_pattern, step):
        """Fresh per-step inputs; identical for both implementations via the shared generator."""
        self.flat_x.copy_(torch.randn(self.D_total, generator=gen))
        self.flat_s.copy_(torch.randn(self.D_total, generator=gen))
        lo, hi = self.scope[:2]
        if step % 2 == 0:  # make conflicts likely so vlr/csr/cone actually fire on some steps
            self.flat_s[lo:hi] -= 0.7 * self.flat_x[lo:hi]
        if self.proto is not None:
            proto = torch.randn(self.n_cls, self.D_scope, generator=gen)
            seen = torch.tensor(SEEN_PATTERNS[seen_pattern](self.n_cls))
            if seen_pattern == 'dup':
                proto[-1] = proto[2] * 3.0
            proto[~seen] = 0.0
            if seen_pattern == 'zero':
                proto[3] = 0.0
            self.proto.copy_(proto)
            self.seen, self.n_seen = seen.to(self.device), int(seen.sum())
            if step % 2 == 1 and self.n_seen > 0:
                # oppose a seen prototype so the cone / csr paths fire
                c = int(seen.nonzero()[-1])
                self.flat_s[lo:hi] -= 1.5 * self.proto[c] / self.proto[c].norm().clamp_min(1e-12)
        sx = self.flat_x[lo:hi]
        ss = self.flat_s[lo:hi]
        self.scope_norm_x_sq = (sx * sx).sum(dtype=torch.float64).item()
        self.scope_norm_s_sq = (ss * ss).sum(dtype=torch.float64).item()
        self.scope_dot_xs = (sx * ss).sum(dtype=torch.float64).item()

    def kwargs(self, iters, logger):
        return dict(iters=iters, flat_x=self.flat_x, flat_s=self.flat_s,
                    scope_lo=self.scope[0], scope_hi=self.scope[1], scope_p_lo=self.scope[2], scope_p_hi=self.scope[3],
                    params=self.params, grad_x_acc=self.grad_x_acc, grad_s_acc=self.grad_s_acc,
                    scope_norm_x_sq=self.scope_norm_x_sq, scope_norm_s_sq=self.scope_norm_s_sq,
                    scope_dot_xs=self.scope_dot_xs, class_anchor=self.class_anchor, seen=self.seen, n_seen=self.n_seen,
                    proto=self.proto, proto_w=self.proto_w, proto_order=self.proto_order,
                    U=self.U, u_k=self.u_k, u_ptr=self.u_ptr, u_iters=self.u_iters, ggr_scratch=self.ggr_scratch,
                    logger=logger)


def run_orig(st, args, iters, logger):
    ns: dict[str, Any] = dict(torch=torch, CLASSES={'toy': st.names}, cfg={'dataset': 'toy'},
                              cone_rectify=cone_rectify, args=args, **ORIG_CONSTANTS, **st.kwargs(iters, logger))
    exec(ORIG_CODE, ns)
    st.u_k, st.u_ptr = ns['u_k'], ns['u_ptr']
    return ns['ggr_stats']


def run_new(st, args, iters, logger):
    stats, st.u_k, st.u_ptr = ggr.rectify(args, class_names=st.names, **st.kwargs(iters, logger))
    return stats


def same_scalar(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def check(tag, a, b, sa, sb, ha, hb):
    assert set(sa) == set(sb), (tag, set(sa) ^ set(sb))
    for k in sa:
        assert same_scalar(sa[k], sb[k]), (tag, k, sa[k], sb[k])
    assert torch.equal(a.flat_s, b.flat_s), (tag, 'flat_s')
    assert torch.equal(a.flat_x, b.flat_x), (tag, 'flat_x')
    for i, (p, q) in enumerate(zip(a.params, b.params)):
        assert (p.grad is None) == (q.grad is None), (tag, i)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), (tag, 'grad', i)
    for name in ('U', 'proto'):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), (tag, name)
        if x is not None:
            assert torch.equal(x, y), (tag, name)
    assert a.u_k == b.u_k and a.u_ptr == b.u_ptr and a.u_iters == b.u_iters, (tag, a.u_k, b.u_k, a.u_ptr, b.u_ptr)
    assert ha.lines == hb.lines, (tag, ha.lines, hb.lines)


def run_scenario(mode, anchor, scope, seen_pattern, *, profile='small', device='cpu',
                 rescale=0.0, reorth=2, d=3, steps=7, iters0=196, seed=0) -> dict[str, Any]:
    """Run one multi-step scenario through both implementations; returns a summary dict."""
    tag = 'ggr=%s anchor=%s scope=%s seen=%s rescale=%g reorth=%d d=%d %s/%s' % (
        mode, anchor, scope, seen_pattern, rescale, reorth, d, profile, device)
    args = argparse.Namespace(ggr=mode, ggr_cone_rescale=rescale, ggr_reorth_every=reorth)
    a = State(profile, scope, mode, anchor, d, device, seed)
    b = State(profile, scope, mode, anchor, d, device, seed)
    la, ha = make_logger('ggr_equiv.orig')
    lb, hb = make_logger('ggr_equiv.new')
    fired, keys = 0, set()
    for step in range(steps):
        a.load_step(torch.Generator().manual_seed(1000 * seed + step), seen_pattern, step)
        b.load_step(torch.Generator().manual_seed(1000 * seed + step), seen_pattern, step)
        assert torch.equal(a.flat_s, b.flat_s) and torch.equal(a.flat_x, b.flat_x)
        iters = iters0 + step
        sa = run_orig(a, args, iters, la)
        sb = run_new(b, args, iters, lb)
        check('%s step=%d iters=%d' % (tag, step, iters), a, b, sa, sb, ha, hb)
        fired += int(sa.get('grad/ggr_fired', 0.0) > 0)
        keys |= set(sa)
    if mode == 'none':
        assert keys == set(), keys
    return dict(tag=tag, fired=fired, steps=steps, keys=keys, u_k=a.u_k, log_lines=len(ha.lines))


def scenarios():
    """(id, kwargs) for every code path; the periodic paths fire at iters 196..202."""
    out = []
    for scope in ('head', 'all', 'backbone'):
        out.append(('none-%s' % scope, dict(mode='none', anchor='recent', scope=scope, seen_pattern='none')))
        out.append(('vlr-recent-%s' % scope, dict(mode='vlr', anchor='recent', scope=scope, seen_pattern='none')))
        for mode in ('osr', 'csr'):
            out.append(('%s-recent-%s-ring' % (mode, scope),
                        dict(mode=mode, anchor='recent', scope=scope, seen_pattern='none')))  # d=3 fills, reorth every 2
            out.append(('%s-recent-%s-noreorth' % (mode, scope),
                        dict(mode=mode, anchor='recent', scope=scope, seen_pattern='none', reorth=0, d=8)))  # never fills
    for seen in SEEN_PATTERNS:
        out.append(('vlr-class-%s' % seen, dict(mode='vlr', anchor='class', scope='head', seen_pattern=seen)))
        out.append(('osr-class-%s' % seen, dict(mode='osr', anchor='class', scope='head', seen_pattern=seen)))
        out.append(('csr-class-%s' % seen, dict(mode='csr', anchor='class', scope='all', seen_pattern=seen)))
        out.append(('cone-%s' % seen, dict(mode='cone', anchor='class', scope='head', seen_pattern=seen)))
        out.append(('cone-%s-rescale2' % seen, dict(mode='cone', anchor='class', scope='head', seen_pattern=seen, rescale=2.0)))
        out.append(('cone-%s-all-rescale1.2' % seen, dict(mode='cone', anchor='class', scope='all', seen_pattern=seen, rescale=1.2)))
    return out


SCENARIOS = scenarios()
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('profile', ['small', 'large'])
@pytest.mark.parametrize('kw', [s[1] for s in SCENARIOS], ids=[s[0] for s in SCENARIOS])
def test_rectify_matches_inline_block(kw, profile, device):
    run_scenario(profile=profile, device=device, **kw)


if __name__ == '__main__':
    all_keys = set()
    for device in DEVICES:
        for profile in ('small', 'large'):
            for _, kw in SCENARIOS:
                r = run_scenario(profile=profile, device=device, **kw)
                all_keys |= r['keys']
                print('ok  %-88s fired %d/%d, %2d stat keys, u_k=%d, %d log lines' % (
                    r['tag'], r['fired'], r['steps'], len(r['keys']), r['u_k'], r['log_lines']))
    print('\nall %d scenarios x %d profiles x %s bitwise-equal; %d distinct stat keys' % (
        len(SCENARIOS), 2, '/'.join(DEVICES), len(all_keys)))
