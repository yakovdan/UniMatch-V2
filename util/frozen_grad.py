r"""Counterfactual gradient of the frozen pretrained backbone (``unimatch_v2_1gpu.py --frozen-grad``).

At every micro-batch the student's losses are recomputed with the pretrained DINOv2 weights
$\theta_0$ in place of the student's backbone and everything else held fixed: the current head
$\theta_H$, the same images, the same targets (the labels; the teacher's CutMixed pseudo-labels,
confidence mask and, under ``--abstention``, the referee labels) and the same Complementary
Dropout mask. The exact gradient with respect to the frozen weights only,

$$g_0 = \nabla_{\theta_0}\, L\big(H_{\theta_H}(B_{\theta_0}(x)),\, y\big),$$

is what a locked-backbone run would compute at this step if its head were at $\theta_H$. It is
never applied. It is accumulated over the step's micro-batches with the applied gradient's
$1/(2\cdot\mathrm{accum})$ scaling, supervised and unsupervised parts separately, and compared at
step end against the gradient the trainable backbone actually received (``grad_x_acc`` /
``grad_s_acc`` before any GGR surgery): per parameter group -- the transformer blocks, the
embeddings (cls/pos/mask tokens and the patch projection) and the final norm -- and over the
whole backbone, cosine and norm ratio for the total, supervised and unsupervised gradients.

Costs one extra forward and backward per micro-batch (frozen backbone + head; no teacher). Draws
no RNG: DINOv2 has no stochastic layer at drop rate 0 and the dropout mask is reused, so the data
stream and every seed pairing are untouched. The head's own gradient under the frozen features is
never materialised (``autograd.grad`` asks for the frozen parameters only).

``--pla`` (pairwise layer-wise alignment) turns the diagnostic into a rule: see ``FrozenGrad.align``.
"""
import math
import re

import torch

from model.backbone.dinov2 import DINOv2

EMBED_PARAMS = ('cls_token', 'pos_embed', 'mask_token', 'patch_embed')


def group_of(name):
    """Parameter group label of a DINOv2 parameter name: 'block00'..'block11', 'embed' or 'norm'."""
    m = re.match(r'blocks\.(\d+)\.', name)
    if m:
        return 'block%02d' % int(m.group(1))
    if name.split('.')[0] in EMBED_PARAMS:
        return 'embed'
    if name.startswith('norm.'):
        return 'norm'
    raise ValueError('unrecognised backbone parameter %s' % name)


def _cos(dot, n2a, n2b):
    return dot / math.sqrt(n2a * n2b) if n2a > 0 and n2b > 0 else float('nan')


def _ratio(n2a, n2b):
    return math.sqrt(n2a / n2b) if n2b > 0 else float('nan')


class FrozenGrad:
    """Frozen pretrained backbone plus the flat accumulators of its counterfactual gradient.

    ``micro_batch`` runs the extra pass and accumulates; ``step_stats`` reduces against the
    trainable backbone's gradient and returns the W&B ``grad/frozen_*`` dict; ``zero_`` resets
    at step start. Parameter order is asserted equal to the student backbone's, so the per-
    parameter views line up with ``grad_x_acc[:n_backbone]`` when the backbone is trainable.
    """

    def __init__(self, model, backbone_name, ckpt_path, device='cuda'):
        self.bb = DINOv2(model_name=backbone_name.split('_')[-1])
        self.bb.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.bb = self.bb.to(device).eval()  # no BN / dropout in DINOv2 at drop rate 0: eval == train
        self.idx = model.intermediate_layer_idx[model.encoder_size]
        self.names = [n for n, _ in self.bb.named_parameters()]
        student_names = [n for n, _ in model.backbone.named_parameters()]
        assert self.names == student_names, 'frozen copy and student backbone differ in parameter order'
        self.params = [p for _, p in self.bb.named_parameters()]
        for p in self.params:
            p.requires_grad_(True)  # so autograd can reach them; no optimizer ever sees them
        self.groups = [group_of(n) for n in self.names]
        self.group_names = sorted(set(self.groups), key=lambda g: (g == 'norm', g != 'embed', g))
        self.group_idx = {g: [i for i, gi in enumerate(self.groups) if gi == g] for g in self.group_names}
        self.numels = [p.numel() for p in self.params]
        D = sum(self.numels)
        self.flat_x = torch.zeros(D, device=device)
        self.flat_s = torch.zeros(D, device=device)
        self.acc_x, self.acc_s = self._views(self.flat_x), self._views(self.flat_s)

    def _views(self, flat):
        views, off = [], 0
        for p, n in zip(self.params, self.numels):
            views.append(flat[off:off + n].view_as(p))
            off += n
        return views

    def zero_(self):
        self.flat_x.zero_()
        self.flat_s.zero_()

    def _accumulate(self, loss, views, scale):
        grads = torch.autograd.grad(loss * scale, self.params, allow_unused=True)
        for v, g in zip(views, grads):
            if g is not None:  # mask_token is reached by no loss, like in the student
                v.add_(g)

    def _forward(self, model, x, dropout_mask, bf16):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            feats = self.bb.get_intermediate_layers(x, self.idx)
            return model.forward_head(feats, patch_h, patch_w, dropout_mask)

    def micro_batch(self, model, img_x, loss_x_fn, img_u=None, loss_u_fn=None, dropout_mask=None,
                    accum=1, bf16=False):
        """One micro-batch's counterfactual pass. ``loss_x_fn(pred)`` and ``loss_u_fn(pred_s1,
        pred_s2)`` are the student's own loss computations on given logits (targets bound and
        autocast handled by the caller, exactly as in its own step); ``img_u`` is the concatenated
        strong views and ``dropout_mask`` the mask the student's forward on them used. Returns the
        two detached losses (unsupervised NaN when skipped)."""
        scale = 1.0 / (2.0 * accum)
        pred_x = self._forward(model, img_x, None, bf16)
        loss_x = loss_x_fn(pred_x)
        self._accumulate(loss_x, self.acc_x, scale)
        loss_u = torch.full((), float('nan'), device=img_x.device)
        if img_u is not None:
            pred_u = self._forward(model, img_u, dropout_mask, bf16)
            loss_u = loss_u_fn(*pred_u.chunk(2))
            self._accumulate(loss_u, self.acc_s, scale)
        return loss_x.detach(), loss_u.detach()

    @torch.no_grad()
    def step_stats(self, grad_x_views=(), grad_s_views=()):
        """Per-group and whole-backbone comparison of the accumulated counterfactual gradient with
        the trainable backbone's (``grad_x_acc[:n_backbone]`` / ``grad_s_acc[:n_backbone]``, in
        the same order; empty under a locked backbone, whose gradient is then all zero and every
        cosine NaN). Keys: ``grad/frozen_{cos,cos_x,cos_s,norm,norm_x,norm_s,norm_ratio}_all`` and
        per group ``grad/frozen_{cos,cos_x,cos_s,norm_ratio}/<group>``."""
        n = len(self.params)
        acc = torch.zeros(n, 9, dtype=torch.float64, device=self.flat_x.device)
        for i in range(n):
            fx, fs = self.acc_x[i], self.acc_s[i]
            ft = fx + fs
            gx = grad_x_views[i] if i < len(grad_x_views) else torch.zeros_like(fx)
            gs = grad_s_views[i] if i < len(grad_s_views) else torch.zeros_like(fs)
            gt = gx + gs
            acc[i] = torch.stack([
                (gt * ft).sum(dtype=torch.float64), (gt * gt).sum(dtype=torch.float64), (ft * ft).sum(dtype=torch.float64),
                (gx * fx).sum(dtype=torch.float64), (gx * gx).sum(dtype=torch.float64), (fx * fx).sum(dtype=torch.float64),
                (gs * fs).sum(dtype=torch.float64), (gs * gs).sum(dtype=torch.float64), (fs * fs).sum(dtype=torch.float64)])
        acc = acc.cpu().numpy()
        out = {}

        def emit(prefix, row):
            dt, gt2, ft2, dx, gx2, fx2, ds, gs2, fs2 = row.tolist()
            out['grad/frozen_cos' + prefix] = _cos(dt, gt2, ft2)
            out['grad/frozen_cos_x' + prefix] = _cos(dx, gx2, fx2)
            out['grad/frozen_cos_s' + prefix] = _cos(ds, gs2, fs2)
            out['grad/frozen_norm_ratio' + prefix] = _ratio(ft2, gt2)
            return ft2, fx2, fs2

        ft2, fx2, fs2 = emit('_all', acc.sum(0))
        out['grad/frozen_norm_all'] = math.sqrt(ft2)
        out['grad/frozen_norm_x_all'] = math.sqrt(fx2)
        out['grad/frozen_norm_s_all'] = math.sqrt(fs2)
        for g in self.group_names:
            emit('/' + g, acc[self.group_idx[g]].sum(0))
        return out

    @torch.no_grad()
    def align(self, grad_x_views, grad_s_views, params, target='both'):
        r"""Pairwise layer-wise alignment (``--pla``). For every parameter group $\ell$ and every
        half $h \in \{x, s\}$ selected by ``target`` (supervised against the frozen supervised
        gradient, unsupervised against the frozen unsupervised one), if
        $\langle g_\ell^h, f_\ell^h \rangle < 0$ the half is replaced by its projection onto the
        plane orthogonal to the counterfactual,

        $$g_\ell^h \leftarrow g_\ell^h - \frac{\langle g_\ell^h, f_\ell^h\rangle}{\|f_\ell^h\|^2}\, f_\ell^h,$$

        one-way PCGrad (GGR's vlr rule) with the frozen gradient as the anchor that is never
        applied. Post-projection the group cosine is exactly 0 and the norm shrinks to
        $\|g\|\sqrt{1-\cos^2}$; groups already non-opposing, and halves whose counterfactual is
        zero (e.g. the unsupervised one while the mask ratio is 0), are untouched. Edits the flat
        views in place and rewrites ``.grad = g_x + g_s`` on the backbone ``params`` of the fired
        groups only (skipping those without a gradient, i.e. under the --unlock-after lock), so a
        step where nothing fires is bitwise identical to the same step without the flag. Returns the W&B
        ``grad/pla_*`` dict: per half and group ``fired`` (0/1), the fraction of groups fired,
        and the kept norm $\|g'\|/\|g\|$ of the half over the whole backbone."""
        n = len(self.params)
        assert len(grad_x_views) == n == len(grad_s_views) == len(params)
        halves = [(t, gv, fv) for t, gv, fv in (('x', grad_x_views, self.acc_x), ('s', grad_s_views, self.acc_s))
                  if target == 'both' or target == {'x': 'sup', 's': 'unsup'}[t]]
        out, touched = {}, set()  # parameters of a fired group in either half: only those get .grad rewritten
        for tag, g_views, f_views in halves:
            fired, before, after = 0, 0.0, 0.0
            for grp in self.group_names:
                idx = self.group_idx[grp]
                d = sum((g_views[i] * f_views[i]).sum(dtype=torch.float64) for i in idx)
                ff = sum((f_views[i] * f_views[i]).sum(dtype=torch.float64) for i in idx)
                gg = sum((g_views[i] * g_views[i]).sum(dtype=torch.float64) for i in idx)
                d, ff, gg = d.item(), ff.item(), gg.item()
                fire = d < 0.0 and ff > 0.0
                if fire:
                    coef = d / ff
                    for i in idx:
                        g_views[i].sub_(coef * f_views[i])
                    touched.update(idx)
                    gg_after = max(gg - d * d / ff, 0.0)
                else:
                    gg_after = gg
                fired += int(fire)
                before += gg
                after += gg_after
                out['grad/pla_fired_%s/%s' % (tag, grp)] = float(fire)
            out['grad/pla_fired_%s_frac' % tag] = fired / len(self.group_names)
            out['grad/pla_norm_kept_%s' % tag] = math.sqrt(after / before) if before > 0 else float('nan')
        # rewrite only what changed, so a step where nothing fired is bitwise the step without --pla
        # (gs was formed as .grad - gx, and the round trip is not exact in floating point)
        for i in sorted(touched):
            if params[i].grad is not None:
                params[i].grad.copy_(grad_x_views[i] + grad_s_views[i])
        return out
