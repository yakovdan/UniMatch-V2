import argparse
from copy import deepcopy
import logging
import os
import pprint
import random
import time
from typing import NamedTuple, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from util.optim import build_adamw
from torch.utils.data import DataLoader
import wandb
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ggr import rectify as ggr_rectify
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed


# Single-GPU variant of unimatch_v2.py. The paper's recipe is 4 GPUs x batch 4;
# here one optimizer step accumulates gradients over `accum` micro-batches of
# cfg['batch_size'] so the effective batch stays 16. Gradient averaging over
# micro-batches is mathematically identical to DDP's per-rank averaging, and the
# micro-batch size stays 4 so CutMix keeps mixing within pools of 4 images and
# Complementary Dropout still sees the (s1, s2) pair in a single forward.
# LR schedule and EMA update advance once per optimizer step, matching the
# original where every iteration is an optimizer step.
# Additionally saves best_ema.pth (keyed on EMA mIoU): the paper reports the
# best EMA over epochs, but the original code only saves best.pth at the
# student's best epoch, so the released checkpoints undershoot some reported
# values (e.g. pascal/1464/base: 90.38 in best.pth vs 90.80 reported).

parser = argparse.ArgumentParser(description='UniMatch V2 single-GPU training (gradient accumulation, effective batch 16)')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--data-root', type=str, default=None, help='override data_root in the config')
parser.add_argument('--backbone', type=str, default=None, help='override the backbone in the config, e.g. dinov2_base')
parser.add_argument('--epochs', type=int, default=None, help='override epochs in the config (also compresses the poly LR schedule)')
parser.add_argument('--epoch-lim', '--epoch_lim', type=int, default=0,
                    help='stop after this many epochs while the poly LR schedule still spans the full '
                         "cfg['epochs'] (unlike --epochs, which compresses it). 0 = disabled [env: EPOCH_LIM]")
parser.add_argument('--bf16', action='store_true', help='bf16 autocast for forwards/losses (recipe deviation: paper trains fp32)')
parser.add_argument('--stop-after', type=int, default=None, help='debug: stop after N optimizer steps (smoke test)')
parser.add_argument('--lr', type=float, default=None,
                    help="override lr in the config: the backbone's base LR (config default 5e-6) and the "
                         'peak of the poly schedule; the head trains at lr * lr_multi [env: LR]')
parser.add_argument('--lr-multi', type=float, default=None,
                    help="override lr_multi in the config: the head's LR is lr * lr_multi (config default 40) "
                         '[env: LR_MULTI]')
parser.add_argument('--lock-backbone', action='store_true',
                    help="freeze the backbone (sets lock_backbone: True over the config): only the DPT head "
                         'trains; the backbone stays at the DINOv2 checkpoint in both student and teacher '
                         '[env: LOCK_BACKBONE]')
parser.add_argument('--sup-only', action='store_true',
                    help='supervised baseline: skip the teacher forward, CutMix, the strong-view student '
                         'forward and the unsupervised loss/backward. The labeled stream, loss scaling, '
                         'LR schedule and EMA are unchanged, so a run is seed-paired with the semi-supervised '
                         'run on its supervised component; grad/* logs a zero unsupervised stub [env: SUP_ONLY]')
parser.add_argument('--ggr', type=str, default='none', choices=['none', 'vlr', 'osr', 'csr', 'cone'],
                    help='rectify the unsupervised gradient against the supervised one (Chen et al., '
                         'Geometric Gradient Rectification): vector-level, orthogonal-subspace or '
                         "conic-subspace. 'cone': exact projection onto the non-conflict cone of the "
                         'per-class prototypes, background excluded (class-anchor only) [env: GGR_MODE]')
parser.add_argument('--ggr-cone-rescale', type=float, default=0.0,
                    help='cone only: rescale the rectified head gradient back toward ||g_u||, factor '
                         'min(this, ||g_u||/||P(g_u)||); 0 disables. The clamp is the apex guard -- '
                         'as P(g_u) -> 0 the unclamped factor diverges [env: GGR_CONE_RESCALE]')
parser.add_argument('--proto-beta', type=float, default=0.9,
                    help='Adam beta for the class-prototype first moment (bias-corrected; effective '
                         'horizon 1/(1-beta) updates, so 0.9 -> 10, 0.99 -> 100, 0.996 -> 250 = the '
                         'old teacher-schedule horizon). Applies to every --ggr-anchor class mode '
                         '[env: BETA]')
parser.add_argument('--ggr-anchor', type=str, default='recent', choices=['recent', 'class'],
                    help="what GGR rectifies against. 'recent': the current supervised gradient (vlr) or a "
                         "ring buffer of the last --ggr-dim supervised gradients (osr/csr). 'class': per-class "
                         'supervised gradients of the DPT head, Adam-style EMA (beta=0.9, bias-corrected); osr/csr '
                         'project onto their orthonormal basis (rarest class first), vlr against their '
                         '1/pixel-frequency weighted sum. Requires --ggr-scope head [env: GGR_ANCHOR]')
parser.add_argument('--ggr-dim', type=int, default=10,
                    help='subspace dimension d for --ggr osr/csr; ignored by vlr [env: GGR_DIM]')
parser.add_argument('--ggr-scope', type=str, default='all', choices=['backbone', 'head', 'all'],
                    help="surgery scope P, defaulting to the whole model. The paper rectifies the "
                         "encoder backbone only; 'all' is the scope this script's existing "
                         'grad/* conflict diagnostics are measured over [env: GGR_SCOPE]')
parser.add_argument('--ggr-reorth-every', type=int, default=0,
                    help='re-orthonormalise U every N steps (0 = off; the Gram-Schmidt append already '
                         're-projects, so drift is normally negligible) [env: GGR_REORTH_EVERY]')
parser.add_argument('--log-class-cos', action='store_true',
                    help='log the pairwise cosines of the per-class supervised DPT-head '
                         'gradients every optimizer step, saved per epoch as '
                         'class_cos_ep*.npz. Independent of --ggr [env: LOG_CLASS_COS]')
parser.add_argument('--seed', type=int, default=0,
                    help='seed for model init, data order and augmentations. Each epoch is reseeded '
                         'from (seed, epoch), so a resume reproduces the original run [env: SEED]')
parser.add_argument('--unlabeled-seed', type=int, default=None,
                    help='seed the unlabeled branch (unlabeled order and augmentations, CutMix boxes, '
                         'Complementary Dropout) separately from --seed, which then covers only head init '
                         'and the labeled stream. Default: same as --seed. --seed A --unlabeled-seed B '
                         'reproduces the unlabeled branch of the run seeded B exactly [env: UNLABELED_SEED]')
parser.add_argument('--init-seed', type=int, default=None,
                    help='seed the DPT head initialisation separately from --seed, which then covers only '
                         'the labeled stream (and the unlabeled one unless --unlabeled-seed is set). '
                         'Default: same as --seed. --seed A --init-seed B gives run B\'s head weights '
                         'with run A\'s data streams [env: INIT_SEED]')
parser.add_argument('--deterministic', action='store_true',
                    help='trade throughput (~4.5x slower) for reproducibility: disable cudnn.benchmark '
                         'and request deterministic kernels, warn-only. Tightens same-seed runs but is '
                         'not bitwise in either precision; under --bf16 it barely helps, since Flash '
                         "Attention's backward needs strict mode (which nll_loss2d then aborts) "
                         '[env: DETERMINISTIC]')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

EFFECTIVE_BATCH = 16  # 4 GPUs x batch 4 in the paper
GRAD_COS_CONFLICT_THRESH = -0.05  # cos(g_x, g_s) below this counts as a conflicting-gradient step
GGR_CLASS_MIN_NORM = 1e-12  # a class prototype below this has no direction to contribute
SMOKE_WARMUP_STEPS = 4  # steps of the epoch excluded from the --stop-after throughput clock (cudnn.benchmark autotunes there)
# the per-step rectification and its own constants live in util/ggr.py


def labeled_pixel_freq(cfg, id_path, save_path, logger):
    """Pixel frequency of every class over the labeled split's masks (ignore 255), cached
    next to the checkpoints. The 1/f_c weights of the class anchor come from this."""
    import json
    from PIL import Image
    cache = os.path.join(save_path, 'class_pixel_freq.json')
    if os.path.exists(cache):
        freq = json.load(open(cache))
        if len(freq) == cfg['nclass']:
            return np.array(freq)
    counts = np.zeros(cfg['nclass'], dtype=np.int64)
    for line in open(id_path).read().splitlines():
        if not line.strip():
            continue
        m = np.array(Image.open(os.path.join(cfg['data_root'], line.split()[-1])))
        vals, n = np.unique(m, return_counts=True)
        for v, k in zip(vals, n):
            if v < cfg['nclass']:
                counts[v] += k
    freq = counts / counts.sum()
    with open(cache, 'w') as f:
        json.dump(freq.tolist(), f)
    logger.info('labeled pixel frequencies (%s): %s' % (id_path, ' '.join(
        '%s=%.4f' % (c, fr) for c, fr in zip(CLASSES[cfg['dataset']], freq))))
    return freq

def boolean(raw):
    lowered = raw.strip().lower()
    if lowered in ('1', 'true', 'yes', 'on'):
        return True
    if lowered in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError

def pairwise_cos(rows, idx, n_cls):
    """(n_cls, n_cls) fp32 matrix of pairwise cosines between the per-class gradient
    `rows`, scattered to the class ids in `idx`. Classes without a row stay NaN."""

    out = np.full((n_cls, n_cls), np.nan, dtype=np.float32)
    if rows is None or rows.shape[0] == 0:
        return out
    r = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    i = idx.cpu().numpy()
    out[np.ix_(i, i)] = (r @ r.t()).float().cpu().numpy()
    return out

def accumulate_class_grads(pred_x, mask_x, anchor_params, proto_acc, proto_cnt):
    """Per-class supervised gradients of one labeled micro-batch, added in place to
    the prototype accumulators.

    For every class c present in ``mask_x`` (255 and ids beyond the accumulator are
    skipped), the plain CE averaged over the pixels of c is backpropagated to
    ``anchor_params`` -- one backward per class, with ``retain_graph`` so the caller's
    full-loss backward can still walk the same graph -- and the gradient, flattened
    and concatenated in ``anchor_params`` order (the layout of the flat gradient
    buffers), is added to ``proto_acc[c]``; ``proto_cnt[c]`` counts the observation.
    Parameters the loss does not reach (``allow_unused``) contribute nothing. Plain CE
    regardless of criterion_l, so the prototypes mean the same thing under OHEM.

    The backward runs in the precision the forward that produced ``pred_x`` recorded:
    under --bf16 the head's convs are bf16 ops in that graph, so these gradients are
    bf16-precision even though this function sits outside autocast.
    """
    n_cls = proto_acc.shape[0]
    ce_px = F.cross_entropy(pred_x, mask_x, ignore_index=255, reduction='none')
    for c in mask_x.unique().tolist():
        if c == 255 or c >= n_cls:
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

def update_class_prototypes(proto, proto_acc, proto_cnt, proto_updates, beta=0.9):
    r"""Adam-style first-moment update of the per-class gradient prototypes, in place.

    ``proto_acc`` / ``proto_cnt`` hold the current step's per-class gradient sums and
    observation counts over the labeled micro-batches; classes with count 0 (absent
    from every crop this step) are left untouched -- frozen, not decayed. Each
    observed class contributes the mean over its micro-batches,
    $\bar g_c = \mathrm{acc}_c / \mathrm{cnt}_c$, folded into the Adam first moment
    $m_c \leftarrow \beta m_c + (1-\beta) \bar g_c$ with bias correction
    $\hat m_c = m_c / (1 - \beta^{t_c})$, counting $t_c$ per class from its own
    first appearance.

    ``proto`` stores the *bias-corrected* estimate $\hat m_c$ directly, so every
    consumer (anchors, cosines, checkpoints) reads a properly normalised prototype
    with no correction of its own. That works because $\hat m$ obeys the exact
    convex recursion

        $\hat m_t = r_t \hat m_{t-1} + (1 - r_t) \bar g_t$,
        $r_t = \beta (1 - \beta^{t-1}) / (1 - \beta^t)$,

    with $r_1 = 0$ -- the first observation replaces the zero row and *is* the
    bias-corrected estimate -- and $r_t \to \beta$, i.e. an effective horizon of
    $1/(1-\beta)$ = 10 updates (much more responsive than the teacher's 250-step
    schedule this replaces).

    Mutates ``proto`` and ``proto_updates``. Does NOT reset the accumulators: the
    caller zeroes ``proto_acc`` / ``proto_cnt`` after the diagnostics that reuse the
    returned observations.

    Returns:
        ``(obs, obs_cls)`` -- the step observations ``(k, D)`` and their class ids
        ``(k,)``, or ``(None, None)`` when no class was observed this step.
    """
    upd = proto_cnt > 0
    if not upd.any():
        return None, None
    obs_cls = upd.nonzero(as_tuple=True)[0]
    obs = proto_acc[upd] / proto_cnt[upd].unsqueeze(1)
    t = proto_updates[upd].to(proto.dtype) + 1.0   # index of THIS update, t >= 1
    r = (beta * (1.0 - beta ** (t - 1.0)) / (1.0 - beta ** t)).unsqueeze(1)
    proto[upd] = r * proto[upd] + (1.0 - r) * obs
    proto_updates[upd] += 1
    return obs, obs_cls



# When set, these win over the command line, matching how USE_WANDB overrides
# wandb_mode. Launch paths that cannot easily thread arguments through to the
# process (Vast's --onstart, which replaces the image entrypoint) can set these
# instead. (env var, args attribute, type, allowed values or None)
ENV_OVERRIDES = (
    ('GGR_MODE', 'ggr', str, ('none', 'vlr', 'osr', 'csr', 'cone')),
    ('GGR_CONE_RESCALE', 'ggr_cone_rescale', float, None),
    ('BETA', 'proto_beta', float, None),
    ('GGR_ANCHOR', 'ggr_anchor', str, ('recent', 'class')),
    ('GGR_DIM', 'ggr_dim', int, None),
    ('GGR_SCOPE', 'ggr_scope', str, ('backbone', 'head', 'all')),
    ('GGR_REORTH_EVERY', 'ggr_reorth_every', int, None),
    ('EPOCH_LIM', 'epoch_lim', int, None),
    ('SUP_ONLY', 'sup_only', boolean, None),
    ('LOCK_BACKBONE', 'lock_backbone', boolean, None),
    ('LR', 'lr', float, None),
    ('LR_MULTI', 'lr_multi', float, None),
    ('SEED', 'seed', int, None),
    ('UNLABELED_SEED', 'unlabeled_seed', int, None),
    ('INIT_SEED', 'init_seed', int, None),
    ('DETERMINISTIC', 'deterministic', boolean, None),
    ('LOG_CLASS_COS', 'log_class_cos', boolean, None),
)


def seed_everything(seed):
    """Seed every RNG the training path draws from.

    The data pipeline uses three separate generators: Python's `random` and
    `numpy.random` (dataset/transform.py: crops, flips, blur, CutMix boxes),
    and torch's CPU RNG (torchvision ColorJitter, and the Binomial sample plus
    randperm behind Complementary Dropout in model/semseg/dpt.py). Model head
    init is torch CPU RNG as well; the backbone is loaded from a checkpoint.

    DataLoader workers do not need a worker_init_fn: torch's _worker_loop
    already derives per-worker seeds for `random`, `numpy` and torch from the
    loader's own generator, so seeding those generators is sufficient. Verified
    on torch 2.10; numpy seeding there dates to torch 1.9.
    """
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_determinism(args, logger):
    # cuBLAS reads this when it first initialises, so it has to be set before any
    # CUDA work; without it torch.use_deterministic_algorithms raises on matmuls.
    if args.deterministic and 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed_everything(args.seed)
    cudnn.enabled = True
    # benchmark picks kernels by timing, so it is a nondeterminism source of its own
    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = args.deterministic
    if args.deterministic:
        # warn_only: nll_loss2d forward (criterion_l, and the per-class F.cross_entropy)
        # has no deterministic CUDA kernel, so strict mode aborts the run. This flag
        # tightens same-seed runs a long way but is not bitwise in either precision.
        # Measured torch 2.10+cu130 / RTX 4090, Aug 2026, same seed, max over 20-25 steps:
        #   bf16: loss_x to ~2e-3 rel, grad_norm_x to ~0.1-0.17 rel, grad_cos_x_s to
        #     ~0.07-0.18 ABSOLUTE (the cosine sits near 0, so relative is meaningless;
        #     one step of 20 flipped across the -0.05 conflict threshold). Same order
        #     with or without this flag: SDPA dispatches to Flash Attention, whose
        #     backward is deterministic only under warn_only=False, so the flag leaves
        #     the dominant bf16 source in.
        #   fp32: ~35-100x tighter -- loss_x to ~5e-5 rel, grad_norm_x to ~2e-3 rel,
        #     cos to ~1.5e-3 abs (default mode); ~2.6e-6 at step 0 under this flag. SDPA
        #     takes the cutlass mem-eff kernel, which IS deterministic; the residual was
        #     not isolated to a single op (CE, conv, Linear, interpolate all test
        #     deterministic individually under this flag).
        # So a same-seed fp32 pair spans a much smaller perturbation than a bf16 one --
        # do not read fp32 replicates as a strong perturbation test. Cost ~2.9x
        # (env_variables.md, 40 steps; a 20-step smoke reads 4.5x because benchmark
        # autotuning is unamortised).
        # (An older comment here blamed bilinear interpolate backward -- dpt.py:165/170,
        # blocks.py:144. It is a real source in DEFAULT mode, ~4.7e-4 in fp32, but on
        # 2.10 it has a deterministic kernel that this flag selects, so it is not the
        # reason for warn_only and strict mode does not abort on it.)
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info('seed=%d deterministic=%s cudnn.benchmark=%s\n' % (
        args.seed, args.deterministic, cudnn.benchmark))


def apply_env_overrides(args, logger):
    for env, attr, cast, choices in ENV_OVERRIDES:
        raw = os.environ.get(env)
        if raw is None:
            continue
        try:
            value = cast(raw)
        except ValueError:
            raise ValueError("invalid %s='%s': expected %s" % (env, raw, cast.__name__))
        if choices is not None and value not in choices:
            raise ValueError("invalid %s='%s': expected one of %s" % (env, raw, ', '.join(choices)))
        previous = getattr(args, attr)
        setattr(args, attr, value)
        if value != previous:
            logger.info('%s=%s overrides --%s %s' % (env, value, attr.replace('_', '-'), previous))


class MicroBatch(NamedTuple):
    """One micro-batch on the GPU, ready for the student's forwards (see
    load_micro_batch). The unlabeled fields are None under --sup-only."""
    img_x: torch.Tensor
    mask_x: torch.Tensor
    img_u_s1: Optional[torch.Tensor] = None   # strong views, CutMix already applied
    img_u_s2: Optional[torch.Tensor] = None
    mask_u_w: Optional[torch.Tensor] = None   # teacher pseudo-label on the weak view
    conf_u_w: Optional[torch.Tensor] = None   # teacher confidence on the weak view
    ignore_mask: Optional[torch.Tensor] = None
    cutmix_box1: Optional[torch.Tensor] = None
    cutmix_box2: Optional[torch.Tensor] = None
    mask_u_gt: Optional[torch.Tensor] = None  # GT of the unlabeled batch; not used by the loss yet


def cutmix_(img, box):
    """In-place CutMix of a strong view: inside `box` ((B, H, W), 1 = mix) every
    image takes the pixels of the batch flipped along dim 0, i.e. image i receives
    image B-1-i. The caller applies the same box to the pseudo-label, confidence and
    ignore mask so the targets line up with the mixed images."""
    sel = box.unsqueeze(1).expand(img.shape) == 1
    img[sel] = img.flip(0)[sel]


def load_micro_batch(loader, model_ema, sup_only, bf16):
    """Pull the next micro-batch off `loader` and prepare what the student step
    consumes: every tensor on the GPU, the teacher's (EMA model) pseudo-label and
    confidence on the weak view, and CutMix applied to the two strong views. Under
    --sup-only the loader yields only the labeled pair. Draws from no RNG: the
    teacher runs in eval mode without Complementary Dropout."""
    if sup_only:
        img_x, mask_x = next(loader)
        return MicroBatch(img_x.cuda(), mask_x.cuda())
    (img_x, mask_x), (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2, mask_u_gt) = next(loader)
    img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
    ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
    mask_u_gt = mask_u_gt.cuda()

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        pred_u_w = model_ema(img_u_w)
        conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
        mask_u_w = pred_u_w.argmax(dim=1)

    cutmix_(img_u_s1, cutmix_box1)
    cutmix_(img_u_s2, cutmix_box2)
    return MicroBatch(img_x.cuda(), mask_x.cuda(), img_u_s1, img_u_s2, mask_u_w, conf_u_w,
                      ignore_mask, cutmix_box1, cutmix_box2, mask_u_gt)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    if args.data_root is not None:
        cfg['data_root'] = args.data_root
    if args.backbone is not None:
        cfg['backbone'] = args.backbone
    if args.epochs is not None:
        cfg['epochs'] = args.epochs

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # before all_args, so the effective values are what gets logged and sent to W&B
    apply_env_overrides(args, logger)
    if args.lock_backbone:
        cfg['lock_backbone'] = True
    if args.lr is not None:
        cfg['lr'] = args.lr
    args.lr = cfg['lr']  # so all_args / W&B record the effective value either way
    if args.lr <= 0:
        raise ValueError('--lr must be > 0 (got %g)' % args.lr)
    if args.lr_multi is not None:
        cfg['lr_multi'] = args.lr_multi
    args.lr_multi = cfg['lr_multi']  # so all_args / W&B record the effective value either way
    if args.sup_only and args.ggr != 'none':
        raise ValueError('--sup-only has no unsupervised gradient to rectify; drop --ggr %s' % args.ggr)
    if args.ggr != 'none' and args.ggr_anchor == 'class' and args.ggr_scope not in ('head', 'all'):
        raise ValueError('--ggr-anchor class builds its anchors in the surgery-scope space; use '
                         '--ggr-scope head or all (got %s)' % args.ggr_scope)
    if args.ggr == 'cone' and args.ggr_anchor != 'class':
        raise ValueError('--ggr cone projects onto the per-class prototype cone; it requires '
                         '--ggr-anchor class (got %s)' % args.ggr_anchor)
    if args.ggr_cone_rescale < 0:
        raise ValueError('--ggr-cone-rescale must be >= 0 (got %g)' % args.ggr_cone_rescale)
    if args.ggr_cone_rescale and args.ggr != 'cone':
        raise ValueError('--ggr-cone-rescale only applies to --ggr cone (got --ggr %s)' % args.ggr)
    if not 0.0 <= args.proto_beta < 1.0:
        raise ValueError('--proto-beta must be in [0, 1) (got %g)' % args.proto_beta)
    # before setup_distributed: the first CUDA call happens in there, and
    # CUBLAS_WORKSPACE_CONFIG has to be in place before cuBLAS initialises
    setup_determinism(args, logger)

    rank, world_size = setup_distributed(port=args.port)
    assert world_size == 1, 'this script is single-GPU only; use unimatch_v2.py for multi-GPU'
    assert EFFECTIVE_BATCH % cfg['batch_size'] == 0
    accum = EFFECTIVE_BATCH // cfg['batch_size']

    all_args = {**cfg, **vars(args), 'ngpus': world_size, 'accum_steps': accum}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    os.makedirs(args.save_path, exist_ok=True)

    # mode precedence: USE_WANDB env var > wandb_mode in the config > online
    wandb_mode = os.environ.get('USE_WANDB', cfg.get('wandb_mode', 'online'))
    if wandb_mode not in ('online', 'offline', 'disabled'):
        raise ValueError(f"invalid wandb mode '{wandb_mode}' (from USE_WANDB or wandb_mode config); "
                         f"expected 'online', 'offline' or 'disabled'")

    save_path = os.path.normpath(args.save_path)
    run_name = os.path.relpath(save_path, 'exp') if save_path.split(os.sep)[0] == 'exp' else save_path
    # keep one W&B run across container restarts: the id is persisted next to latest.pth,
    # so a relaunch resumes the run (like the checkpoint) instead of splitting its history
    # into a CRASHED run plus a fresh one. An explicit WANDB_RUN_ID env var wins if set.
    run_id_file = os.path.join(args.save_path, 'wandb_run_id.txt')
    run_id = os.environ.get('WANDB_RUN_ID')
    if run_id is None and os.path.exists(run_id_file):
        run_id = open(run_id_file).read().strip()
    if not run_id:
        # any unique string is a valid wandb id (wandb.util.generate_id is gone in 0.28)
        run_id = os.urandom(4).hex()
    with open(run_id_file, 'w') as f:
        f.write(run_id)
    wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'unimatch-v2'),
        name=run_name,
        config=all_args,
        dir=args.save_path,
        mode=wandb_mode,
        id=run_id,
        resume='allow'
    )
    # train/* and grad/* are logged per optimizer step, eval/* per epoch
    wandb.define_metric('iters')
    wandb.define_metric('epoch')
    wandb.define_metric('train/*', step_metric='iters')
    wandb.define_metric('grad/*', step_metric='iters')
    wandb.define_metric('eval/*', step_metric='epoch')

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    if args.init_seed is not None:
        # nothing between setup_determinism's seed_everything and this line draws from
        # torch's RNG, and every epoch reseeds it before training, so this changes the
        # head init (the backbone is overwritten from the checkpoint below) and nothing else
        torch.manual_seed(args.init_seed)
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)

    if cfg['lock_backbone']:
        model.lock_backbone()

    optimizer = build_adamw(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ],
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )

    logger.info('Total params: {:.1f}M'.format(count_params(model)))
    logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
    logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))

    model.cuda()

    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )

    # one generator per loader: it drives both the shuffle order and, through
    # torch's worker-seeding, the augmentation RNGs inside the workers. Distinct
    # seeds so the labeled and unlabeled streams do not share a permutation.
    gen_l, gen_u, gen_val = (torch.Generator() for _ in range(3))
    unlabeled_seed = args.seed if args.unlabeled_seed is None else args.unlabeled_seed
    gen_l.manual_seed(args.seed + 1)
    gen_u.manual_seed(unlabeled_seed + 2)
    gen_val.manual_seed(args.seed + 3)

    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True,
        generator=gen_l
    )
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True,
        generator=gen_u
    )
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, generator=gen_val
    )

    # partially filled accumulation groups at epoch end are dropped, so leftover
    # gradients can never leak into the next epoch's first optimizer step
    steps_per_epoch = len(trainloader_u) // accum
    total_steps = steps_per_epoch * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    # running count of optimizer steps whose cos(g_x, g_s) < GRAD_COS_CONFLICT_THRESH (supervised and
    # unsupervised gradients pulling against each other), over all steps seen so far
    # (persisted in the checkpoint across resumes)
    grad_cos_conflict_steps, grad_cos_total_steps = 0, 0

    resume_proto = (None, None)
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        # weights_only=False: this file is written by the block at the end of this loop, so
        # it is as trusted as the script itself. The default flipped to True in torch 2.6,
        # and checkpoints written before the float() cast below carry a numpy scalar in
        # previous_best, which the restricted unpickler refuses ("Unsupported global:
        # numpy._core.multiarray.scalar") -- i.e. every resume died at this line.
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu',
                                weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        # .get: checkpoints written before these counters existed start counting from here
        grad_cos_conflict_steps = checkpoint.get('grad_cos_conflict_steps', 0)
        grad_cos_total_steps = checkpoint.get('grad_cos_total_steps', 0)
        resume_proto = checkpoint.get('ggr_proto'), checkpoint.get('ggr_proto_updates')

        # per-epoch reseeding makes a resume reproducible only under the same seed
        ckpt_seed = checkpoint.get('seed')
        if ckpt_seed is not None and ckpt_seed != args.seed:
            logger.warning('resuming a run written with seed %s under seed %s: epochs up to %d used '
                           'the old seed' % (ckpt_seed, args.seed, epoch))

        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    # grad_x_acc accumulates the supervised (loss_x) part of the current optimizer
    # step's gradient; grad_s_acc receives the unsupervised part at step end as
    # .grad minus grad_x_acc. Both are lists of *views* into one contiguous flat
    # buffer, so the same memory is addressable per-parameter (for accumulation)
    # and as a single D-vector (for the GGR projections, which are defined on the
    # flattened scope). model.parameters() yields the backbone first, so every
    # surgery scope is a contiguous slice of both the param list and the buffer.
    params = [p for p in model.parameters() if p.requires_grad]
    n_backbone = len([p for p in model.backbone.parameters() if p.requires_grad])
    assert all(a is b for a, b in zip(params[:n_backbone],
                                      (p for p in model.backbone.parameters() if p.requires_grad))), \
        'expected the trainable backbone parameters to come first in model.parameters()'
    numels = [p.numel() for p in params]
    D_total = sum(numels)
    D_backbone = sum(numels[:n_backbone])

    def _flat_views(flat):
        views, off = [], 0
        for p, n in zip(params, numels):
            views.append(flat[off:off + n].view_as(p))
            off += n
        return views

    flat_x = torch.zeros(D_total, device=params[0].device, dtype=params[0].dtype)
    flat_s = torch.zeros(D_total, device=params[0].device, dtype=params[0].dtype)
    grad_x_acc = _flat_views(flat_x)
    grad_s_acc = _flat_views(flat_s)

    # surgery scope P: a contiguous [lo, hi) slice of the flat buffer and the
    # matching [p_lo, p_hi) slice of the parameter list
    if args.ggr_scope == 'backbone':
        scope_lo, scope_hi, scope_p_lo, scope_p_hi = 0, D_backbone, 0, n_backbone
    elif args.ggr_scope == 'head':
        scope_lo, scope_hi, scope_p_lo, scope_p_hi = D_backbone, D_total, n_backbone, len(params)
    else:
        scope_lo, scope_hi, scope_p_lo, scope_p_hi = 0, D_total, 0, len(params)
    D_scope = scope_hi - scope_lo

    # U holds an orthonormal basis of the d most recent supervised directions,
    # stored row-major as (d, D_scope) so each basis vector is contiguous.
    # u_k counts the filled rows; u_ptr is the ring-buffer slot the next append
    # overwrites (i.e. the oldest direction). U is deliberately not checkpointed:
    # it rebuilds from scratch in d optimizer steps after a resume.
    # u_iters tracks which optimizer step wrote each row so a the age of row i
    # is iters - u_iters[i]; It resets on resume like U itself does
    U, u_k, u_ptr, u_iters, ggr_scratch = None, 0, 0, None, None
    # Class-prototype anchors (--ggr-anchor class): proto[c] is the EMA of the per-class
    # supervised gradient (CE averaged over the pixels of class c) w.r.t. the head
    # parameters, laid out like flat_x[D_backbone:]. proto_acc / proto_cnt gather the
    # micro-batches of the current step; proto_updates counts EMA updates per class so
    # each class runs its Adam-style bias correction from its own first appearance
    # (see update_class_prototypes: proto stores the bias-corrected first moment).
    proto, proto_acc, proto_cnt, proto_updates, proto_w, proto_order, anchor_params = (None,) * 7
    # per-class prototypes are needed either as GGR anchors or purely as a diagnostic;
    # class_anchor is what the GGR block below keys on, so the log-only flag can never
    # redirect vlr/osr/csr away from their recent anchors
    class_anchor = args.ggr != 'none' and args.ggr_anchor == 'class'
    if args.sup_only:
        logger.info('supervised-only: unsupervised branch skipped; loss_s, mask_ratio and the '
                    'grad/*_s diagnostics are a zero stub\n')
    if class_anchor or args.log_class_cos:
        # Per-class supervised gradients live in the surgery-scope space: under
        # --ggr-scope head (the protocol default) that is the DPT head, laid out like
        # flat_x[D_backbone:]; under --ggr-scope all the anchors span the whole model,
        # laid out like flat_x. Log-only use (no GGR) stays in head space so the
        # class-cos artifacts keep their historical meaning. Whole-model anchors cost
        # 3-8 FULL backwards per micro-batch (the head-only ones stop at the head)
        # and ~3x the prototype memory on dinov2_small.
        n_cls = cfg['nclass']
        if class_anchor and args.ggr_scope == 'all':
            D_anchor, anchor_params = D_total, params
        else:
            D_anchor, anchor_params = D_total - D_backbone, params[n_backbone:]
        if D_anchor == 0:
            raise ValueError("trainable parameters are required for per-class gradients")
        proto = torch.zeros(n_cls, D_anchor, device=flat_x.device, dtype=flat_x.dtype)
        proto_acc = torch.zeros_like(proto)
        proto_cnt = torch.zeros(n_cls, device=flat_x.device)
        proto_updates = torch.zeros(n_cls, dtype=torch.long, device=flat_x.device)
        logger.info('per-class anchor gradients (scope=%s): %d classes x D=%.2fM (%.2f GiB)\n' % (
            args.ggr_scope if class_anchor else 'head',
            n_cls, D_anchor / 1e6, 2 * proto.numel() * proto.element_size() / 2 ** 30))
    if args.ggr != 'none':
        if D_scope == 0:
            raise ValueError('--ggr %s --ggr-scope %s selects no trainable parameters '
                             '(lock_backbone?)' % (args.ggr, args.ggr_scope))
        logger.info('GGR: mode=%s scope=%s D=%.2fM' % (args.ggr, args.ggr_scope, D_scope / 1e6))
        if class_anchor:
            freq = labeled_pixel_freq(cfg, args.labeled_id_path, args.save_path, logger)
            inv = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12), 0.0)
            proto_w = torch.tensor(inv / inv.sum(), device=flat_x.device, dtype=flat_x.dtype)
            # Gram-Schmidt order for the class basis: rarest class first, so the rare
            # classes' gradient directions enter the basis unaltered and the common ones
            # contribute only what is orthogonal to them
            proto_order = torch.tensor(np.argsort(np.where(freq > 0, freq, np.inf)), device=flat_x.device)
            if args.ggr in ('osr', 'csr'):
                U = torch.zeros(n_cls, D_scope, device=flat_x.device, dtype=flat_x.dtype)
            logger.info('GGR: class-prototype anchors (Adam beta=%.4g, horizon ~%.0f), %d classes x D=%.2fM (%.2f GiB incl. basis); 1/f weights: %s' % (
                args.proto_beta, 1.0 / max(1.0 - args.proto_beta, 1e-9),
                n_cls, D_scope / 1e6, (2 + (U is not None)) * proto.numel() * proto.element_size() / 2 ** 30,
                ' '.join('%s=%.3f' % (c, w) for c, w in zip(CLASSES[cfg['dataset']], proto_w.tolist()))))
        elif args.ggr in ('osr', 'csr'):
            if args.ggr_dim <= 0:
                raise ValueError('--ggr %s needs --ggr-dim > 0' % args.ggr)
            U = torch.zeros(args.ggr_dim, D_scope, device=flat_x.device, dtype=flat_x.dtype)
            u_iters = [None] * args.ggr_dim

            ggr_scratch = torch.empty(D_scope, device=flat_x.device, dtype=flat_x.dtype)
            logger.info('GGR: d=%d, U occupies %.2f GiB' % (
                args.ggr_dim, U.numel() * U.element_size() / 2 ** 30))
        logger.info('')
    if proto is not None and resume_proto[0] is not None and tuple(resume_proto[0].shape) == tuple(proto.shape):
        proto.copy_(resume_proto[0].to(proto.device))
        proto_updates.copy_(resume_proto[1].to(proto_updates.device))
        logger.info('GGR: restored class prototypes for %d classes from the checkpoint\n' % int((proto_updates > 0).sum()))

    cos_iters, cos_inst_log, cos_ema_log = [], [], []

    def save_class_cos(ep):
        if not cos_iters:
            return
        path = os.path.join(args.save_path, 'class_cos_ep%03d.npz' % ep)
        np.savez_compressed(path, iters=np.array(cos_iters),
                            cos_inst=np.stack(cos_inst_log), cos_ema=np.stack(cos_ema_log),
                            classes=np.array(CLASSES[cfg['dataset']]))
        wandb.save(path, base_path=args.save_path)
        logger.info('saved %d class-cosine matrices to %s' % (len(cos_iters), path))
        cos_iters.clear(), cos_inst_log.clear(), cos_ema_log.clear()

    # the loop bound is capped but total_steps above is not, so a limited run walks
    # the first epoch_lim epochs of the full schedule instead of a compressed one
    epoch_end = cfg['epochs']
    if args.epoch_lim > 0:
        epoch_end = min(cfg['epochs'], args.epoch_lim)
        logger.info('epoch_lim=%d: running %d of %d epochs; LR schedule spans all %d\n' % (
            args.epoch_lim, epoch_end, cfg['epochs'], cfg['epochs']))

    for epoch in range(epoch + 1, epoch_end):
        logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                    'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        # Reseed from (seed, epoch) rather than letting the streams run on. This
        # makes each epoch's data order and augmentations a pure function of the
        # two, so resuming at epoch N reproduces the uninterrupted run without
        # having to serialise RNG state into the checkpoint. Must happen before
        # the iterator below, which is when the samplers draw their permutations.
        epoch_seed = (args.seed + 100003 * (epoch + 1)) % 2 ** 31
        epoch_seed_u = (unlabeled_seed + 100003 * (epoch + 1)) % 2 ** 31
        seed_everything(epoch_seed)
        gen_l.manual_seed(epoch_seed + 1)
        gen_u.manual_seed(epoch_seed_u + 2)
        if args.unlabeled_seed is not None:
            # the only main-process RNG draws inside an epoch are Complementary Dropout's
            # (torch CPU: Binomial sample + randperm, model/semseg/dpt.py), which belong to
            # the unlabeled branch. Both loaders own their generators, so nothing on the
            # labeled side reads this RNG. Identical to seed_everything's call when the
            # two seeds coincide, hence the guard keeps the default path byte-for-byte.
            torch.manual_seed(epoch_seed_u)

        # In --sup-only mode the unlabeled loader is never iterated, so its workers
        # never spawn and the strong-view augmentations are never computed. gen_l is
        # per-loader, so the labeled order and augmentations are identical either way.
        loader = iter(trainloader_l) if args.sup_only else iter(zip(trainloader_l, trainloader_u))

        model.train()

        t_start = None

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            # one kernel each, now that the per-parameter buffers are views into these
            flat_x.zero_()
            flat_s.zero_()

            group_loss, group_loss_x, group_loss_s = 0.0, 0.0, 0.0

            for _ in range(accum):
                mb = load_micro_batch(loader, model_ema, args.sup_only, args.bf16)
                img_x, mask_x = mb.img_x, mb.mask_x

                # labeled and unlabeled losses live in separate graphs, so
                # backward each part right after its forward to halve peak memory.
                # The labeled part uses autograd.grad + manual accumulation into
                # .grad (numerically identical to .backward()) so grad_x_acc can
                # track the supervised gradient separately from the unsupervised
                # one that later backwards into the same .grad buffers.
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                    pred_x = model(img_x)
                    loss_x = criterion_l(pred_x, mask_x)
                if proto is not None:
                    # one anchor-scope backward per class present in the crop; the graph
                    # is kept for the full-loss backward below
                    accumulate_class_grads(pred_x, mask_x, anchor_params, proto_acc, proto_cnt)
                grads = torch.autograd.grad(loss_x / (2.0 * accum), params, allow_unused=True)
                for p, g_x, g in zip(params, grad_x_acc, grads):
                    if g is None:
                        continue
                    g_x += g
                    p.grad = g.contiguous() if p.grad is None else p.grad.add_(g)
                del grads

                if args.sup_only:
                    # nothing else backwards into .grad, so at step end grad_s_acc = .grad - grad_x_acc
                    # is exactly zero: the unsupervised stub every grad/* metric sees. The loss keeps
                    # its (loss_x + loss_u_s) / 2 form so train/loss_all and the applied gradient
                    # scale match the semi-supervised runs.
                    loss_u_s = torch.zeros((), device=img_x.device)
                    mask_ratio = 0.0
                else:
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                        pred_u_s1, pred_u_s2 = model(torch.cat((mb.img_u_s1, mb.img_u_s2)), comp_drop=True).chunk(2)

                    mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mb.mask_u_w.clone(), mb.conf_u_w.clone(), mb.ignore_mask.clone()
                    mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mb.mask_u_w.clone(), mb.conf_u_w.clone(), mb.ignore_mask.clone()

                    mask_u_w_cutmixed1[mb.cutmix_box1 == 1] = mb.mask_u_w.flip(0)[mb.cutmix_box1 == 1]
                    conf_u_w_cutmixed1[mb.cutmix_box1 == 1] = mb.conf_u_w.flip(0)[mb.cutmix_box1 == 1]
                    ignore_mask_cutmixed1[mb.cutmix_box1 == 1] = mb.ignore_mask.flip(0)[mb.cutmix_box1 == 1]

                    mask_u_w_cutmixed2[mb.cutmix_box2 == 1] = mb.mask_u_w.flip(0)[mb.cutmix_box2 == 1]
                    conf_u_w_cutmixed2[mb.cutmix_box2 == 1] = mb.conf_u_w.flip(0)[mb.cutmix_box2 == 1]
                    ignore_mask_cutmixed2[mb.cutmix_box2 == 1] = mb.ignore_mask.flip(0)[mb.cutmix_box2 == 1]

                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                        loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                        loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
                        loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

                        loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                        loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
                        loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

                        loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
                    (loss_u_s / (2.0 * accum)).backward()
                    mask_ratio = ((mb.conf_u_w >= cfg['conf_thresh']) & (mb.ignore_mask != 255)).sum().item() / (mb.ignore_mask != 255).sum().item()

                loss = (loss_x.detach() + loss_u_s.detach()) / 2.0

                group_loss += loss.item() / accum
                group_loss_x += loss_x.item() / accum
                group_loss_s += loss_u_s.item() / accum

                total_loss.update(loss.item())
                total_loss_x.update(loss_x.item())
                total_loss_s.update(loss_u_s.item())
                total_mask_ratio.update(mask_ratio)

            # norms/cosine of the supervised vs unsupervised parts of this step's
            # gradient, as applied (i.e. including the 1/(2*accum) loss scaling).
            # The reduction is split at the surgery-scope boundary so GGR gets the
            # scope-restricted inner products for free; the whole-model figures the
            # existing diagnostics use are just the two halves summed.
            stats_scope = torch.zeros(3, dtype=torch.float64, device='cuda')
            stats_rest = torch.zeros(3, dtype=torch.float64, device='cuda')
            for i, (p, g_x, g_s) in enumerate(zip(params, grad_x_acc, grad_s_acc)):
                if p.grad is None:
                    continue  # flat_s was zeroed at the top of the step
                torch.subtract(p.grad, g_x, out=g_s)
                tgt = stats_scope if scope_p_lo <= i < scope_p_hi else stats_rest
                tgt += torch.stack((
                    (g_x * g_x).sum(dtype=torch.float64),
                    (g_s * g_s).sum(dtype=torch.float64),
                    (g_x * g_s).sum(dtype=torch.float64)
                ))
            norm_x_sq, norm_s_sq, dot_xs = (stats_scope + stats_rest).tolist()
            scope_norm_x_sq, scope_norm_s_sq, scope_dot_xs = stats_scope.tolist()
            grad_norm_x, grad_norm_s = norm_x_sq ** 0.5, norm_s_sq ** 0.5
            grad_ratio_s_x = grad_norm_s / max(grad_norm_x, 1e-12)
            grad_cos_x_s = dot_xs / max(grad_norm_x * grad_norm_s, 1e-12)
            grad_cos_total_steps += 1
            grad_cos_conflict_steps += int(grad_cos_x_s < GRAD_COS_CONFLICT_THRESH)
            grad_cos_conflict_pct = 100.0 * grad_cos_conflict_steps / grad_cos_total_steps

            iters = epoch * steps_per_epoch + step

            seen, n_seen = None, 0
            if proto is not None:
                obs, obs_cls = update_class_prototypes(proto, proto_acc, proto_cnt, proto_updates,
                                                       beta=args.proto_beta)
                seen = (proto_updates > 0) & (proto.norm(dim=1) > GGR_CLASS_MIN_NORM)
                n_seen = int(seen.sum())
                if args.log_class_cos:
                    # instantaneous: only the classes in this step's labeled crops have a
                    # row; EMA: every class seen so far. Absent classes stay NaN.
                    cos_inst_log.append(pairwise_cos(obs, obs_cls, cfg['nclass']))
                    cos_ema_log.append(pairwise_cos(proto[seen], seen.nonzero(as_tuple=True)[0],
                                                    cfg['nclass']))
                    cos_iters.append(iters)
                proto_acc.zero_()
                proto_cnt.zero_()
                del obs

            # ------------------------------ GGR ------------------------------
            # Rectify the unsupervised gradient against the supervised anchor and
            # reassemble .grad on the surgery scope (util/ggr.py). Runs after the
            # raw-conflict diagnostics above and before optimizer.step(), so
            # grad/grad_cos_x_s keeps its meaning as the *pre*-rectification conflict
            # metric (paper section C.1) while grad/ggr_cos_after is the
            # post-rectification (applied) one. Empty ggr_stats under --ggr none.
            ggr_stats, u_k, u_ptr = ggr_rectify(
                args, iters=iters, flat_x=flat_x, flat_s=flat_s,
                scope_lo=scope_lo, scope_hi=scope_hi, scope_p_lo=scope_p_lo, scope_p_hi=scope_p_hi,
                params=params, grad_x_acc=grad_x_acc, grad_s_acc=grad_s_acc,
                scope_norm_x_sq=scope_norm_x_sq, scope_norm_s_sq=scope_norm_s_sq,
                scope_dot_xs=scope_dot_xs,
                class_anchor=class_anchor, seen=seen, n_seen=n_seen,
                proto=proto, proto_w=proto_w, proto_order=proto_order,
                U=U, u_k=u_k, u_ptr=u_ptr, u_iters=u_iters, ggr_scratch=ggr_scratch,
                class_names=CLASSES[cfg['dataset']], logger=logger)
            # ---------------------------- end GGR ----------------------------

            optimizer.step()
            lr = cfg['lr'] * (1 - iters / total_steps) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            wandb.log({
                'train/loss_all': group_loss,
                'train/loss_x': group_loss_x,
                'train/loss_s': group_loss_s,
                'train/mask_ratio': total_mask_ratio.val,
                'grad/grad_norm_x': grad_norm_x,
                'grad/grad_norm_s': grad_norm_s,
                'grad/grad_norm_s_over_x': grad_ratio_s_x,
                'grad/grad_cos_x_s': grad_cos_x_s,
                'grad/grad_cos_x_s_lt_%g_pct' % GRAD_COS_CONFLICT_THRESH: grad_cos_conflict_pct,
                'iters': iters,
                **ggr_stats
            })

            if step % max(steps_per_epoch // 8, 1) == 0:
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(step, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                                            total_loss_s.avg, total_mask_ratio.avg))

            # smoke-test throughput: the clock starts once the epoch's warm-up steps are
            # done and is read at the stop below; a stop earlier than that prints no timing
            if args.stop_after is not None and step == SMOKE_WARMUP_STEPS:
                torch.cuda.synchronize()
                t_start = time.time()

            if args.stop_after is not None and iters + 1 >= args.stop_after:
                torch.cuda.synchronize()
                timed_steps = step - SMOKE_WARMUP_STEPS if t_start is not None else 0
                if timed_steps > 0:
                    sec_per_step = (time.time() - t_start) / timed_steps
                    logger.info('Smoke test: {:.2f} s/step (avg over {} steps), total steps {}, projected {:.1f} h'.format(
                        sec_per_step, timed_steps, total_steps, sec_per_step * total_steps / 3600))
                logger.info('Peak GPU memory: {:.1f} GB'.format(torch.cuda.max_memory_allocated() / 1e9))
                save_class_cos(epoch)
                wandb.finish()
                return

        save_class_cos(epoch)

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        # evaluate returns np.mean(...), a numpy scalar, which propagates into
        # previous_best and from there into the checkpoint. torch.load's weights_only=True
        # path (the default since torch 2.6) cannot unpickle a numpy scalar, so an
        # uncast value here is what breaks the auto-resume above.
        mIoU, mIoU_ema = float(mIoU), float(mIoU_ema)

        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                        'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))

        eval_log = {'eval/mIoU': mIoU, 'eval/mIoU_ema': mIoU_ema, 'epoch': epoch}
        for i, iou in enumerate(iou_class):
            eval_log['eval/%s_IoU' % (CLASSES[cfg['dataset']][i])] = iou
            eval_log['eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i])] = iou_class_ema[i]
        wandb.log(eval_log)

        is_best = mIoU >= previous_best
        is_best_ema = mIoU_ema >= previous_best_ema

        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch

        checkpoint = {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
            'previous_best_ema': previous_best_ema,
            'best_epoch': best_epoch,
            'best_epoch_ema': best_epoch_ema,
            'grad_cos_conflict_steps': grad_cos_conflict_steps,
            'grad_cos_total_steps': grad_cos_total_steps,
            'seed': args.seed,
            'unlabeled_seed': unlabeled_seed,
            'init_seed': args.seed if args.init_seed is None else args.init_seed,
            'ggr_proto': proto.cpu() if proto is not None else None,
            'ggr_proto_updates': proto_updates.cpu() if proto is not None else None
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        if is_best_ema:
            torch.save(checkpoint, os.path.join(args.save_path, 'best_ema.pth'))

    wandb.finish()


if __name__ == '__main__':
    main()
