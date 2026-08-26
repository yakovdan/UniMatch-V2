import argparse
from copy import deepcopy
import logging
import os
import pprint
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from util.optim import build_adamw
from torch.utils.data import DataLoader
import wandb
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
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
parser.add_argument('--bf16', action='store_true', help='bf16 autocast for forwards/losses (recipe deviation: paper trains fp32)')
parser.add_argument('--stop-after', type=int, default=None, help='debug: stop after N optimizer steps (smoke test)')
parser.add_argument('--ggr', type=str, default='none', choices=['none', 'vlr', 'osr', 'csr'],
                    help='rectify the unsupervised gradient against the supervised one (Chen et al., '
                         'Geometric Gradient Rectification): vector-level, orthogonal-subspace or conic-subspace')
parser.add_argument('--ggr-dim', type=int, default=10,
                    help='subspace dimension d for --ggr osr/csr; ignored by vlr')
parser.add_argument('--ggr-scope', type=str, default='all', choices=['backbone', 'head', 'all'],
                    help='surgery scope P; the paper applies GGR to the encoder backbone by default')
parser.add_argument('--ggr-reorth-every', type=int, default=0,
                    help='re-orthonormalise U every N steps (0 = off; the Gram-Schmidt append already '
                         're-projects, so drift is normally negligible)')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

EFFECTIVE_BATCH = 16  # 4 GPUs x batch 4 in the paper
GRAD_COS_CONFLICT_THRESH = -0.05  # cos(g_x, g_s) below this counts as a conflicting-gradient step
GGR_MIN_RESIDUAL = 1e-8   # below this the current anchor adds no direction the basis lacks
GGR_ORTH_LOG_EVERY = 200  # how often to log max|U^T U - I| as a basis-health check


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

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
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

    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True
    )
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True
    )
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False
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

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
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
    U, u_k, u_ptr, ggr_scratch = None, 0, 0, None
    if args.ggr != 'none':
        if D_scope == 0:
            raise ValueError('--ggr %s --ggr-scope %s selects no trainable parameters '
                             '(lock_backbone?)' % (args.ggr, args.ggr_scope))
        logger.info('GGR: mode=%s scope=%s D=%.2fM' % (args.ggr, args.ggr_scope, D_scope / 1e6))
        if args.ggr in ('osr', 'csr'):
            if args.ggr_dim <= 0:
                raise ValueError('--ggr %s needs --ggr-dim > 0' % args.ggr)
            U = torch.zeros(args.ggr_dim, D_scope, device=flat_x.device, dtype=flat_x.dtype)
            ggr_scratch = torch.empty(D_scope, device=flat_x.device, dtype=flat_x.dtype)
            logger.info('GGR: d=%d, U occupies %.2f GiB' % (
                args.ggr_dim, U.numel() * U.element_size() / 2 ** 30))
        logger.info('')

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                    'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = iter(zip(trainloader_l, trainloader_u))

        model.train()

        t_start, timed_steps = None, 0

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            # one kernel each, now that the per-parameter buffers are views into these
            flat_x.zero_()
            flat_s.zero_()

            group_loss, group_loss_x, group_loss_s = 0.0, 0.0, 0.0

            for _ in range(accum):
                (img_x, mask_x), (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2, mask_u_gt) = next(loader)

                img_x, mask_x = img_x.cuda(), mask_x.cuda()
                img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
                ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
                mask_u_gt = mask_u_gt.cuda()  # GT of the unlabeled batch; not used by the loss yet

                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                    pred_u_w = model_ema(img_u_w).detach()
                    conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                    mask_u_w = pred_u_w.argmax(dim=1)

                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
                img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

                # labeled and unlabeled losses live in separate graphs, so
                # backward each part right after its forward to halve peak memory.
                # The labeled part uses autograd.grad + manual accumulation into
                # .grad (numerically identical to .backward()) so grad_x_acc can
                # track the supervised gradient separately from the unsupervised
                # one that later backwards into the same .grad buffers.
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                    pred_x = model(img_x)
                    loss_x = criterion_l(pred_x, mask_x)
                grads = torch.autograd.grad(loss_x / (2.0 * accum), params, allow_unused=True)
                for p, g_x, g in zip(params, grad_x_acc, grads):
                    if g is None:
                        continue
                    g_x += g
                    p.grad = g.contiguous() if p.grad is None else p.grad.add_(g)
                del grads

                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                    pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)

                mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]

                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]

                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                    loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                    loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
                    loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

                    loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                    loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
                    loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

                    loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
                (loss_u_s / (2.0 * accum)).backward()

                loss = (loss_x.detach() + loss_u_s.detach()) / 2.0

                group_loss += loss.item() / accum
                group_loss_x += loss_x.item() / accum
                group_loss_s += loss_u_s.item() / accum

                total_loss.update(loss.item())
                total_loss_x.update(loss_x.item())
                total_loss_s.update(loss_u_s.item())
                mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
                total_mask_ratio.update(mask_ratio.item())

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

            # ------------------------------ GGR ------------------------------
            # Chen et al., "Geometric Gradient Rectification for Safe Open-Set SSL".
            # The supervised gradient is the anchor and is never modified; only the
            # unsupervised part is projected, and the two are then re-summed into
            # .grad. This runs after the raw-conflict diagnostics above and before
            # optimizer.step(), so grad/grad_cos_x_s keeps its meaning as the
            # *pre*-rectification conflict metric (paper section C.1) while
            # grad/ggr_cos_after is the post-rectification (applied) one.
            ggr_stats = {}
            if args.ggr != 'none':
                g_x_scope = flat_x[scope_lo:scope_hi]  # the paper's g_s, the anchor
                g_u_scope = flat_s[scope_lo:scope_hi]  # the paper's g_u, the auxiliary
                delta_norm = 0.0

                if args.ggr in ('osr', 'csr'):
                    # UpdateSubspace (Alg. 1 lines 14-19): keep only the part of the
                    # current anchor that span(U) does not already contain, normalise
                    # it, and write it over the oldest basis vector.
                    v = ggr_scratch.copy_(g_x_scope)
                    for _ in range(2):  # project twice: fp32 Gram-Schmidt loses orthogonality once
                        if u_k > 0:
                            v -= (U[:u_k] @ v) @ U[:u_k]
                    nv = v.norm().item()
                    if nv > GGR_MIN_RESIDUAL:
                        U[u_ptr] = v / nv
                        u_ptr = (u_ptr + 1) % U.shape[0]
                        u_k = min(u_k + 1, U.shape[0])

                if args.ggr == 'vlr':
                    # Eq. (5): fires only on a conflict, and removes exactly the
                    # component of g_u that points along -g_s.
                    if scope_dot_xs < 0.0 and scope_norm_x_sq > 0.0:
                        c = scope_dot_xs / scope_norm_x_sq  # c < 0 here
                        g_u_scope.add_(g_x_scope, alpha=-c)  # -c > 0: injects |c|*g_x
                        delta_norm = abs(scope_dot_xs) / scope_norm_x_sq ** 0.5
                        ggr_stats['grad/ggr_fired'] = 1.0
                    else:
                        ggr_stats['grad/ggr_fired'] = 0.0
                elif u_k > 0:
                    coef = U[:u_k] @ g_u_scope  # coordinates of g_u in the basis
                    if args.ggr == 'csr':
                        # Eq. (8): clip only the coordinates opposing a stored anchor
                        clipped = (coef < 0)
                        ggr_stats['grad/ggr_fired'] = float(clipped.any().item())
                        ggr_stats['grad/ggr_clipped_frac'] = clipped.float().mean().item()
                        coef = coef.clamp(max=0.0)
                    else:
                        # Eq. (7): remove the whole component inside span(U)
                        ggr_stats['grad/ggr_fired'] = 1.0
                    # U has orthonormal rows, so ||U^T coef|| == ||coef||
                    delta_norm = coef.norm().item()
                    torch.addmv(g_u_scope, U[:u_k].t(), coef, beta=1.0, alpha=-1.0, out=g_u_scope)

                # reassemble .grad = g_x + rectified g_u, on the scope only
                for p, g_x, g_s in zip(params[scope_p_lo:scope_p_hi],
                                       grad_x_acc[scope_p_lo:scope_p_hi],
                                       grad_s_acc[scope_p_lo:scope_p_hi]):
                    if p.grad is None:
                        continue
                    torch.add(g_x, g_s, out=p.grad)

                # applied-conflict diagnostics: >= 0 for vlr (Prop. 2), ~0 for osr
                # once the anchor is in span(U) (Prop. 4)
                dot_after = torch.dot(g_x_scope, g_u_scope).item()
                norm_s_after = g_u_scope.norm().item()
                ggr_stats.update({
                    'grad/ggr_cos_after': dot_after / max(scope_norm_x_sq ** 0.5 * norm_s_after, 1e-12),
                    'grad/ggr_delta_norm': delta_norm,
                    'grad/ggr_delta_rel': delta_norm / max(scope_norm_s_sq ** 0.5, 1e-12),
                    'grad/ggr_subspace_k': float(u_k),
                })

                if U is not None and u_k > 1:
                    if args.ggr_reorth_every and iters % args.ggr_reorth_every == 0:
                        for a in range(u_k):  # modified Gram-Schmidt, in place
                            for b in range(a):
                                U[a] -= torch.dot(U[b], U[a]) * U[b]
                            U[a] /= U[a].norm().clamp_min(1e-12)
                    if GGR_ORTH_LOG_EVERY and iters % GGR_ORTH_LOG_EVERY == 0:
                        gram = U[:u_k] @ U[:u_k].t() - torch.eye(u_k, device=U.device, dtype=U.dtype)
                        ggr_stats['grad/ggr_orth_err'] = gram.abs().max().item()
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

            # skip the first steps when timing: cudnn.benchmark autotunes there
            if step == 4:
                torch.cuda.synchronize()
                t_start = time.time()
            elif t_start is not None:
                timed_steps = step - 4

            if args.stop_after is not None and iters + 1 >= args.stop_after:
                torch.cuda.synchronize()
                if timed_steps > 0:
                    sec_per_step = (time.time() - t_start) / timed_steps
                    logger.info('Smoke test: {:.2f} s/step (avg over {} steps), total steps {}, projected {:.1f} h'.format(
                        sec_per_step, timed_steps, total_steps, sec_per_step * total_steps / 3600))
                logger.info('Peak GPU memory: {:.1f} GB'.format(torch.cuda.max_memory_allocated() / 1e9))
                wandb.finish()
                return

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)

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
            'grad_cos_total_steps': grad_cos_total_steps
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        if is_best_ema:
            torch.save(checkpoint, os.path.join(args.save_path, 'best_ema.pth'))

    wandb.finish()


if __name__ == '__main__':
    main()
