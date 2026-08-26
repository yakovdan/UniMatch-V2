import argparse
from copy import deepcopy
import logging
import os
import pprint
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
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
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

EFFECTIVE_BATCH = 16  # 4 GPUs x batch 4 in the paper
GRAD_COS_CONFLICT_THRESH = -0.05  # cos(g_x, g_s) below this counts as a conflicting-gradient step
CLASS_GRAD_EMA_BETA = 0.95  # decay of the per-class supervised gradient EMA (see SUP_CLASS_GRAD_EMA)
PROTO_EMA_BETA = 0.95  # decay of the per-class feature-prototype EMA (see PROTO_STATS)


def rank_auc(scores, positive):
    """AUC of `scores` as a detector of `positive` (bool mask): P(score_pos > score_neg), ties counted 1/2.
    Returns nan if either group is empty."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    scores = scores.double()
    # average ranks (1-based) so ties contribute 1/2
    order = scores.argsort()
    _, inv, counts = torch.unique_consecutive(scores[order], return_inverse=True, return_counts=True)
    counts = counts.double()
    first = torch.cumsum(counts, 0) - counts + 1                 # 1-based rank of the first member of each tie group
    ranks = torch.empty_like(scores)
    ranks[order] = (first + (counts - 1) / 2.0)[inv]             # mean rank within the tie group
    return ((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)).item()


def proto_bank_means(proto_S, proto_R, proto_rho, proto_updates, min_rho=1e-4):
    """Class prototypes from the per-class EMA statistics (all per labeled pixel of class c):
    proto_S[c] = mean f, proto_R[c, k] = mean p_k f, proto_rho[c, k] = mean p_k.
      mu[c]        = (S_c - R_cc) / (1 - rho_cc): the (1 - p_c)-weighted class mean, i.e. the direction of the
                     'toward' row [a^(c)]_c of the class-c classifier-gradient anchor;
      mu_sub[c, k] = R_ck / rho_ck (k != c): the p_k-weighted mean of class-c pixels confused with k, i.e. the
                     direction of the residual row [a^(c)]_k. Rows are unit-normalised; unavailable entries are
                     flagged in the returned masks (class never seen, or rho_ck < min_rho)."""
    K = proto_S.shape[0]
    seen = torch.tensor([u > 0 for u in proto_updates], device=proto_S.device)
    diag_rho = proto_rho.diagonal()
    diag_R = proto_R[torch.arange(K), torch.arange(K)]
    mu = (proto_S - diag_R) / (1.0 - diag_rho).clamp_min(1e-6)[:, None]
    mu = F.normalize(mu, dim=1)
    mu_sub = proto_R / proto_rho.clamp_min(min_rho)[:, :, None]
    mu_sub = F.normalize(mu_sub, dim=2)
    sub_ok = seen[:, None] & (proto_rho >= min_rho) & ~torch.eye(K, dtype=torch.bool, device=proto_S.device)
    return mu, seen, mu_sub, sub_ok


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
    # train/*, grad/*, grad_ema/*, grad_cls/* and proto/* are logged per optimizer step, eval/* per epoch
    wandb.define_metric('iters')
    wandb.define_metric('epoch')
    wandb.define_metric('train/*', step_metric='iters')
    wandb.define_metric('grad/*', step_metric='iters')
    wandb.define_metric('grad_ema/*', step_metric='iters')
    wandb.define_metric('grad_cls/*', step_metric='iters')
    wandb.define_metric('proto/*', step_metric='iters')
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

    # The classifier: the final 1x1 conv of the DPT head (dpt.py output_conv[2]). Every pixel's logits are a
    # linear function of its input feature f (d channels, head resolution, before the x14 upsample), so the
    # per-pixel classifier gradient is (p - e_label) f^T and its row k is the gradient w.r.t. prototype w_k.
    classifier = model.head.scratch.output_conv[2]
    cls_params = [p for p in classifier.parameters() if p.requires_grad]  # [weight (K, d, 1, 1), bias (K)]
    n_cls_params = sum(p.numel() for p in cls_params)
    class_names = CLASSES[cfg['dataset']]

    # PROTO_STATS: feature-level prototype diagnostics. A forward hook on the classifier captures its input
    # (f) and output (the logits) for the most recent forward of `model`; registered after the deepcopy so
    # the EMA teacher's forwards never overwrite it. Detached, so no graph is kept alive.
    proto_stats = os.environ.get('PROTO_STATS', '1').strip().lower() not in ('0', 'false', 'no', 'off')
    cls_capture = {}
    if proto_stats:
        def _capture_cls_io(module, inputs, output):
            cls_capture['f'] = inputs[0].detach()
            cls_capture['z'] = output.detach()
        classifier.register_forward_hook(_capture_cls_io)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    # Per-class EMA of the supervised classifier gradient: for every GT class c, an EMA over the
    # optimizer steps where c occurs of the gradient of the class's own mean CE over the effective
    # batch w.r.t. the classifier only (the final 1x1 conv of the DPT head, dpt.py output_conv[2]),
    #   g_c = d/dtheta_cls [ sum_{p: mask_x[p] = c} CE_p / N_c ].
    # Measured only, never applied. autograd.grad w.r.t. those params only backprops through the
    # CE and that conv, so the extra cost is small; SUP_CLASS_GRAD_EMA=0 disables it.
    class_grad_ema = os.environ.get('SUP_CLASS_GRAD_EMA', '1').strip().lower() not in ('0', 'false', 'no', 'off')
    if class_grad_ema:
        if cfg['criterion']['name'] != 'CELoss':
            raise NotImplementedError('the per-class supervised gradient EMA needs the CELoss criterion (OHEM\'s '
                                      'per-pixel selection is not decomposed); set SUP_CLASS_GRAD_EMA=0 to skip it')
        # per-pixel CE with the same kwargs (ignore_index) as criterion_l
        criterion_l_pix = nn.CrossEntropyLoss(**{**cfg['criterion']['kwargs'], 'reduction': 'none'}).cuda()

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

    # accumulates the supervised (loss_x) part of the current optimizer step's
    # gradient; the unsupervised part is recovered at step end as .grad minus this
    params = [p for p in model.parameters() if p.requires_grad]
    grad_x_acc = [torch.zeros_like(p) for p in params]
    # gradients of the unconfident-correct (uc) / -incorrect (ui) losses, accumulated like
    # grad_x_acc but only measured, never applied. UNCONF_GRAD=full (default, also '1'): whole-model
    # gradients, two extra full backward passes per micro-batch, whole-model cosines vs g_x / g_s.
    # UNCONF_GRAD=cls: classifier-only gradients (two tiny backward passes), enough for the per-class
    # anchor comparison below but no whole-model cosines. UNCONF_GRAD=0: off.
    unconf_mode = os.environ.get('UNCONF_GRAD', 'full').strip().lower()
    unconf_mode = 'off' if unconf_mode in ('0', 'false', 'no', 'off') else ('cls' if unconf_mode in ('cls', 'classifier') else 'full')
    unconf_grad = unconf_mode != 'off'
    unconf_params = cls_params if unconf_mode == 'cls' else params
    logger.info('Unconfident-loss gradients: {}'.format({'full': 'whole model (UNCONF_GRAD=full)', 'cls': 'classifier only (UNCONF_GRAD=cls)', 'off': 'off (UNCONF_GRAD=0)'}[unconf_mode]))
    grad_uc_acc = [torch.zeros_like(p) for p in unconf_params] if unconf_grad else None
    grad_ui_acc = [torch.zeros_like(p) for p in unconf_params] if unconf_grad else None
    # positions of the classifier's parameters inside the accumulators (identity match), so their rows
    # can be sliced out in either mode
    unconf_cls_idx = [next(i for i, p in enumerate(unconf_params) if p is q) for q in cls_params] if unconf_grad else None
    cls_idx_in_params = [next(i for i, p in enumerate(params) if p is q) for q in cls_params]

    if class_grad_ema:
        logger.info('Per-class supervised classifier-gradient EMA: {} classes x {} classifier params, beta={} '
                    '(set SUP_CLASS_GRAD_EMA=0 to skip)'.format(cfg['nclass'], n_cls_params, CLASS_GRAD_EMA_BETA))
        # one flat (n_cls_params,) row per class, in `cls_params` order (weight, bias); a row is
        # meaningless until the class's first update (grad_ema_cls_updates[c] > 0), which initializes it
        grad_ema_cls = torch.zeros(cfg['nclass'], n_cls_params, device='cuda')
        grad_ema_cls_updates = [0] * cfg['nclass']
        grad_ema_path = os.path.join(args.save_path, 'grad_ema.pth')
        if os.path.exists(os.path.join(args.save_path, 'latest.pth')) and os.path.exists(grad_ema_path):
            grad_ema_ckpt = torch.load(grad_ema_path, map_location='cpu')
            grad_ema_cls.copy_(grad_ema_ckpt['grad_ema_cls'])
            grad_ema_cls_updates = grad_ema_ckpt['updates']
            logger.info('************ Loaded per-class gradient EMA from {}\n'.format(grad_ema_path))
    else:
        logger.info('Per-class supervised gradient EMA: off (SUP_CLASS_GRAD_EMA=0)')

    if proto_stats:
        K, d_feat = cfg['nclass'], classifier.in_channels
        logger.info('Prototype diagnostics (PROTO_STATS): {} classes x {} feature channels at head resolution, '
                    'beta={} (set PROTO_STATS=0 to skip)'.format(K, d_feat, PROTO_EMA_BETA))
        # per-labeled-pixel-of-class-c means, EMA'd over the optimizer steps where c occurs (first
        # observation initializes): S[c] = mean f, R[c, k] = mean p_k f, rho[c, k] = mean p_k.
        # See proto_bank_means() for what is derived from them.
        proto_S = torch.zeros(K, d_feat, device='cuda')
        proto_R = torch.zeros(K, K, d_feat, device='cuda')
        proto_rho = torch.zeros(K, K, device='cuda')
        proto_updates = [0] * K
        proto_path = os.path.join(args.save_path, 'proto_ema.pth')
        if os.path.exists(os.path.join(args.save_path, 'latest.pth')) and os.path.exists(proto_path):
            proto_ckpt = torch.load(proto_path, map_location='cpu')
            proto_S.copy_(proto_ckpt['S']); proto_R.copy_(proto_ckpt['R']); proto_rho.copy_(proto_ckpt['rho'])
            proto_updates = proto_ckpt['updates']
            logger.info('************ Loaded prototype EMA from {}\n'.format(proto_path))
    else:
        logger.info('Prototype diagnostics: off (PROTO_STATS=0)')

    log_debug_prefixes = tuple(p for p in os.environ.get('LOG_DEBUG_PREFIXES', '').split(',') if p)

    def downsample_mask(m, size):
        # nearest-neighbour resize of a (B, H, W) label / confidence map to the head resolution
        return F.interpolate(m[:, None].float(), size=size, mode='nearest')[:, 0]

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                    'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_s_unconf = AverageMeter()
        total_loss_s_unconf_correct = AverageMeter()
        total_loss_s_unconf_incorrect = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = iter(zip(trainloader_l, trainloader_u))

        model.train()

        t_start, timed_steps = None, 0

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            for g_x in grad_x_acc:
                g_x.zero_()
            if unconf_grad:
                for g_uc, g_ui in zip(grad_uc_acc, grad_ui_acc):
                    g_uc.zero_()
                    g_ui.zero_()
            # this step's per-class sums of d/dtheta_cls sum_{p in c} CE_p and pixel counts N_c,
            # for the classes present in the step; divided at step end to get the mean-CE gradient
            cls_grad_sum, cls_pix_count = {}, {}
            # this step's per-class feature sums over labeled pixels (S, R, rho as in proto_bank_means,
            # un-normalised) and the complement-pixel scores accumulated for the step-end statistics
            proto_acc = {'S': {}, 'N': {}, 'R': {}, 'rho': {}}
            comp_buf = {'proto': [], 'proto_sub': [], 'conf': [], 'incorrect': [], 'disagree': [], 'student_eq_gt': []}
            group_loss, group_loss_x, group_loss_s, group_loss_s_unconf = 0.0, 0.0, 0.0, 0.0
            group_loss_s_unconf_correct, group_loss_s_unconf_incorrect = 0.0, 0.0

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
                    if class_grad_ema:
                        ce_x_pix = criterion_l_pix(pred_x, mask_x)
                # prototype statistics of this labeled micro-batch, from the classifier's captured input
                # (features at head resolution) and output (logits), with the labels resized to match
                if proto_stats:
                    with torch.no_grad():
                        f_l, z_l = cls_capture['f'].float(), cls_capture['z'].float()
                        y_l = downsample_mask(mask_x, f_l.shape[-2:]).long().reshape(-1)
                        F_l = f_l.permute(0, 2, 3, 1).reshape(-1, f_l.shape[1])
                        P_l = z_l.softmax(dim=1).permute(0, 2, 3, 1).reshape(-1, z_l.shape[1])
                        for c in torch.unique(y_l).tolist():
                            if not 0 <= c < cfg['nclass']:
                                continue
                            sel = y_l == c
                            Fc, Pc = F_l[sel], P_l[sel]
                            if c not in proto_acc['N']:
                                proto_acc['S'][c] = Fc.sum(0); proto_acc['N'][c] = int(sel.sum())
                                proto_acc['R'][c] = Pc.T @ Fc; proto_acc['rho'][c] = Pc.sum(0)
                            else:
                                proto_acc['S'][c] += Fc.sum(0); proto_acc['N'][c] += int(sel.sum())
                                proto_acc['R'][c] += Pc.T @ Fc; proto_acc['rho'][c] += Pc.sum(0)
                        del f_l, z_l, y_l, F_l, P_l
                # per-class classifier gradients first, with the graph retained; the existing
                # supervised backward below then consumes it exactly as before
                if class_grad_ema:
                    for c in torch.unique(mask_x).tolist():
                        if not 0 <= c < cfg['nclass']:  # ignore_index pixels belong to no class
                            continue
                        cls_mask = mask_x == c
                        grads = torch.autograd.grad((ce_x_pix * cls_mask).sum(), cls_params, retain_graph=True)
                        flat = torch.cat([g.reshape(-1) for g in grads])
                        del grads
                        if c in cls_grad_sum:
                            cls_grad_sum[c] += flat
                            cls_pix_count[c] += cls_mask.sum().item()
                        else:
                            cls_grad_sum[c] = flat
                            cls_pix_count[c] = cls_mask.sum().item()
                        del flat
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

                # GT of the unlabeled images in the same CutMix frame as the strong views
                mask_u_gt_cutmixed1, mask_u_gt_cutmixed2 = mask_u_gt.clone(), mask_u_gt.clone()
                mask_u_gt_cutmixed1[cutmix_box1 == 1] = mask_u_gt.flip(0)[cutmix_box1 == 1]
                mask_u_gt_cutmixed2[cutmix_box2 == 1] = mask_u_gt.flip(0)[cutmix_box2 == 1]

                # prototype diagnostics on the complement (conf < thresh, GT known) at head resolution, for
                # both strong views (the hook holds the 2B-batch forward): per pixel, the cosine of its
                # classifier-input feature to the EMA class prototypes gives the comparison-rule score
                #   s_other - s_label  (> 0: the labeled data's features say another class fits better),
                # its boundary-subpopulation variant, the confidence baseline, and the student/teacher
                # disagreement. Scores are pooled over the step and summarised at step end.
                if proto_stats and any(u > 0 for u in proto_updates):
                    with torch.no_grad():
                        f_u, z_u = cls_capture['f'].float(), cls_capture['z'].float()
                        hw = f_u.shape[-2:]
                        lab = torch.cat((downsample_mask(mask_u_w_cutmixed1, hw), downsample_mask(mask_u_w_cutmixed2, hw))).long()
                        gt = torch.cat((downsample_mask(mask_u_gt_cutmixed1, hw), downsample_mask(mask_u_gt_cutmixed2, hw))).long()
                        conf = torch.cat((downsample_mask(conf_u_w_cutmixed1, hw), downsample_mask(conf_u_w_cutmixed2, hw)))
                        ign = torch.cat((downsample_mask(ignore_mask_cutmixed1, hw), downsample_mask(ignore_mask_cutmixed2, hw))).long()
                        mu, seen, mu_sub, sub_ok = proto_bank_means(proto_S, proto_R, proto_rho, proto_updates)
                        comp = (conf < cfg['conf_thresh']) & (ign != 255) & (gt != 255) & seen[lab.clamp(max=cfg['nclass'] - 1)] & (lab < cfg['nclass'])
                        if comp.any():
                            Fu = F.normalize(f_u.permute(0, 2, 3, 1)[comp], dim=1)   # (n, d) unit features
                            lab_c, gt_c, conf_c = lab[comp], gt[comp], conf[comp]
                            student = z_u.argmax(dim=1)[comp]
                            sims = Fu @ mu.T                                         # (n, K) cosine to each prototype
                            sims[:, ~seen] = float('-inf')
                            s_label = sims.gather(1, lab_c[:, None])[:, 0]
                            s_other = sims.scatter(1, lab_c[:, None], float('-inf')).max(dim=1).values
                            # boundary-subpopulation variant: for a pixel labelled L, compare s_label with the
                            # best cosine to mu_sub[c, L] (class-c labeled pixels confused with L), c != L
                            s_other_sub = torch.full_like(s_label, float('-inf'))
                            for L in torch.unique(lab_c).tolist():
                                idx = lab_c == L
                                ok_c = sub_ok[:, L]
                                if ok_c.any():
                                    s_other_sub[idx] = (Fu[idx] @ mu_sub[:, L, :].T)[:, ok_c].max(dim=1).values
                            fin = torch.isfinite(s_other)
                            comp_buf['proto'].append((s_other - s_label)[fin])
                            comp_buf['proto_sub'].append((s_other_sub - s_label)[fin])
                            comp_buf['conf'].append(-conf_c[fin])
                            comp_buf['incorrect'].append((lab_c != gt_c)[fin])
                            comp_buf['disagree'].append((student != lab_c)[fin])
                            comp_buf['student_eq_gt'].append((student == gt_c)[fin])
                        del f_u, z_u, lab, gt, conf, ign, mu, mu_sub

                # "unconfident" losses: the same per-pixel CE over the complementary pixel set
                # (conf < thresh instead of >=), with the same normalizer, so
                # loss_u_s + loss_u_s_unconf is the CE mean over all valid pixels.
                # Kept in the graph (not detached) but not added to the loss and not
                # backpropagated yet; logged only for now.
                # Each unconfident loss is further split by whether the pseudo-label (the EMA
                # teacher's argmax, i.e. the CE target) matches the GT (pixels with GT == 255
                # are neither correct nor incorrect).
                # Each half is normalized as if the other half's pixels were removed from the
                # image, i.e. by (#valid - #other), clamped to >= 1.
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                    loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                    loss_u_s1_unconf = loss_u_s1 * ((conf_u_w_cutmixed1 < cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
                    loss_u_s1_unconf = loss_u_s1_unconf.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
                    with torch.no_grad():
                        unconf_known1 = (conf_u_w_cutmixed1 < cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255) & (mask_u_gt_cutmixed1 != 255)
                        correct1 = unconf_known1 & (mask_u_w_cutmixed1 == mask_u_gt_cutmixed1)
                        incorrect1 = unconf_known1 & ~correct1
                        n_valid1 = (ignore_mask_cutmixed1 != 255).sum().item()
                    loss_u_s1_unconf_correct = (loss_u_s1 * correct1).sum() / max(n_valid1 - incorrect1.sum().item(), 1)
                    loss_u_s1_unconf_incorrect = (loss_u_s1 * incorrect1).sum() / max(n_valid1 - correct1.sum().item(), 1)
                    loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
                    loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

                    loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                    loss_u_s2_unconf = loss_u_s2 * ((conf_u_w_cutmixed2 < cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
                    loss_u_s2_unconf = loss_u_s2_unconf.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
                    with torch.no_grad():
                        unconf_known2 = (conf_u_w_cutmixed2 < cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255) & (mask_u_gt_cutmixed2 != 255)
                        correct2 = unconf_known2 & (mask_u_w_cutmixed2 == mask_u_gt_cutmixed2)
                        incorrect2 = unconf_known2 & ~correct2
                        n_valid2 = (ignore_mask_cutmixed2 != 255).sum().item()
                    loss_u_s2_unconf_correct = (loss_u_s2 * correct2).sum() / max(n_valid2 - incorrect2.sum().item(), 1)
                    loss_u_s2_unconf_incorrect = (loss_u_s2 * incorrect2).sum() / max(n_valid2 - correct2.sum().item(), 1)
                    loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
                    loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

                    loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
                    loss_u_s_unconf = (loss_u_s1_unconf + loss_u_s2_unconf) / 2.0
                    loss_u_s_unconf_correct = (loss_u_s1_unconf_correct + loss_u_s2_unconf_correct) / 2.0
                    loss_u_s_unconf_incorrect = (loss_u_s1_unconf_incorrect + loss_u_s2_unconf_incorrect) / 2.0
                # measure the gradients of the two unconfident losses before the graph is
                # consumed: autograd.grad leaves .grad alone (so the optimizer step is unchanged)
                # and retain_graph keeps the same forward (same CutMix, same complementary-dropout
                # masks) for all three backward passes. Scaled like g_x / g_s so norms compare.
                if unconf_grad:
                    for loss_part, acc in ((loss_u_s_unconf_correct, grad_uc_acc), (loss_u_s_unconf_incorrect, grad_ui_acc)):
                        grads = torch.autograd.grad(loss_part / (2.0 * accum), unconf_params, retain_graph=True, allow_unused=True)
                        for g_acc, g in zip(acc, grads):
                            if g is not None:
                                g_acc += g
                        del grads
                (loss_u_s / (2.0 * accum)).backward()

                loss = (loss_x.detach() + loss_u_s.detach()) / 2.0

                group_loss += loss.item() / accum
                group_loss_x += loss_x.item() / accum
                group_loss_s += loss_u_s.item() / accum
                group_loss_s_unconf += loss_u_s_unconf.item() / accum
                group_loss_s_unconf_correct += loss_u_s_unconf_correct.item() / accum
                group_loss_s_unconf_incorrect += loss_u_s_unconf_incorrect.item() / accum

                total_loss.update(loss.item())
                total_loss_x.update(loss_x.item())
                total_loss_s.update(loss_u_s.item())
                total_loss_s_unconf.update(loss_u_s_unconf.item())
                total_loss_s_unconf_correct.update(loss_u_s_unconf_correct.item())
                total_loss_s_unconf_incorrect.update(loss_u_s_unconf_incorrect.item())
                mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
                total_mask_ratio.update(mask_ratio.item())

            # norms/cosines of the supervised (x), unsupervised (s) and, if enabled, the
            # unconfident-correct (uc) / -incorrect (ui) gradients of this step, as applied
            # (i.e. including the 1/(2*accum) loss scaling)
            stats = torch.zeros(10, dtype=torch.float64, device='cuda')
            whole_model_unconf = unconf_grad and unconf_mode == 'full'
            for i, (p, g_x) in enumerate(zip(params, grad_x_acc)):
                if p.grad is None:
                    continue
                g_s = p.grad - g_x
                terms = [g_x * g_x, g_s * g_s, g_x * g_s]
                if whole_model_unconf:
                    g_uc, g_ui = grad_uc_acc[i], grad_ui_acc[i]
                    terms += [g_uc * g_uc, g_ui * g_ui, g_uc * g_x, g_uc * g_s, g_ui * g_x, g_ui * g_s, g_uc * g_ui]
                stats[:len(terms)] += torch.stack([t.sum(dtype=torch.float64) for t in terms])
            (norm_x_sq, norm_s_sq, dot_xs,
             norm_uc_sq, norm_ui_sq, dot_uc_x, dot_uc_s, dot_ui_x, dot_ui_s, dot_uc_ui) = stats.tolist()
            grad_norm_x, grad_norm_s = norm_x_sq ** 0.5, norm_s_sq ** 0.5
            grad_ratio_s_x = grad_norm_s / max(grad_norm_x, 1e-12)
            grad_cos_x_s = dot_xs / max(grad_norm_x * grad_norm_s, 1e-12)
            if whole_model_unconf:
                grad_norm_uc, grad_norm_ui = norm_uc_sq ** 0.5, norm_ui_sq ** 0.5
                grad_cos_x_uc = dot_uc_x / max(grad_norm_x * grad_norm_uc, 1e-12)
                grad_cos_s_uc = dot_uc_s / max(grad_norm_s * grad_norm_uc, 1e-12)
                grad_cos_x_ui = dot_ui_x / max(grad_norm_x * grad_norm_ui, 1e-12)
                grad_cos_s_ui = dot_ui_s / max(grad_norm_s * grad_norm_ui, 1e-12)
                grad_cos_uc_ui = dot_uc_ui / max(grad_norm_uc * grad_norm_ui, 1e-12)
            grad_cos_total_steps += 1
            grad_cos_conflict_steps += int(grad_cos_x_s < GRAD_COS_CONFLICT_THRESH)
            grad_cos_conflict_pct = 100.0 * grad_cos_conflict_steps / grad_cos_total_steps

            optimizer.step()

            iters = epoch * steps_per_epoch + step
            lr = cfg['lr'] * (1 - iters / total_steps) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            log = {
                'train/loss_all': group_loss,
                'train/loss_x': group_loss_x,
                'train/loss_s': group_loss_s,
                'train/mask_ratio': total_mask_ratio.val,
                'grad/grad_norm_x': grad_norm_x,
                'grad/grad_norm_s': grad_norm_s,
                'grad/grad_norm_s_over_x': grad_ratio_s_x,
                'grad/grad_cos_x_s': grad_cos_x_s,
                'grad/grad_cos_x_s_lt_%g_pct' % GRAD_COS_CONFLICT_THRESH: grad_cos_conflict_pct,
                'iters': iters
            }

            if whole_model_unconf:
                log.update({
                    'grad/grad_norm_unconf_correct': grad_norm_uc,
                    'grad/grad_norm_unconf_incorrect': grad_norm_ui,
                    'grad/grad_cos_x_unconf_correct': grad_cos_x_uc,
                    'grad/grad_cos_s_unconf_correct': grad_cos_s_uc,
                    'grad/grad_cos_x_unconf_incorrect': grad_cos_x_ui,
                    'grad/grad_cos_s_unconf_incorrect': grad_cos_s_ui,
                    'grad/grad_cos_unconf_correct_incorrect': grad_cos_uc_ui
                })
            if class_grad_ema:
                # EMA update for the classes present in this step; the first observation
                # initializes the class's row instead of decaying from zero
                for c, g_sum in cls_grad_sum.items():
                    g_c = g_sum / cls_pix_count[c]
                    if grad_ema_cls_updates[c] == 0:
                        grad_ema_cls[c].copy_(g_c)
                    else:
                        grad_ema_cls[c].mul_(CLASS_GRAD_EMA_BETA).add_(g_c, alpha=1 - CLASS_GRAD_EMA_BETA)
                    grad_ema_cls_updates[c] += 1
                    del g_c
                cls_grad_sum, cls_pix_count = {}, {}
                ema_norms = grad_ema_cls.norm(dim=1).tolist()
                for c, name in enumerate(class_names):
                    if grad_ema_cls_updates[c] > 0:
                        log['grad_ema/norm_%s' % name] = ema_norms[c]
                        log['grad_ema/updates_%s' % name] = grad_ema_cls_updates[c]

                # Tier 1a: row-cosine matrices between a step gradient's classifier rows and the per-class
                # anchor bank. With A[c, k] = row k of anchor a^(c) (the weight part of grad_ema_cls[c]) and
                # G[k] = row k of the gradient's classifier weight, M[k, c] = cos(G[k], A[c, k]):
                #   diagonal  M[k, k]: row k against its own class's 'toward' row;
                #   off-diag  M[k, c]: row k against class c's residual row -- the second sign check.
                # Logged per gradient: mean diagonal, mean over rows of min_{c != k} M[k, c], and the fraction
                # of rows whose off-diagonal minimum is negative. Rows/classes without an initialised anchor
                # or with a zero gradient row are excluded. g_x's own diagonal is a consistency check: it
                # tends to +1 as the labeled set is fitted (residual confusion rho << C), but is negative early,
                # when the near-uniform softmax gives every class ~0.95 confusion mass per pixel and the
                # background pixels' 'push w_k away' terms dominate each mixed row.
                K, d_cls = cfg['nclass'], cls_params[0].shape[1]
                with torch.no_grad():
                    A = grad_ema_cls[:, :K * d_cls].view(K, K, d_cls)
                    A_norm = A.norm(dim=2)                                                   # (c, k)
                    inited = torch.tensor([u > 0 for u in grad_ema_cls_updates], device='cuda')
                    eye = torch.eye(K, dtype=torch.bool, device='cuda')
                    row_sources = [('x', [grad_x_acc[i] for i in cls_idx_in_params]),
                                   ('s', [params[i].grad - grad_x_acc[i] for i in cls_idx_in_params])]
                    if unconf_grad:
                        row_sources += [('unconf_correct', [grad_uc_acc[i] for i in unconf_cls_idx]),
                                        ('unconf_incorrect', [grad_ui_acc[i] for i in unconf_cls_idx])]
                    for src_name, tensors in row_sources:
                        G = torch.cat([t.reshape(-1) for t in tensors])[:K * d_cls].view(K, d_cls)
                        G_norm = G.norm(dim=1)
                        M = torch.einsum('kd,ckd->kc', G, A) / (G_norm[:, None] * A_norm.T).clamp_min(1e-12)
                        row_ok = inited & (G_norm > 0)
                        valid = row_ok[:, None] & inited[None, :]
                        diag = M.diagonal()[row_ok]
                        off_min = M.masked_fill(~valid | eye, float('inf')).min(dim=1).values
                        off_min = off_min[torch.isfinite(off_min)]
                        if diag.numel() > 0:
                            log['grad_cls/rowcos_diag_mean_%s' % src_name] = diag.mean().item()
                        if off_min.numel() > 0:
                            log['grad_cls/rowcos_offmin_mean_%s' % src_name] = off_min.mean().item()
                            log['grad_cls/rowcos_offmin_neg_frac_%s' % src_name] = (off_min < 0).float().mean().item()

            if proto_stats:
                # EMA update of the prototype bank for the classes present in this step's labeled batches
                for c, N_c in proto_acc['N'].items():
                    S_c, R_c, rho_c = proto_acc['S'][c] / N_c, proto_acc['R'][c] / N_c, proto_acc['rho'][c] / N_c
                    if proto_updates[c] == 0:
                        proto_S[c].copy_(S_c); proto_R[c].copy_(R_c); proto_rho[c].copy_(rho_c)
                    else:
                        proto_S[c].mul_(PROTO_EMA_BETA).add_(S_c, alpha=1 - PROTO_EMA_BETA)
                        proto_R[c].mul_(PROTO_EMA_BETA).add_(R_c, alpha=1 - PROTO_EMA_BETA)
                        proto_rho[c].mul_(PROTO_EMA_BETA).add_(rho_c, alpha=1 - PROTO_EMA_BETA)
                    proto_updates[c] += 1
                # per-class residual confusion mass on labeled data (per pixel of class c, summed over k != c):
                # the classes with any signal for the second sign check
                offdiag_conf = (proto_rho - torch.diag(proto_rho.diagonal())).sum(dim=1).tolist()
                for c, name in enumerate(class_names):
                    if proto_updates[c] > 0:
                        log['proto/confusion_%s' % name] = offdiag_conf[c]
                # complement statistics pooled over the step
                if comp_buf['incorrect']:
                    inc = torch.cat(comp_buf['incorrect'])
                    sc_proto, sc_sub, sc_conf = torch.cat(comp_buf['proto']), torch.cat(comp_buf['proto_sub']), torch.cat(comp_buf['conf'])
                    disagree, stud_gt = torch.cat(comp_buf['disagree']), torch.cat(comp_buf['student_eq_gt'])
                    n_inc, n_cor = int(inc.sum()), int((~inc).sum())
                    log['proto/n_unconf_incorrect'] = n_inc
                    log['proto/n_unconf_correct'] = n_cor
                    if n_cor > 0:
                        log['proto/suspect_rate_correct'] = (sc_proto[~inc] > 0).float().mean().item()
                        log['proto/disagree_rate_correct'] = disagree[~inc].float().mean().item()
                    if n_inc > 0:
                        log['proto/suspect_rate_incorrect'] = (sc_proto[inc] > 0).float().mean().item()
                        log['proto/disagree_rate_incorrect'] = disagree[inc].float().mean().item()
                        log['proto/incorrect_student_eq_gt_rate'] = stud_gt[inc].float().mean().item()
                    if n_inc > 0 and n_cor > 0:
                        # AUC of each score as a detector of incorrect pseudo-labels on the complement
                        log['proto/auc_proto'] = rank_auc(sc_proto, inc)
                        sub_fin = torch.isfinite(sc_sub)
                        if sub_fin.any() and inc[sub_fin].any() and (~inc[sub_fin]).any():
                            log['proto/auc_proto_sub'] = rank_auc(sc_sub[sub_fin], inc[sub_fin])
                        log['proto/auc_conf'] = rank_auc(sc_conf, inc)
                        # are the prototype score and confidence complementary? pixel-level correlation of the
                        # two scores, the AUC of their z-scored sum, and the same with student/teacher
                        # disagreement added (a binary score; its own AUC is (TPR + TNR) / 2)
                        z = lambda s: (s - s.mean()) / s.std().clamp_min(1e-12)
                        zp, zc, zd = z(sc_proto.float()), z(sc_conf.float()), z(disagree.float())
                        log['proto/corr_proto_conf'] = (zp * zc).mean().item()
                        log['proto/auc_combo_pc'] = rank_auc(zp + zc, inc)
                        log['proto/auc_combo_pcd'] = rank_auc(zp + zc + zd, inc)
                        log['proto/auc_disagree'] = rank_auc(disagree.float(), inc)
            wandb.log(log)
            # LOG_DEBUG_PREFIXES="grad_cls/,proto/": also print the matching keys of each step's log dict
            # (for --stop-after smoke tests with wandb disabled)
            if log_debug_prefixes:
                logger.info('log[%s] = %s' % (','.join(log_debug_prefixes),
                            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(log.items()) if k.startswith(log_debug_prefixes)}))
            if step % max(steps_per_epoch // 8, 1) == 0:
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss s unconf: {:.3f} '
                            '(correct: {:.3f}, incorrect: {:.3f}), Mask ratio: {:.3f}'.format(
                                step, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                total_loss_s_unconf.avg, total_loss_s_unconf_correct.avg, total_loss_s_unconf_incorrect.avg,
                                total_mask_ratio.avg))

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
            'previous_best': float(previous_best),  # numpy scalar from evaluate(); torch>=2.6 weights_only load rejects it
            'previous_best_ema': float(previous_best_ema),
            'best_epoch': best_epoch,
            'best_epoch_ema': best_epoch_ema,
            'grad_cos_conflict_steps': grad_cos_conflict_steps,
            'grad_cos_total_steps': grad_cos_total_steps
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if class_grad_ema:
            torch.save({'grad_ema_cls': grad_ema_cls.cpu(), 'updates': grad_ema_cls_updates, 'beta': CLASS_GRAD_EMA_BETA},
                       grad_ema_path)
        if proto_stats:
            torch.save({'S': proto_S.cpu(), 'R': proto_R.cpu(), 'rho': proto_rho.cpu(), 'updates': proto_updates,
                        'beta': PROTO_EMA_BETA, 'class_names': list(class_names)}, proto_path)
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        if is_best_ema:
            torch.save(checkpoint, os.path.join(args.save_path, 'best_ema.pth'))

    wandb.finish()


if __name__ == '__main__':
    main()
