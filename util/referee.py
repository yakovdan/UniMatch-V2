r"""k-NN referee on frozen pretrained DINOv2 patch features, and the abstention loss.

Used by ``unimatch_v2_1gpu.py --abstention`` (independent of the backbone lock): a second,
independent labeler for the unlabeled pixels whose errors do not drift with training.

**Referee.** A bank of L2-normalised patch features is built once from the labeled images of the
split (native resolution, no augmentation; only patches whose 14x14 block is >= ``PURITY`` one
class, ignore excluded). At training time the weak view of an unlabeled micro-batch goes through
the frozen pretrained backbone, every patch votes with its ``K`` nearest bank patches (cosine
similarity, similarity-weighted), the per-class score maps are upsampled bilinearly to pixel
resolution and argmaxed. This is exactly the probe in ``feature_probe.py``, which imports its
building blocks from here.

**Abstention loss.** At a pixel where the teacher says $y_1$ and the referee $y_2 \ne y_1$, the two
cross-entropy gradients $p - e_{y_1}$ and $p - e_{y_2}$ agree on every logit except the two
disputed ones. ``abstention_loss_px`` keeps exactly that shared part: with $S = \{y_1, y_2\}$,

$$L = \operatorname{logsumexp}(z') - \operatorname{logsumexp}(\bar z_S), \qquad
  z'_k = z_k \ (k \notin S),\ z'_k = \bar z_k \ (k \in S),$$

where $\bar z$ is ``z.detach()``. Its value is $-\log(p_{y_1} + p_{y_2})$ and its gradient is
$p_k$ on $k \notin S$ and exactly $0$ on both disputed logits: push the rejected classes down,
no opinion between the candidates. Where the labelers agree the caller uses ordinary CE
(this loss with $|S| = 1$ would NOT reduce to CE: it lacks the $p_y - 1$ term on the label).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.semi import SemiDataset
from model.backbone.dinov2 import DINOv2

PATCH = 14
K = 20            # nearest bank patches per query
PURITY = 0.75     # min fraction of one class in a 14x14 block for it to enter the bank
BANK_CHUNK = 131072  # bank patches per similarity chunk (query x chunk fp16 stays well under 1 GB)


def plain_dataset(dataset, data_root, ids):
    """The images in `ids` (a list of id lines, or a path to a file of them), loaded unaugmented at
    full resolution with their GT masks.

    SemiDataset's 'val' MODE is the only code path that loads (image, mask) plainly -- 'train_l'
    applies random crops, flips, blur and color jitter, which would misalign patches and labels.
    The mode pre-fills `ids` from splits/<dataset>/val.txt; that list is replaced here."""
    if isinstance(ids, str):
        ids = [l for l in open(ids).read().splitlines() if l.strip()]
    ds = SemiDataset(dataset, data_root, 'val')
    ds.ids = list(ids)
    return ds


def load_pretrained_backbone(backbone_name, ckpt_path, device='cuda'):
    """The DINOv2 trunk of `backbone_name` (e.g. 'dinov2_small') at the pretrained checkpoint,
    frozen and in eval mode."""
    bb = DINOv2(model_name=backbone_name.split('_')[-1])
    bb.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    bb = bb.to(device).eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    return bb


@torch.no_grad()
def patch_features(bb, x, layer):
    """Unit-norm patch tokens of block `layer` for a batch: (B, h*w, C) fp16, plus the patch grid
    (h, w) and the resolution the backbone actually saw (rounded to a multiple of PATCH)."""
    H, W = x.shape[-2:]
    nh, nw = int(H / PATCH + 0.5) * PATCH, int(W / PATCH + 0.5) * PATCH
    if (nh, nw) != (H, W):
        x = F.interpolate(x, (nh, nw), mode='bilinear', align_corners=True)
    out = bb.get_intermediate_layers(x, n=[layer], reshape=True, norm=True)[0]  # (B, C, h, w)
    B, C, h, w = out.shape
    f = F.normalize(out.permute(0, 2, 3, 1).reshape(B, h * w, C).float(), dim=-1)
    return f.half(), (h, w), (nh, nw)


@torch.no_grad()
def patch_labels(mask, size, nclass, purity):
    """Majority class per patch of a (1, H, W) GT mask resized to `size`; 255 where the block is
    impure or mostly ignore. Returns (h*w,)."""
    m = mask.long()
    m = torch.where(m == 255, torch.full_like(m, nclass), m)
    oh = F.one_hot(m[0], nclass + 1).permute(2, 0, 1).float()[None]
    frac = F.avg_pool2d(F.interpolate(oh, size, mode='nearest'), PATCH)[0]  # (nclass+1, h, w)
    val, lab = frac.max(0)
    lab = torch.where((val >= purity) & (lab < nclass), lab, torch.full_like(lab, 255))
    return lab.reshape(-1)


def build_bank(bb, ids, data_root, dataset, layer, nclass, purity=PURITY, num_workers=4):
    """Feature bank (N, C) fp16 and labels (N,) long, on the backbone's device, from the labeled
    images `ids` at native resolution."""
    device = next(bb.parameters()).device
    feats, labels = [], []
    for img, mask, _ in DataLoader(plain_dataset(dataset, data_root, ids), batch_size=1,
                                   num_workers=num_workers, pin_memory=True):
        f, _, size = patch_features(bb, img.to(device), layer)
        y = patch_labels(mask.to(device), size, nclass, purity)
        keep = y != 255
        feats.append(f[0][keep]); labels.append(y[keep])
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_topk(q, bank_f, bank_y, k, chunk=BANK_CHUNK):
    """The k nearest bank patches of each query q (P, C) fp16: their cosine similarities (P, k)
    fp16, sorted descending, and their labels (P, k); the bank is streamed in chunks so the
    similarity matrix never exceeds P x chunk."""
    top_s = torch.full((q.shape[0], k), -2.0, device=q.device, dtype=torch.half)
    top_y = torch.zeros((q.shape[0], k), device=q.device, dtype=torch.long)
    for lo in range(0, bank_f.shape[0], chunk):
        s = q @ bank_f[lo:lo + chunk].t()
        cs, ci = s.topk(min(k, s.shape[1]), dim=1)
        s_all = torch.cat([top_s, cs], 1)
        y_all = torch.cat([top_y, bank_y[lo:lo + chunk][ci]], 1)
        top_s, idx = s_all.topk(k, dim=1)
        top_y = y_all.gather(1, idx)
    return top_s, top_y


@torch.no_grad()
def knn_scores(q, bank_f, bank_y, k, nclass, chunk=BANK_CHUNK):
    """(P, nclass) similarity-weighted votes of the k nearest bank patches for queries q (P, C)
    fp16 (``knn_topk`` with negative similarities clamped to zero before the class scatter)."""
    top_s, top_y = knn_topk(q, bank_f, bank_y, k, chunk)
    return torch.zeros(q.shape[0], nclass, device=q.device).scatter_add_(1, top_y, top_s.float().clamp_min(0))


class KNNReferee:
    """Frozen-feature k-NN labeler. ``predict(x)`` maps an image batch (B, 3, H, W) to per-pixel
    labels (B, H, W) long, by upsampling the per-class vote maps bilinearly and taking the argmax."""

    def __init__(self, backbone, bank_f, bank_y, nclass, k=K, layer=None):
        self.bb = backbone
        self.bank_f, self.bank_y = bank_f, bank_y
        self.nclass, self.k = nclass, k
        self.layer = len(backbone.blocks) - 1 if layer is None else layer

    @classmethod
    def from_pretrained(cls, backbone_name, ckpt_path, ids, data_root, dataset, nclass,
                        k=K, purity=PURITY, device='cuda', logger=None):
        t0 = time.time()
        bb = load_pretrained_backbone(backbone_name, ckpt_path, device)
        layer = len(bb.blocks) - 1
        bank_f, bank_y = build_bank(bb, ids, data_root, dataset, layer, nclass, purity)
        ref = cls(bb, bank_f, bank_y, nclass, k, layer)
        if logger is not None:
            n_ids = len(ids) if not isinstance(ids, str) else sum(1 for l in open(ids) if l.strip())
            counts = torch.bincount(bank_y, minlength=nclass).tolist()
            logger.info('referee: frozen pretrained %s, block %d, k=%d, purity %.2f; bank of %d patches from %d '
                        'labeled images (%.2f GiB), built in %.0f s; patches per class: %s' % (
                            backbone_name, layer, k, purity, bank_f.shape[0], n_ids,
                            bank_f.numel() * bank_f.element_size() / 2 ** 30, time.time() - t0,
                            ' '.join(str(c) for c in counts)))
        return ref

    @torch.no_grad()
    def predict(self, x):
        B, _, H, W = x.shape
        feats, (h, w), _ = patch_features(self.bb, x, self.layer)
        out = torch.empty(B, H, W, dtype=torch.long, device=x.device)
        for b in range(B):  # one image at a time keeps the P x bank similarity matrix small
            scores = knn_scores(feats[b], self.bank_f, self.bank_y, self.k, self.nclass)
            maps = scores.t().reshape(1, self.nclass, h, w)
            out[b] = F.interpolate(maps, (H, W), mode='bilinear', align_corners=True)[0].argmax(0)
        return out


def abstention_loss_px(logits, y_a, y_b):
    r"""Per-pixel abstention loss (B, H, W) for logits (B, C, H, W) and two label maps (B, H, W).

    Value $-\log(p_{y_a} + p_{y_b})$; gradient $p_k$ on every class outside $\{y_a, y_b\}$ and
    exactly zero on the two disputed logits (see the module docstring). Labels are clamped into
    range so ignore values (255) at masked-out pixels cannot index out of bounds; the caller is
    expected to mask those pixels. Meant for pixels where $y_a \ne y_b$; where they agree use CE.
    """
    C = logits.shape[1]
    sel = torch.zeros_like(logits, dtype=torch.bool)
    sel.scatter_(1, y_a.clamp(0, C - 1).unsqueeze(1), True)
    sel.scatter_(1, y_b.clamp(0, C - 1).unsqueeze(1), True)
    z_det = logits.detach()
    lse_all = torch.logsumexp(torch.where(sel, z_det, logits), dim=1)
    lse_disputed = torch.logsumexp(z_det.masked_fill(~sel, float('-inf')), dim=1)
    return lse_all - lse_disputed
