import argparse

import torch
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Evaluate a UniMatch V2 checkpoint on the val set')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--data-root', type=str, required=True)
parser.add_argument('--backbone', type=str, default=None, help='override the backbone in the config, e.g. dinov2_base')
parser.add_argument('--limit', type=int, default=None, help='evaluate only the first N val images')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg['data_root'] = args.data_root
    if args.backbone is not None:
        cfg['backbone'] = args.backbone

    rank, world_size = setup_distributed()
    assert world_size == 1, 'run with torchrun --nproc_per_node=1'

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})

    try:
        state_dict = torch.load(args.checkpoint, map_location='cpu')
    except Exception:
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.cuda()

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    if args.limit is not None:
        valset.ids = valset.ids[:args.limit]
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2, drop_last=False)

    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)

    if rank == 0:
        for cls_idx, iou in enumerate(iou_class):
            print('Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        print('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}'.format(eval_mode, mIoU))


if __name__ == '__main__':
    main()
