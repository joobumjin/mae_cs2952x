import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from tqdm import trange

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

import wandb

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.data import get_train_loader, get_test_loader

import models_mae

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--data_path', default="users/bjoo2/data/bjoo2/mae")
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # Model parameters
    parser.add_argument('--model', default="mae_vit_base_patch16", type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    return parser

# --------------------------------------------------------
import math
import sys
from typing import Iterable

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    model.train(True)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    metrics = {"Train Loss": misc.SmoothedValue(), "lr": misc.SmoothedValue()}

    for data_iter_step, samples in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metrics["Train Loss"].update(loss)

        lr = optimizer.param_groups[0]["lr"]
        metrics["lr"].update(lr)

    # gather the stats from all processes
    # for key in metrics: metrics[key].synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metrics.items()}

# --------------------------------------------------------

def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    loader_args = {
        "batch_size": args.batch_size,
        "cache_dir": args.data_path
    }
    train_loader = get_train_loader(**loader_args)
    # test_loader = get_test_loader(**loader_args)
    
    
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    config = {
        "Model": args.model,
        "lr": args.lr,
        **loader_args
    }

    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project=f"mae-test", 
        name=f"Base MAE", 
        config=config
    )

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for epoch in pbar:
        train_stats = train_one_epoch(model, train_loader,
                                      optimizer, device, epoch, loss_scaler,
                                      args=args)
        
        run.log(train_stats)

        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
