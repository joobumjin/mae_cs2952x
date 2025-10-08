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
# import timm.optim.optim_factory as optim_factory

import wandb

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.data import get_train_loader, get_test_loader, get_train_loader_dist

import models_mae

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--data_path', default="users/bjoo2/data/bjoo2/mae")
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # Model parameters
    parser.add_argument('--model', default="mae_vit_base_patch16", type=str, metavar='MODEL',
                        help='Name of model to train')

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

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    return parser

# --------------------------------------------------------

import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import utils

class DistributedDataParallel():
    """
    A module implementation of Distributed Data Parallel (DDP)
    
    Attributes:
        - model (nn.Module): the underlying model to be trained
        - device (torch.device): the device the model is located on for this rank
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running DDP
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def broadcast_params(self):
        """
        Broadcasts the underlying model's parameters across all ranks
        """
        # TODO (Task 1.1): Implement!
        for ind, param in enumerate(self.model.parameters()):
            dist.broadcast(param, src=0)

    def average_gradients(self):
        """
        Averages the gradients of all model parameters across all ranks.
        """
        # TODO (Task 1.1): Implement!
        for param in self.model.parameters():
            if param.grad is None: pass
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.world_size

# --------------------------------------------------------

import math
import sys
from typing import Iterable

import util.lr_sched as lr_sched
from datasets import load_dataset
from util.data import t_func


def train_ddp_worker(
    rank: int, model, world_size: int, stats_queue: mp.Queue, args, epoch, 
    cores_per_rank: int = 1, batch_size: int = 32, learning_rate: float = 1e-2,
):
    
    device = torch.device("cuda")
    stats = {
        utils.RANK: rank,
        utils.TRAIN_LOSS: 0.0,
        utils.LR: learning_rate,
    }
    
    utils.debug_print(f"Initializing DDP on rank {rank}.")
    utils.parallel_setup(rank, world_size)
    utils.pin_to_core(rank, cores_per_rank)
    utils.seed_everything(0)

    train = load_dataset("matthieulel/galaxy10_decals", split="train", cache_dir = args.data_path)

    sampler_train = DistributedSampler(train, 
                                       num_replicas=world_size, 
                                       rank=rank, 
                                       shuffle=True)

    transform_train = transforms.Compose([transforms.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    train = train.with_transform(lambda data: t_func(data, transform_train))

    train_loader = DataLoader(
        train, 
        batch_size=batch_size,
        sampler=sampler_train,
        shuffle = True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    # loss_scaler = NativeScaler()

    model = model.to(device)
    model = DistributedDataParallel(model, device)
    model.model.eval()

    # broadcast model
    utils.debug_print(f"Rank {rank}: Broadcasting model to all ranks")
    model.broadcast_params()
    utils.debug_print(f"Rank {rank}: Model finished broadcasting to all ranks")


    for data_iter_step, samples in enumerate(train_loader):
        samples = samples["image"]

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        # with torch.amp.autocast('cuda'):
        loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
        model.average_gradients()
        optimizer.zero_grad()

        torch.cuda.synchronize()


        lr = optimizer.param_groups[0]["lr"]
        stats[utils.TRAIN_LOSS] += loss
        stats[utils.LR] = lr
        stats[utils.NUM_BATCHES] += 1

    # gather the stats from all processes
    # for key in metrics: metrics[key].synchronize_between_processes()
    stats_queue.put(stats)
    utils.parallel_cleanup()

def train_ddp(
    args, epoch, model, world_size: int = 1, batch_size: int = 32,
    learning_rate: float = 1e-2, cores_per_rank: int = 1, 
) -> dict:
    """
    Trains a VGG16 model on the CIFAR-10 dataset using DDP.
    
    Args:
        world_size (int): the number of processes to train with; defaults to 1
        num_batches (int): the number of batches to train for; defaults to 5
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        learning_rate (float): the optimizer's learning rate; defaults to 1e-2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        check_weights (bool): boolean to determine whether weights are saved;
        defaults to False
        check_output (bool): boolean to determine whether the model's output is
        saved; defaults to True
    """
    stats_queue = mp.Queue()
    mp.spawn(
        train_ddp_worker, 
        args=(model, world_size, stats_queue, args, epoch,
            cores_per_rank, batch_size, learning_rate), 
        nprocs=world_size, join=True
    )
    return utils.agg_stats_per_rank(stats_queue)


def main(args):
    # misc.init_distributed_mode(args)
    device = torch.device("cuda")

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    cudnn.benchmark = True    
    
    # model = models_mae.__dict__[args.model](img_size = 256, norm_pix_loss=args.norm_pix_loss)
    # model.to(device)
    
    config = {
        "Model": args.model,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "cache_dir": args.data_path
    }

    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project=f"mae-test", 
        name=f"Base MAE", 
        config=config
    )

    model = models_mae.__dict__[args.model](img_size = 256, norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for epoch in pbar:
        stats = train_ddp(args = args, 
                          epoch = epoch, 
                          model = model,
                          world_size = args.world_size, 
                          batch_size = args.batch_size,
                          learning_rate = args.blr, 
                          cores_per_rank = 1)
        
        stats[utils.TRAIN_LOSS] = stats[utils.TRAIN_LOSS] / stats[utils.NUM_BATCHES] 

        run.log(stats)

        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
