import os
import argparse
import math
import sys
from typing import Iterable
import numpy as np
from tqdm import trange

import torch
import torch.distributed as dist 
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
# from torch.distributed.optim import DistributedOptimizer


import timm
# import timm.optim.optim_factory as optim_factory

import wandb

import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.data import get_train_loader, get_test_loader

import models_mae

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_path', default="/users/bjoo2/data/bjoo2/mae",
                        help="directory to which the data will be downloaded")
    
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument("--nnodes", type=int)

    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # Model parameters
    parser.add_argument('--model', default="base", type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--save_path', default="/users/bjoo2/scratch/mae/weights",
                        help="directory to which the pretrained model weights should be saved")

    parser.add_argument('--disable_wandb', action="store_true")
    return parser

# --------------------------------------------------------

# def setup(): 
#   """Initialize the process group for distributed training. """
#   dist.init_process_group(backend="nccl", init_method="env://") 
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
#     # such as CUDA, MPS, MTIA, or XPU.
#     acc = torch.accelerator.current_accelerator()
#     backend = torch.distributed.get_default_backend_for_device(acc)
#     # initialize the process group
#     dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup(): 
  """Destroy the process group after training is complete. """
  dist.destroy_process_group() 


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, sampler, optimizer: torch.optim.Optimizer,
                    device_id: int, epoch: int, loss_scaler,
                    args=None):
    model.train(True)

    optimizer.zero_grad()

    metrics = {"Train Loss": misc.SmoothedValue(), "lr": misc.SmoothedValue()}

    for data_iter_step, samples in enumerate(data_loader):
        sampler.set_epoch(epoch)
        samples = samples["image"]

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = samples.to(device_id, non_blocking=True)
        samples = samples.to(device_id)

        with torch.amp.autocast('cuda'):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)

        optimizer.zero_grad()

        torch.cuda.synchronize()

        metrics["Train Loss"].update(loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metrics["lr"].update(lr)

    # gather the stats from all processes
    for _, meter in metrics.items():
        meter.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metrics.items()}

def test(model: torch.nn.Module, data_loader: Iterable, sampler, device_id: int, args=None):
    
    model.eval()

    metrics = {"Test Loss": misc.SmoothedValue()}

    for samples in data_loader:
        samples = samples["image"]
        # samples = samples.to(device_id, non_blocking=True)
        samples = samples.to(device_id)

        with torch.no_grad():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        metrics["Test Loss"].update(loss.item())

    # gather the stats from all processes
    for _, meter in metrics.items():
        meter.synchronize_between_processes()
        
    return {k: meter.global_avg for k, meter in metrics.items()}

# --------------------------------------------------------

def main(args):
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    
    model_args = {
        "img_size": 256,
        "norm_pix_loss":args.norm_pix_loss
    }

    model_dict = {
        "base": "mae_vit_base_patch16",
        "large": "mae_vit_large_patch16",
        "huge": "mae_vit_huge_patch14"
    }

    model = models_mae.__dict__[model_dict[args.model]](**model_args)

    # create model and move it to GPU with id rank
    device_id = rank % torch.accelerator.device_count()
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    loader_args = {
        "batch_size": args.batch_size,
        "cache_dir": args.data_path,
        "dist": True
    }
    train_loader, train_sampler = get_train_loader(hard_aug = False, **loader_args)
    test_loader, test_sampler = get_test_loader(**loader_args)

    config = {
        "Model": args.model,
        "lr": args.lr,
        **loader_args
    }

    if args.disable_wandb: run = None
    else:
        if rank == 0:
            run = wandb.init(
                entity="bumjin_joo-brown-university", 
                project=f"MAE DDP", 
                name=f"MAE DDP Test", 
                config=config
            )

    print(f"Start training for {args.epochs} epochs")
    pbar = trange(0, args.epochs, desc="Training Epochs", postfix={})
    for epoch in pbar:
        train_stats = train_one_epoch(model, train_loader, train_sampler,
                                      optimizer, device_id, epoch, loss_scaler,
                                      args=args)
        test_stats = test(model, test_loader, test_sampler, device_id, args=args)

        postfix = {**train_stats, **test_stats}
        if rank == 0 and run is not None: run.log(postfix)
        pbar.set_postfix(postfix)

    # torch.save({"model_str": model_dict[args.model],
    #             "model_args": model_args,
    #             "model_state_dict": model.state_dict()},
    #             f"{args.save_path}/mae_{args.model}_aug_{args.epochs}e")

    cleanup()
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
