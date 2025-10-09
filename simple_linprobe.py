import argparse
import math
import sys
from typing import Iterable
import numpy as np
from tqdm import trange

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
# import timm.optim.optim_factory as optim_factory

import wandb
import optuna

import util.misc as misc
import util.lr_sched as lr_sched
from util.data import get_train_loader, get_test_loader

import models_mae

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_path', default="users/bjoo2/data/bjoo2/mae")
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

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    
    parser.add_argument('--save_path', default="users/bjoo2/data/bjoo2/mae/weights")
    parser.add_argument('--save_file', default="mae_large_scaled_50e")
    return parser

def load_model(save_fp):
    checkpoint = torch.load(save_fp, weights_only=True)

    model_str = checkpoint["model_str"]
    model_class = models_mae.__dict__[model_str]

    model_args = checkpoint["model_args"]
    model = model_class(**model_args)

    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, model_args


def train_one_epoch(model: torch.nn.Module, probe: torch.nn.module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device):
    model.eval()
    probe.train()

    optimizer.zero_grad()

    metrics = {"Train Loss": misc.SmoothedValue(), "Train Accuracy": misc.SmoothedValue(), "lr": misc.SmoothedValue()}

    for samples in data_loader:
        samples["image"] = samples["image"].to(device)
        samples["label"] = samples["label"].to(device)

        embeds, _, _ = model.forward_encoder(samples["image"], 0)
        loss, preds = probe(embeds, samples["label"])

        loss.backward()
        optimizer.zero_grad()

        correct_preds = torch.sum(torch.argmax(preds.item(), dim=-1) == samples["label"])

        metrics["Train Loss"].update(loss.item)
        metrics["Train Accuracy"].update(correct_preds)

        lr = optimizer.param_groups[0]["lr"]
        metrics["lr"].update(lr)

    return {k: meter.global_avg for k, meter in metrics.items()}

def test(model: torch.nn.Module, probe: torch.nn.Module, data_loader: Iterable, device: torch.device):
    
    model.eval()
    probe.eval()

    metrics = {"Test Loss": misc.SmoothedValue(), "Test Accuracy": misc.SmoothedValue()}

    for samples in data_loader:
        samples["image"] = samples["image"].to(device)
        samples["label"] = samples["label"].to(device)


        with torch.no_grad():
            embeds, _, _ = model.forward_encoder(samples["image"], 0)
            loss, preds = probe(embeds, samples["label"])

        metrics["Test Loss"].update(loss.item())

        correct_preds = torch.sum(torch.argmax(preds.item(), dim=-1) == samples["label"])
        metrics["Train Accuracy"].update(correct_preds)

    return {k: meter.global_avg for k, meter in metrics.items()}


def objective(trial, args, model):
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
    test_loader = get_test_loader(**loader_args)

    model.to(device)

    probe = models_mae.LinearProbe(10)
    probe.to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, betas=(0.9, 0.95))
    
    config = {
        "Model": args.model,
        "lr": args.lr,
        **loader_args
    }

    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project=f"MAE FineTune", 
        name=f"Test MAE - {args.model} ViT", 
        config=config
    )

    pbar = trange(0, args.epochs, desc="Probe Training Epochs", postfix={})
    for _ in pbar:
        train_stats = train_one_epoch(model, train_loader,
                                      optimizer, device)
        test_stats = test(model, test_loader, device)

        postfix = {**train_stats, **test_stats}
        run.log(postfix)
        pbar.set_postfix(postfix)

    return test_stats['Test Accuracy']


def main(args):
    model = load_model(f"{args.save_path}/{args.save_file}")

    # study = optuna.create_study(study_name=f"mae_probe", direction="minimize")
    # study.set_metric_names(["RMSE"])

    # study.optimize(lambda trial: objective(trial, args, model), n_trials=10)
    # print(f"Best value: {study.best_value} (params: {study.best_params})")
    _ = objective(None, args, model)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
 