import argparse
from typing import Iterable
import os
import numpy as np
from tqdm import trange

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import wandb
import optuna

import util.misc as misc
import util.lr_sched as lr_sched
from util.lars import LARS
from util.data import get_train_loader, get_test_loader

import models_mae

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--data_path', default="users/bjoo2/data/bjoo2/mae")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # Model parameters

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--save_path', default="users/bjoo2/data/bjoo2/mae/weights")
    parser.add_argument('--save_file', default="mae_large_scaled_50e")
    
    parser.add_argument('--cache_path', default="users/bjoo2/data/bjoo2/mae/cache")
    return parser

def load_model(save_fp):
    checkpoint = torch.load(save_fp, weights_only=True)

    model_str = checkpoint["model_str"]
    model_class = models_mae.__dict__[model_str]

    model_args = checkpoint["model_args"]
    model = model_class(**model_args)

    model.load_state_dict(checkpoint["model_state_dict"])

    model_args["size"] = model_str
    
    return model, model_args


def train_one_epoch(model: torch.nn.Module, probe: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, cache_file: str):
    model.eval()
    probe.train()

    optimizer.zero_grad()

    metrics = {"Train Loss": misc.SmoothedValue(), "Train Accuracy": misc.SmoothedValue(), "lr": misc.SmoothedValue()}

    cached = os.path.exists(cache_file)
    cache = torch.load(cache_file) if cached else []

    
    for ind, samples in enumerate(data_loader):
        samples["image"] = samples["image"].to(device)
        samples["label"] =samples["label"].to(device)

        with torch.no_grad():
            if not cached: 
                embeds, _, _ = model.forward_encoder(samples["image"], 0)
                embeds = embeds[:, 0, :]
                cache.append(embeds)
            else:
                embeds = cache[ind].to(device)

        loss, preds = probe(embeds, samples["label"])

        loss.backward()
        optimizer.zero_grad()

        correct_preds = torch.sum(torch.argmax(preds.detach(), dim=-1) == samples["label"])

        metrics["Train Loss"].update(loss.item())
        metrics["Train Accuracy"].update(correct_preds, n = len(samples["label"]))

        lr = optimizer.param_groups[0]["lr"]
        metrics["lr"].update(lr)

    if not cached: torch.save(torch.stack(cache, dim = 0), cache_file)

    return {k: meter.global_avg for k, meter in metrics.items()}

def test(model: torch.nn.Module, probe: torch.nn.Module, data_loader: Iterable, device: torch.device, cache_file: str):
    
    model.eval()
    probe.eval()

    metrics = {"Test Loss": misc.SmoothedValue(), "Test Accuracy": misc.SmoothedValue()}

    cached = os.path.exists(cache_file)
    cache = torch.load(cache_file) if cached else []

    for ind, samples in enumerate(data_loader):
        samples["image"] = samples["image"].to(device)
        samples["label"] = samples["label"].to(device)


        with torch.no_grad():
            if not cached: 
                embeds, _, _ = model.forward_encoder(samples["image"], 0)
                embeds = embeds[:, 0, :]
                cache.append(embeds)
            else:
                embeds = cache[ind].to(device)
            loss, preds = probe(embeds, samples["label"])

        metrics["Test Loss"].update(loss.item())

        correct_preds = torch.sum(torch.argmax(preds.detach(), dim=-1) == samples["label"])
        metrics["Test Accuracy"].update(correct_preds, n = len(samples["label"]))

    if not cached: torch.save(torch.stack(cache, dim = 0), cache_file)

    return {k: meter.global_avg for k, meter in metrics.items()}


def objective(trial, args, model, model_args):
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

    for param in model.parameters(): param.requires_grad = False
    model.to(device)

    input_shapes = {"mae_vit_base_patch16": 768, "mae_vit_large_patch16": 1024, "mae_vit_huge_patch14": 1280}

    probe_args = {
        "in_dim": input_shapes[model_args["size"]],
        "out_dim": 10,
        "num_layers": 3, # trial.suggest_int("probe layers", 1, 3),
        "moco_init": 0, #trial.suggest_int("mocov3-esque init", 0, 1),
        "pre_bn": 0, #trial.suggest_int("pre_batchnorm", 0, 1),
    }
    probe = models_mae.LinearProbe(**probe_args)
    probe.to(device)

    opts = {
        "AdamW": (torch.optim.AdamW, {"betas": (0.9, 0.95)}),
        "LARS": (LARS, {"weight_decay": args.weight_decay})
    }

    opt_args = {
        "lr": 1e-3, # trial.suggest_float("learning_rate", 1e-4, 3e-3, step=1e-4),
        "optimizer": "AdamW", #trial.suggest_categorical("optimizer type", opts.keys())
    }

    config = {
        **opt_args,
        **loader_args, 
        **probe_args,
        **model_args
    }

    (opt_class, misc_args) = opts[opt_args["optimizer"]]
    optimizer = opt_class(probe.parameters(), lr=opt_args["lr"], **misc_args)

    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project=f"MAE FineTune", 
        name=f"Test MAE - {model_args["size"]} ViT, {opt_args["optimizer"]}", 
        config=config
    )

    os.makedirs(args.cache_path, exist_ok=True)
    train_cache_file = f"{args.cache_path}/{args.save_file}_train"
    test_cache_file = f"{args.cache_path}/{args.save_file}_test"

    pbar = trange(0, args.epochs, desc="Probe Training Epochs", postfix={})
    for epoch in pbar:
        train_stats = train_one_epoch(model, probe, 
                                      train_loader, optimizer, 
                                      device, train_cache_file)
        test_stats = test(model, probe, test_loader, device, test_cache_file)

        postfix = {**train_stats, **test_stats}
        run.log(postfix)
        pbar.set_postfix(postfix)

    return test_stats['Test Accuracy']


def main(args):
    model, model_args = load_model(f"{args.save_path}/{args.save_file}")

    # study = optuna.create_study(study_name=f"mae_probe", direction="minimize")
    # study.set_metric_names(["Test Accuracy"])

    # study.optimize(lambda trial: objective(trial, args, model, model_args), n_trials=15)
    # print(f"Best value: {study.best_value} (params: {study.best_params})")
    _ = objective(None, args, model, model_args)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
 