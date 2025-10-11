import argparse
from typing import Iterable
import os
import time
import numpy as np
from tqdm import trange, tqdm

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
import reconstruction_vis as recon

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--data_path', default="/users/bjoo2/data/bjoo2/mae")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # Model parameters

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--save_path', default="/users/bjoo2/scratch/mae/weights")
    parser.add_argument('--save_file', default="mae_large_scaled_40e")
    
    parser.add_argument('--cache_path', default="/users/bjoo2/scratch/mae/cache")

    parser.add_argument('--fb_weights', action="store_true")

    parser.add_argument('--mean_pool', action="store_true")
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

def build_cache(model: torch.nn.Module, loader: Iterable, device: torch.device, cache_file: str, args):
    model.eval()
    
    cache, cache_labels = [], []
    
    for samples in tqdm(loader, desc="building cache"):
        samples["image"] = samples["image"].to(device)
        samples["label"] = samples["label"].to(device)

        with torch.no_grad():
            embeds, _, _ = model.forward_encoder(samples["image"], 0)
            embeds = embeds[:, 0, :] if args.mean_pool else torch.mean(embeds[:, 1:, :], dim = 1)
            labels = samples["label"]
            cache.append(embeds.detach().cpu())
            cache_labels.append(labels.detach().cpu())

    torch.save({"image": torch.cat(cache, dim=0), "label": torch.cat(cache_labels, dim=0)}, cache_file)

def train_one_epoch(model: torch.nn.Module, probe: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int,
                    device: torch.device, cache_file: str, args):
    model.eval()
    probe.train(True)

    optimizer.zero_grad()

    metrics = {"Train Loss": misc.SmoothedValue(), "Train Accuracy": misc.SmoothedValue(), "lr": misc.SmoothedValue()}

    cached = os.path.exists(cache_file)
    if cached:
        load = torch.load(cache_file)
        cache = load["image"]
        cache_labels = load["label"]
    else: 
        print("Building Cache")
        cache, cache_labels = [], []

    for ind, samples in enumerate(data_loader):
        if not cached: 
            samples["image"] = samples["image"].to(device)
            samples["label"] = samples["label"].to(device)

        lr_sched.adjust_learning_rate(optimizer, ind / len(data_loader) + epoch, args)

        # with torch.no_grad():
        if not cached: 
            embeds, _, _ = model.forward_encoder(samples["image"], 0)
            embeds = embeds[:, 0, :] if args.mean_pool else torch.mean(embeds[:, 1:, :], dim = 1)
            labels = samples["label"]
            cache.append(embeds.detach().cpu())
            cache_labels.append(labels.detach().cpu())
        else:
            embeds = cache[ind*data_loader.batch_size:(ind+1)*data_loader.batch_size].to(device)
            labels = cache_labels[ind*data_loader.batch_size:(ind+1)*data_loader.batch_size].to(device)

        loss, preds = probe(embeds, labels)

        loss.backward()
        optimizer.step()

        correct_preds = torch.sum(torch.argmax(preds.detach(), dim=-1) == labels).item()

        metrics["Train Loss"].update(loss.item(), n = len(labels))
        metrics["Train Accuracy"].update(correct_preds, n = len(labels))

        lr = optimizer.param_groups[0]["lr"]
        metrics["lr"].update(lr)

    if not cached: torch.save({"image": torch.cat(cache, dim=0), "label": torch.cat(cache_labels, dim=0)}, cache_file)

    return {k: meter.global_avg for k, meter in metrics.items()}

def test(model: torch.nn.Module, probe: torch.nn.Module, data_loader: Iterable, device: torch.device, cache_file: str, args):
    model.eval()
    probe.eval()

    metrics = {"Test Loss": misc.SmoothedValue(), "Test Accuracy": misc.SmoothedValue()}

    cached = os.path.exists(cache_file)
    if cached:
        load = torch.load(cache_file)
        cache = load["image"]
        cache_labels = load["label"]
    else: cache, cache_labels = [], []

    for ind, samples in enumerate(data_loader):
        if not cached: 
            samples["image"] = samples["image"].to(device)
            samples["label"] = samples["label"].to(device)

        with torch.no_grad():
            if not cached: 
                embeds, _, _ = model.forward_encoder(samples["image"], 0)
                embeds = embeds[:, 0, :] if args.mean_pool else torch.mean(embeds[:, 1:, :], dim = 1)
                labels = samples["label"]
                cache.append(embeds.detach().cpu())
                cache_labels.append(labels.detach().cpu())
            else:
                embeds = cache[ind*data_loader.batch_size:(ind+1)*data_loader.batch_size].to(device)
                labels = cache_labels[ind*data_loader.batch_size:(ind+1)*data_loader.batch_size].to(device)
            loss, preds = probe(embeds, labels)

        correct_preds = torch.sum(torch.argmax(preds.detach(), dim=-1) == labels).item()

        metrics["Test Loss"].update(loss.item(), n = len(labels))
        metrics["Test Accuracy"].update(correct_preds, n = len(labels))

    if not cached: torch.save({"image": torch.cat(cache, dim=0), "label": torch.cat(cache_labels, dim=0)}, cache_file)

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
    if args.fb_weights: loader_args["img_size"] = 224

    train_loader = get_train_loader(**loader_args)
    test_loader = get_test_loader(**loader_args)


    for param in model.parameters(): param.requires_grad = False
    model.to(device)

    input_shapes = {"mae_vit_base_patch16": 768, "mae_vit_large_patch16": 1024, "mae_vit_huge_patch14": 1280}

    probe_args = {
        "in_dim": input_shapes[model_args["size"]],
        "out_dim": 10,
        "num_layers": 1, # trial.suggest_int("probe layers", 1, 3),
        "moco_init": 1, #trial.suggest_int("mocov3-esque init", 0, 1),
        "pre_bn": 1, #trial.suggest_int("pre_batchnorm", 0, 1),
    }
    probe = models_mae.LinearProbe(**probe_args)
    for param in probe.parameters(): param.requires_grad = True
    probe.to(device)

    opts = {
        "AdamW": (torch.optim.AdamW, {"lr": 1e-3, "betas": (0.9, 0.95)}),
        "LARS": (LARS, {"lr": 0.1, "weight_decay": args.weight_decay}),
        "SGD": (torch.optim.SGD, {"lr": 0.01, "weight_decay": args.weight_decay})
    }

    opt_args = {
        "optimizer": "AdamW", #trial.suggest_categorical("optimizer type", opts.keys())
    }


    config = {
        **opt_args,
        **loader_args, 
        **probe_args,
        **model_args
    }

    (opt_class, opt_kwargs) = opts[opt_args["optimizer"]]
    args.lr = opt_kwargs["lr"]
    args.min_lr = 0.0
    optimizer = opt_class(probe.parameters(), **opt_kwargs)

    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project=f"MAE FineTune", 
        name=f"ViTMAE, {probe_args["num_layers"]}Dense, {opt_args["optimizer"]}, WD", 
        config=config
    )

    os.makedirs(args.cache_path, exist_ok=True)
    pooled = "_meanpooled" if args.mean_pool else ""
    train_cache_file = f"{args.cache_path}/{args.save_file}_train{pooled}"
    test_cache_file = f"{args.cache_path}/{args.save_file}_test{pooled}"
    
    cached = os.path.exists(train_cache_file)
    if not cached:
        build_cache(model, train_loader, device, train_cache_file, args)
        build_cache(model, test_loader, device, test_cache_file, args)

    pbar = trange(0, args.epochs, desc="Probe Training Epochs", postfix={})
    for epoch in pbar:
        train_stats = train_one_epoch(model, probe, 
                                      train_loader, optimizer, epoch,
                                      device, train_cache_file, args)
        test_stats = test(model, probe, test_loader, device, test_cache_file, args)

        postfix = {**train_stats, **test_stats}
        run.log(postfix)
        pbar.set_postfix(postfix)

    return test_stats['Test Accuracy']


def main(args):
    if not args.fb_weights:
        model, model_args = load_model(f"{args.save_path}/{args.save_file}")
    else:
        model = recon.prepare_model(f"{args.save_path}/{args.save_file}", 'mae_vit_large_patch16')
        model_args = {"size": "mae_vit_large_patch16"}

    # study = optuna.create_study(study_name=f"mae_probe", direction="minimize")
    # study.set_metric_names(["Test Accuracy"])

    # study.optimize(lambda trial: objective(trial, args, model, model_args), n_trials=15)
    # print(f"Best value: {study.best_value} (params: {study.best_params})")
    _ = objective(None, args, model, model_args)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
 