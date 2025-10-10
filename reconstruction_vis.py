import sys
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from util.data import get_train_loader, get_test_loader


import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def load_model(save_fp):
    checkpoint = torch.load(save_fp, weights_only=True)

    model_str = checkpoint["model_str"]
    model_class = models_mae.__dict__[model_str]

    model_args = checkpoint["model_args"]
    model = model_class(**model_args)

    model.load_state_dict(checkpoint["model_state_dict"])

    model_args["size"] = model_str
    
    return model, model_args


def run_one_image(img, model, split="train"):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    _, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.savefig(f"/users/bjoo2/scratch/mae/vis/large_pretrain_{split}.png")

    plt.close()

def main():
    # load an image
    loader_args = {
        "batch_size": 1,
        "cache_dir": "/users/bjoo2/data/bjoo2/mae"
    }
    train_loader = get_train_loader(**loader_args)
    test_loader = get_test_loader(**loader_args)

    train_image = next(iter(train_loader))["image"][0].permute(1,2,0)
    test_image = next(iter(test_loader))["image"][0].permute(1,2,0)
    print(train_image.shape)

    assert train_image.shape == (256, 256, 3)
    assert test_image.shape == (256, 256, 3)

    # normalize by ImageNet mean and std
    train_image = train_image - imagenet_mean
    train_image = train_image / imagenet_std

    test_image = test_image - imagenet_mean
    test_image = test_image / imagenet_std

    model, model_args = load_model(f"/users/bjoo2/scratch/mae/weights/mae_large_scaled_40e")

    run_one_image(train_image, model, "train")
    run_one_image(test_image, model, "test")

if __name__ == '__main__':
    main()