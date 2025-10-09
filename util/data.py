# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch
from datasets import load_dataset

import util.misc as misc


def t_func(data, transformation):
    data["image"] = [transformation(sample) for sample in data["image"]]
    return data

def collate(data):
    images = []
    labels = []
    for example in data:
        images.append((example["images"]))
        labels.append(example["labels"])
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return {"images": images, "labels": labels}

def get_train_loader(batch_size, cache_dir = ""):
    
    train = load_dataset("matthieulel/galaxy10_decals", split="train", cache_dir = cache_dir)

    transform_train = transforms.Compose([transforms.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    train = train.with_transform(lambda data: t_func(data, transform_train))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    
    return train_loader

def get_train_loader_dist(batch_size, world_size, rank, cache_dir = ""):
    
    train = load_dataset("matthieulel/galaxy10_decals", split="train", cache_dir = cache_dir)

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
    
    return train_loader

def get_test_loader(batch_size, cache_dir = ""):
    test = load_dataset("matthieulel/galaxy10_decals", split="test", cache_dir = cache_dir)

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    test = test.with_transform(lambda data: t_func(data, transform_test))


    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return test_loader 

# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     root = os.path.join(args.data_path, 'train' if is_train else 'val')
#     dataset = datasets.ImageFolder(root, transform=transform)

#     print(dataset)

#     return dataset


# def build_transform(is_train, args):
#     mean = IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_DEFAULT_STD
#     # train transform
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation='bicubic',
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         return transform

#     # eval transform
#     t = []
#     if args.input_size <= 224:
#         crop_pct = 224 / 256
#     else:
#         crop_pct = 1.0
#     size = int(args.input_size / crop_pct)
#     t.append(
#         transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
#     )
#     t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)
