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

def get_train_loader(batch_size, cache_dir = "", hard_aug = False, img_size = 256, dist=False):
    
    train = load_dataset("matthieulel/galaxy10_decals", split="train", cache_dir = cache_dir)

    if hard_aug:
        transform_train = transforms.Compose([transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                                              transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                                              transforms.RandomRotation(degrees=(0, 180)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform_train = transforms.Compose([transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train = train.with_transform(lambda data: t_func(data, transform_train))
    sampler = DistributedSampler(train) if dist else None
    samplers = [sampler] if dist else []

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    
    return train_loader, *samplers

def get_test_loader(batch_size, cache_dir = "", img_size = 256, dist=False):
    test = load_dataset("matthieulel/galaxy10_decals", split="test", cache_dir = cache_dir)
    
    transformations = [transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    if img_size != 256: transformations = [transforms.RandomCrop(size=img_size)] + transformations

    transform_test = transforms.Compose(transformations)

    test = test.with_transform(lambda data: t_func(data, transform_test))
    sampler = DistributedSampler(test, shuffle=False) if dist else None
    samplers = [sampler] if dist else []

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, sampler=sampler)
    return test_loader, *samplers