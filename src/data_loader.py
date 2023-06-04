from typing import Callable, Tuple, Optional
import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder, DatasetFolder

from nn.resnet50 import preprocess_resnet50_pixel, preprocess_resnet50_latent

"""
This assumes we have organized the data in the following way:
    root
    ├── real
    └── fake
or in the following way:
    root
    ├── train
    ├── val
    └── test
We automatically determine the type of data based on the folder structure.
"""


def ldire_loader(path: str) -> torch.Tensor:
    return torch.load(path)


def get_dataloaders(
    root: str,
    model: str,
    type: str,
    batch_size: int,
    num_workers: int = 0,
    split: Optional[Tuple[float, float, float]] = [0.8, 0.1, 0.1],
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # Determine transform
    if model in ["resnet50_latent"]:
        transform = preprocess_resnet50_latent
    elif model in ["resnet50_pixel"]:
        transform = preprocess_resnet50_pixel
    
    # Determine folder structure
    if set({"train", "val", "test"}).issubset(os.listdir(root)):
        if type == "images":
            train_dataset = ImageFolder(f"{root}/train", transform=transform)
            val_dataset = ImageFolder(f"{root}/val", transform=transform)
            test_dataset = ImageFolder(f"{root}/test", transform=transform)
        elif type == "latent":
            train_dataset = DatasetFolder(f"{root}/train", loader=ldire_loader, extensions=(".pt",))
            val_dataset = DatasetFolder(f"{root}/val", loader=ldire_loader, extensions=(".pt",))
            test_dataset = DatasetFolder(f"{root}/test", loader=ldire_loader, extensions=(".pt",))
    else:
        if type == "images":
            dataset = ImageFolder(root, transform=transform)
        elif type == "latent":
            dataset = DatasetFolder(root, loader=ldire_loader, extensions=(".pt",))
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, lengths=split)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader
