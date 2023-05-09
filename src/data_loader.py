from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from models.resnet50 import preprocess_resnet50_pixel


def get_dataset(root):
    return ImageFolder(root, transform=preprocess_resnet50_pixel)


def get_dataloader(root, batch_size: int = 32, shuffle: bool = True):
    return DataLoader(get_dataset(root), batch_size=batch_size, shuffle=shuffle)
