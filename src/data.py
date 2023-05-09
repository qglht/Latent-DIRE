import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from typing import Tuple

class DIREDataset(Dataset):
    def __init__(self, root: str, transform: Compose = None):
        self.dataset = ImageFolder(root, transform=transform)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        dire, label = self.dataset[index]
        return dire, label

    def __len__(self) -> int:
        return len(self.dataset)

def load_data():
    pass