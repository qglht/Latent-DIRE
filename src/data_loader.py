from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from nn.resnet50 import preprocess_resnet50_pixel

"""
This assumes we have organized the data in the following way:
    root
    ├── real
    └── fake
"""


def get_dataloaders(
    root: str, batch_size: int, num_workers: int = 0, shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = ImageFolder(root, transform=preprocess_resnet50_pixel)
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[0.8, 0.1, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader
