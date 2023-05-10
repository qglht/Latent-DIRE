from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from nn.resnet50 import preprocess_resnet50_pixel


def get_dataset(root):
    return ImageFolder(root, transform=preprocess_resnet50_pixel)


def get_dataloader(root, batch_size: int = 32, shuffle: bool = True):
    return DataLoader(get_dataset(root), batch_size=batch_size, shuffle=shuffle)
