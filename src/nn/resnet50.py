import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, ResNet
import torchvision.transforms.functional as F


def build_resnet50_pixel(pretrained: bool = True) -> ResNet:
    """ResNet50 with custom classifier for testing normal DIRE"""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
    return model


def preprocess_resnet50_pixel(img):
    weights = ResNet50_Weights.DEFAULT
    batch = weights.transforms()(img)
    return batch


def build_resnet50_latent(device: str = "cpu", pretrained: bool = True, kernel_size: int = 3) -> ResNet:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=1, padding=2, bias=False)
    model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
    return model


def preprocess_resnet50_latent(img):
    if not isinstance(img, torch.Tensor):
        img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float)
    return img
