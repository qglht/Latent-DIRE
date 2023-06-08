import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, ResNet
import torchvision.transforms.functional as F


def build_resnet50_pixel(freeze_backbone: bool = True) -> ResNet:
    """ResNet50 with custom classifier for testing normal DIRE"""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = not freeze_backbone
    model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
    return model


def preprocess_resnet50_pixel(img):
    weights = ResNet50_Weights.DEFAULT
    batch = weights.transforms()(img)
    return batch


def build_resnet50_latent(freeze_backbone: bool = True, kernel_size: int = 3) -> ResNet:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = not freeze_backbone
    model.conv1 = nn.Conv2d(4, 64, kernel_size=kernel_size, stride=1, padding=2, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
    return model


def preprocess_resnet50_latent(img):
    img = torch.from_numpy(img)
    img = F.convert_image_dtype(img, torch.float)
    return img
