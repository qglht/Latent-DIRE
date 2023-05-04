import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # FIXME

    def forward(self, x):
        # FIXME
        return x

def build_cnn(device: str='cpu') -> nn.Module:
    return CNN().to(device)