import torch
from torch import nn
import numpy as np
from typing import Optional, List, Callable, Union


class MLP(nn.Module):
    def __init__(
        self,
        device: str,
        layers: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dropout: Optional[float] = 0.0,
    ) -> None:
        super.__init__()
        self.layers = []
        for layer_i in range(len(layers) - 2):
            layers.append(nn.Linear(layers[layer_i], layers[layer_i + 1]))
            if norm_layer is not None:
                layers.append(norm_layer(layers[layer_i + 1]))
            layers.append(activation_layer())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(layers[-2], layers[-1]))
        layers.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)


class ResNet50:
    def __init__(self, device) -> None:
        resnet50 = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_resnet50", pretrained=True)
        # Where is it stored. Need to load again?
        utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_convnets_processing_utils")
        resnet50.eval().to(device)
        self.model = resnet50

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_classifier(model_name: str, layers: Optional[List], device: str) -> nn.Module:
    if model_name == "resnet50":
        return ResNet50(device)
    elif model_name == "mlp":
        return MLP(device, layers)
    else:
        raise ValueError("Invalid model name")
