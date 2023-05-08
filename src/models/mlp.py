import torch
from torch import nn
import numpy as np
from typing import Optional, List, Callable, Union


class MLP(nn.Module):
    def __init__(
        self,
        layers: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dropout: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = []
        for layer_i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[layer_i], layers[layer_i + 1]))
            if norm_layer is not None:
                self.layers.append(norm_layer(layers[layer_i + 1]))
            self.layers.append(activation_layer(inplace=True))
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.model(x)

def build_mlp(
    input_shape: Optional[Union[int, List[int]]] = None,
    layers: Optional[List[int]] = None,
    device: str = 'cpu',
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    dropout: Optional[float] = 0.0,
) -> nn.Module:
    if not layers:
        input_shape = 3 * 64 * 64
        layers = [input_shape, 1024, 256, 2]
    return MLP(layers, norm_layer, activation_layer, dropout).to(device)
