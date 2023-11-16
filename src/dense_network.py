from typing import List

import torch
import torch.nn as nn
import torch.nn.functional


class DenseNetwork(nn.Module):
    def __init__(self, inputs: int, sizes: List[int], activation_name: str = "relu") -> None:
        super(DenseNetwork, self).__init__()
        self.inputs = inputs
        self.activation_name = activation_name
        self.activation = self.__get_activation(activation_name)
        self.__init_layers(sizes)

    def __init_layers(self, sizes: List[int]) -> None:
        self.layers = nn.ModuleList()
        inputs = self.inputs

        for layer_size in sizes:
            self.layers.append(nn.Linear(inputs, layer_size))
            inputs = layer_size

    def __get_activation(self, activation: str) -> torch.nn.functional:
        if activation == "relu":
            return torch.nn.functional.relu

        if activation == "leaky-relu":
            return torch.nn.functional.leaky_relu

        if activation == "abs":
            return torch.abs

        raise ValueError(f"Invalid activation '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))