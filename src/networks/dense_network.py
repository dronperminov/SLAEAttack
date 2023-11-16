from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional

from src.networks.network import Network


class DenseNetwork(Network):
    def __init__(self, inputs: int, sizes: List[int], activation_name: str = "relu") -> None:
        super(DenseNetwork, self).__init__(activation_name=activation_name)
        self.inputs = inputs
        self.__init_layers(sizes)

    def __init_layers(self, sizes: List[int]) -> None:
        self.layers = nn.ModuleList()
        inputs = self.inputs

        for layer_size in sizes:
            self.layers.append(nn.Linear(inputs, layer_size))
            inputs = layer_size

    def forward_first_layer(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.layers[0](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_first_layer(x)

        for layer in self.layers[1:]:
            x = layer(self.activation(x))

        return x

    def get_slae(self, x: torch.tensor, to_numpy: bool) -> Tuple[torch.tensor, torch.tensor]:
        output = self.forward_first_layer(torch.unsqueeze(x, 0))[0]
        matrix = self.layers[0].weight.detach()
        b = (output - self.layers[0].bias).detach()

        if to_numpy:
            matrix, b = matrix.cpu().numpy(), b.cpu().numpy()

        return matrix, b
