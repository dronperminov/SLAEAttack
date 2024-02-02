from typing import List, Tuple

import numpy as np
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

    def __get_scales(self, signs: torch.tensor) -> torch.tensor:
        scales = torch.ones_like(signs, device=signs.device)

        if self.activation_name == "relu":
            scales[signs == -1] = 0
            scales[signs == 1] = 1
        elif self.activation_name == "abs":
            scales[signs == -1] = -1
            scales[signs == 1] = 1
        elif self.activation_name == "leaky-relu":
            scales[signs == -1] = 0.01
            scales[signs == 1] = 1

        return scales

    def get_slae(self, x: torch.tensor, to_numpy: bool) -> Tuple[torch.tensor, torch.tensor]:
        output = self.forward_first_layer(torch.unsqueeze(x, 0))[0]
        matrix = self.layers[0].weight.detach()
        b = (output - self.layers[0].bias).detach()

        if to_numpy:
            matrix, b = matrix.cpu().numpy(), b.cpu().numpy()

        return matrix, b

    def get_matrices(self, x: torch.tensor, p: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        x = torch.flatten(torch.unsqueeze(x, 0), 1)
        y, y_signs, y_scales = [], [], []
        matrices, biases = [], []

        x_input = x

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                y_i = layer(x)
                y_activated_i = self.activation(y_i)
                x = y_activated_i

                signs = y_i[0].sign().detach()

                if i < len(self.layers) - 1:
                    mask = torch.rand(signs.shape, device="cuda") < p
                    signs[mask] *= -1

                scales = self.__get_scales(signs)

                y.append(y_i[0].detach())
                y_signs.append(signs)
                y_scales.append(scales)

                w_i = layer.weight.detach()
                b_i = layer.bias.detach()

                if i == 0:
                    matrices.append(w_i)
                    biases.append(b_i)
                else:
                    w_scales = matrices[-1] * y_scales[i - 1][:, None]
                    b_scales = biases[-1] * y_scales[i - 1]

                    matrices.append(torch.matmul(w_i, w_scales))
                    biases.append(b_i + torch.sum(w_i * b_scales[None, :], dim=1))

        print(matrices[-1].shape, x_input.shape, biases[-1].shape)
        print(torch.max(torch.abs(torch.matmul(x_input, matrices[-1].T) + biases[-1] - y_i)))

        matrices = [matrix.cpu().numpy() for matrix in matrices]
        biases = [bias.cpu().numpy() for bias in biases]
        y_signs = [signs.cpu().numpy() for signs in y_signs]
        y = [yi.cpu().numpy() for yi in y]

        return matrices, biases, y_signs, y
