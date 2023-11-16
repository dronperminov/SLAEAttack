from abc import abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional


class Network(nn.Module):
    def __init__(self, activation_name: str = "relu") -> None:
        super(Network, self).__init__()
        self.activation_name = activation_name
        self.activation = self.__get_activation(activation_name)

    def __get_activation(self, activation: str) -> torch.nn.functional:
        if activation == "relu":
            return torch.nn.functional.relu

        if activation == "leaky-relu":
            return torch.nn.functional.leaky_relu

        if activation == "abs":
            return torch.abs

        raise ValueError(f"Invalid activation '{activation}'")

    @abstractmethod
    def forward_first_layer(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_slae(self, x: torch.tensor, to_numpy: bool) -> Tuple[torch.tensor, torch.tensor]:
        pass

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
