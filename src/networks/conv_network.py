from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional

from src.networks.network import Network


class ConvNetwork(Network):
    def __init__(self, img_w: int, img_h: int, img_d: int, conv_params: List[dict], sizes: List[int], activation_name: str = "relu") -> None:
        super(ConvNetwork, self).__init__(activation_name=activation_name)
        self.img_w, self.img_h, self.img_d = img_w, img_h, img_d
        self.__init_layers(conv_params, sizes)

    def __init_layers(self, conv_params: List[dict], sizes: List[int]) -> None:
        self.conv_layers = nn.ModuleList()
        self.layers = nn.ModuleList()

        img_w, img_h, img_d = self.img_w, self.img_h, self.img_d

        for params in conv_params:
            filters, fs, stride, padding = params["filters"], params["fs"], params.get("stride", 1), params.get("padding", 0)
            self.conv_layers.append(nn.Conv2d(in_channels=img_d, out_channels=filters, kernel_size=fs, stride=stride, padding=padding))
            img_w, img_h, img_d = (img_w + 2 * padding - fs) // stride + 1, (img_h + 2 * padding - fs) // stride + 1, filters

        inputs = img_w * img_h * img_d

        for layer_size in sizes:
            self.layers.append(nn.Linear(inputs, layer_size))
            inputs = layer_size

    def forward_first_layer(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers[0](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_first_layer(x)
        for layer in self.conv_layers[1:]:
            x = layer(self.activation(x))

        x = torch.flatten(x, 1)
        for layer in self.layers:
            x = layer(self.activation(x))

        return x

    def get_slae(self, x: torch.tensor, to_numpy: bool) -> Tuple[torch.tensor, torch.tensor]:
        output = self.forward_first_layer(torch.unsqueeze(x, 0))[0].detach()

        out_d, out_h, out_w = output.shape[0], output.shape[1], output.shape[2]
        input_size = self.img_w * self.img_h * self.img_d
        output_size = out_w * out_h * out_d
        matrix = torch.zeros((output_size, input_size))

        layer: nn.Conv2d = self.conv_layers[0]
        weight = layer.weight.detach()
        bias = layer.bias.detach()

        fs, fs = layer.kernel_size
        padding, padding = layer.padding
        stride, stride = layer.stride

        for f in range(layer.out_channels):
            output[f] -= bias[f]

            for i in range(out_h):
                for j in range(out_w):
                    for fs_i in range(fs):
                        for fs_j in range(fs):
                            i0 = stride * i + fs_i - padding
                            j0 = stride * j + fs_j - padding

                            if i0 < 0 or i0 >= self.img_h or j0 < 0 or j0 >= self.img_w:
                                continue

                            for c in range(self.img_d):
                                matrix[f * out_h * out_w + i * out_w + j, c * self.img_h * self.img_w + i0 * self.img_w + j0] = weight[f, c, fs_i, fs_j]

        output = output.reshape(output_size)
        x_vec = torch.flatten(x).cpu()
        mat_x_vec = torch.mv(matrix, x_vec)
        test = mat_x_vec - output.cpu()

        print("x_vec:", x_vec.shape, torch.min(x_vec), torch.max(x_vec))
        print("matrix:", matrix.shape, torch.min(matrix), torch.max(matrix))
        print("output:", output.shape, torch.min(output), torch.max(output))
        print("max_x_vec:", mat_x_vec.shape)
        print("test:", torch.min(test), torch.max(test))

        if to_numpy:
            matrix, output = matrix.cpu().numpy(), output.cpu().numpy()

        return matrix, output.reshape(output_size)
