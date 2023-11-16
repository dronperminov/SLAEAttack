import numpy as np
import torch
from qpsolvers import solve_qp

from src.dense_network import DenseNetwork


class SLAEAttack:
    def __init__(self, model: DenseNetwork, image_width: int, image_height: int, image_depth: int) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.image_size = (self.image_depth, self.image_height, self.image_width)
        self.size = image_width * image_height * image_depth

        self.device = torch.device("cuda")
        self.model = model

    def numpy2tensor(self, image: np.ndarray) -> torch.tensor:
        return torch.from_numpy(image.reshape(self.image_size)).to(self.device)

    def predict(self, image: np.ndarray) -> dict:
        torch_input = torch.unsqueeze(self.numpy2tensor(image), 0)
        first_layer = self.model.layers[0](torch.flatten(torch_input, 1))[0].detach().cpu().numpy().tolist()
        output = self.model(torch_input)[0].detach().cpu().numpy().tolist()
        return {
            "first_layer": first_layer,
            "output": output
        }

    def attack(self, input_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        image_tensor = self.numpy2tensor(input_image)

        bias = (self.model.layers[0](torch.flatten(image_tensor, 1)) - self.model.layers[0].bias).cpu().detach().numpy()
        matrix = self.model.layers[0].weight.cpu().detach().numpy()
        x_t = solve_qp(
            P=np.eye(self.size, self.size).astype(np.float64), q=-target_image.reshape(self.size).astype(np.float64),
            A=matrix.astype(np.float64), b=bias.astype(np.float64),
            lb=np.zeros(self.size).astype(np.float64), ub=np.ones(self.size).astype(np.float64),
            solver='ecos'
        )

        attacked_image = x_t.reshape(self.image_height, self.image_width).astype(np.float32)
        return attacked_image
