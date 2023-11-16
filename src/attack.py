import numpy as np
import torch
from qpsolvers import solve_qp

import config
from src.dataset import DATASET_TO_CLASS_NAMES
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

    def numpy2vector(self, image: np.ndarray) -> np.ndarray:
        if config.IMAGE_DEPTH == 3:
            image = np.moveaxis(image, 2, 0)

        return image.reshape(self.size)

    def numpy2tensor(self, image: np.ndarray) -> torch.tensor:
        if config.IMAGE_DEPTH == 3:
            image = np.moveaxis(image, 2, 0)

        return torch.from_numpy(image.reshape(self.image_size)).to(self.device)

    def vector2numpy(self, tensor: torch.tensor) -> np.ndarray:
        if config.IMAGE_DEPTH == 3:
            image = tensor.reshape((self.image_depth, self.image_height, self.image_width)).astype(np.float32)
            image = np.moveaxis(image, 0, 2)
        else:
            image = tensor.reshape((self.image_height, self.image_width)).astype(np.float32)

        return image

    def predict(self, image: np.ndarray) -> dict:
        torch_input = torch.unsqueeze(self.numpy2tensor(image), 0)
        first_layer = self.model.layers[0](torch.flatten(torch_input, 1))[0].detach().cpu().numpy().tolist()
        output = self.model(torch_input)[0].detach().cpu().numpy().tolist()
        return {
            "class_names": DATASET_TO_CLASS_NAMES[config.DATASET],
            "first_layer": first_layer,
            "output": output
        }

    def attack(self, input_image: np.ndarray, target_image: np.ndarray, scale: float) -> np.ndarray:
        image_tensor = torch.unsqueeze(self.numpy2tensor(input_image), 0)
        target_vector = self.numpy2vector(target_image)

        b = (self.model.layers[0](torch.flatten(image_tensor, 1))[0] - self.model.layers[0].bias).cpu().detach().numpy().astype(np.float64)
        matrix = self.model.layers[0].weight.cpu().detach().numpy().astype(np.float64)

        lb = np.zeros(self.size, dtype=np.float64)
        ub = np.ones(self.size, dtype=np.float64)

        mask = np.random.choice([True, False], size=self.size, p=[scale, 1 - scale])
        lb[mask] = target_vector[mask]
        ub[mask] = target_vector[mask]

        attacked_image = solve_qp(
            P=np.eye(self.size, self.size).astype(np.float64),
            q=-target_vector.astype(np.float64),
            A=matrix, b=b,
            lb=lb, ub=ub,
            solver='ecos'
        )

        print(np.min(attacked_image), np.max(attacked_image))
        return self.vector2numpy(attacked_image)
