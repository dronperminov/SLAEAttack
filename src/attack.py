from typing import Optional

import numpy as np
import torch
from qpsolvers import solve_qp

import config
from src.dataset import DATASET_TO_CLASS_NAMES
from src.networks.network import Network


class SLAEAttack:
    def __init__(self, model: Network, image_width: int, image_height: int, image_depth: int) -> None:
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
        first_layer = torch.flatten(self.model.forward_first_layer(torch_input)[0], 0).detach().cpu().numpy().tolist()
        output = self.model(torch_input)[0].detach().cpu().numpy().tolist()
        return {
            "class_names": DATASET_TO_CLASS_NAMES[config.DATASET],
            "x_min": torch.min(torch_input).item(),
            "x_max": torch.max(torch_input).item(),
            "first_layer": first_layer,
            "output": output
        }

    def __get_spiral_mask(self, spiral_size: int) -> np.ndarray:
        mask = np.full((self.image_height, self.image_width), True)
        pnt = np.zeros(2, dtype=np.int32)
        direct = 0
        directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        for _ in range(spiral_size):
            mask[pnt[0]][pnt[1]] = False
            pnt_to = pnt + directions[direct]
            if pnt_to[0] < 0 or pnt_to[0] >= self.image_height or pnt_to[1] < 0 or pnt_to[1] >= self.image_width or not mask[pnt_to[0]][pnt_to[1]]:
                direct = direct + 1 if direct < 3 else 0
                pnt_to = pnt + directions[direct]
            pnt = pnt_to
        return mask.reshape(self.size)

    def qp_attack(self, input_image: np.ndarray, target_image: np.ndarray, ignore_target: bool, scale: float, diff: int, mask_type: str) -> Optional[np.ndarray]:
        image_tensor = self.numpy2tensor(input_image)
        target_vector = self.numpy2vector(target_image)

        matrix, b = self.model.get_slae(image_tensor, to_numpy=True)

        lb = np.zeros(self.size, dtype=np.float64)
        ub = np.ones(self.size, dtype=np.float64)

        if ignore_target:
            p = np.diag(np.random.uniform(0.001, 1, self.size))
            q = np.random.uniform(0, 1, self.size)
        else:
            p = np.eye(self.size, self.size).astype(np.float64)
            q = -target_vector.astype(np.float64)

            if mask_type == "spiral":
                mask = self.__get_spiral_mask(int((1 - scale) * self.size))
            else:
                mask = np.random.choice([True, False], size=self.size, p=[scale, 1 - scale])

            lb[mask] = np.clip(target_vector[mask] - diff / 255, 0, 1)
            ub[mask] = np.clip(target_vector[mask] + diff / 255, 0, 1)

        attacked_image = solve_qp(P=p, q=q, A=matrix, b=b, lb=lb, ub=ub, solver='ecos')

        if attacked_image is None:
            return None

        print(np.min(attacked_image), np.max(attacked_image))
        return self.vector2numpy(attacked_image)

    def split_matrix_attack(self, input_image: np.ndarray, target_image: np.ndarray, ignore_target: bool) -> np.ndarray:
        input_tensor = self.numpy2tensor(input_image)
        matrix, b = self.model.get_slae(input_tensor, to_numpy=True)

        indices = np.random.choice(range(matrix.shape[1]), matrix.shape[0], replace=False)
        other_indices = np.array([i for i in range(matrix.shape[1]) if i not in indices])

        x = np.random.uniform(0, 1, self.size) if ignore_target else self.numpy2vector(target_image)
        x_changed = np.linalg.solve(matrix[:, indices], (b.T - np.matmul(matrix[:, other_indices], x[other_indices])))
        x[indices] = x_changed
        print(x_changed)
        return self.vector2numpy(x)
