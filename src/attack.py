from typing import Optional

import numpy as np
import qpsolvers
import torch
from qpsolvers import solve_qp

import config
from src.dataset import DATASET_TO_CLASS_NAMES
from src.networks.network import Network
from scipy.optimize import linprog


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

    def __get_spiral_mask(self, scale: float) -> np.ndarray:
        spiral_size = int((1 - scale) * self.image_height * self.image_width)
        mask = np.full((self.image_depth, self.image_height, self.image_width), True)
        pnt = np.zeros(2, dtype=np.int32)
        direct = 0
        directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        for _ in range(spiral_size):
            mask[:, pnt[0], pnt[1]] = False
            pnt_to = pnt + directions[direct]

            if pnt_to[0] < 0 or pnt_to[0] >= self.image_height or pnt_to[1] < 0 or pnt_to[1] >= self.image_width or not mask[0, pnt_to[0], pnt_to[1]]:
                direct = direct + 1 if direct < 3 else 0
                pnt_to = pnt + directions[direct]

            pnt = pnt_to

        return mask.reshape(self.size)

    def __get_random_mask(self, scale: float) -> np.ndarray:
        mask = np.random.choice([True, False], size=(self.image_height, self.image_width), p=[scale, 1 - scale])

        if config.IMAGE_DEPTH == 3:
            mask = np.concatenate([mask, mask, mask])

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
                mask = self.__get_spiral_mask(scale)
            else:
                mask = self.__get_random_mask(scale)

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

    def milti_layers_attack(self, input_image: np.ndarray, target_image: np.ndarray, ignore_target: bool, scale: float, diff: int, mask_type: str, signs_p: float, target: str) -> np.ndarray:
        image_tensor = self.numpy2tensor(input_image)
        target_vector = self.numpy2vector(target_image)

        lb = np.zeros(self.size, dtype=np.float64)
        ub = np.ones(self.size, dtype=np.float64)

        if ignore_target:
            p = np.diag(np.random.uniform(0.001, 1, self.size))
            q = np.random.uniform(0, 1, self.size)
        else:
            p = np.eye(self.size, self.size).astype(np.float64)
            q = -target_vector.astype(np.float64)

            if mask_type == "spiral":
                mask = self.__get_spiral_mask(scale)
            else:
                mask = self.__get_random_mask(scale)

            lb[mask] = np.clip(target_vector[mask] - diff / 255, 0, 1)
            ub[mask] = np.clip(target_vector[mask] + diff / 255, 0, 1)

        matrices, biases, signs, y = self.model.get_matrices(image_tensor, signs_p)

        G = np.concatenate([-matrix * sign[:, None] for matrix, sign in zip(matrices[:-1], signs[:-1])])
        h = np.concatenate([bias * sign for bias, sign in zip(biases[:-1], signs[:-1])])

        if target == "values":
            A = matrices[-1]
            b = y[-1] - biases[-1]

            attacked_image = solve_qp(P=p, q=q, A=A, b=b, G=G, h=h, lb=lb, ub=ub, solver='scs')
        elif target == "class":
            target_class = np.argmax(y[-1])
            y_lower = np.zeros_like(y[-1]) - 100
            y_upper = np.zeros_like(y[-1])
            y_lower[target_class] = 2
            y_upper[target_class] = 20

            matrix = np.concatenate([G, -matrices[-1], matrices[-1]], axis=0)
            vector = np.concatenate([h, -(y_lower - biases[-1]), y_upper - biases[-1]])

            attacked_image = solve_qp(P=p, q=q, G=matrix, h=vector, lb=lb, ub=ub, solver='scs')
        else:
            raise ValueError(f'Unknown target "{target}"')

        if attacked_image is None:
            return None

        print(np.min(attacked_image), np.max(attacked_image))
        return self.vector2numpy(attacked_image)
