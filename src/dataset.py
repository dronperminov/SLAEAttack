import os
from collections import defaultdict
from typing import Set, Tuple

import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchvision.utils import save_image


DATASET_TO_CLASS_NAMES = {
    "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "fashion-mnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "cifar10": ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}


class Dataset:
    def __init__(self, config: dict, transform: transforms) -> None:
        self.dataset_name = config['name']
        self.img_w, self.img_h, self.img_d = config['img_w'], config['img_h'], config.get('img_d', 1)

        if self.dataset_name == "mnist":
            train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('data', train=False, transform=transform)
        elif self.dataset_name == "fashion-mnist":
            train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST('data', train=False, transform=transform)
        elif self.dataset_name == "cifar10":
            train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('data', train=False, transform=transform)
        else:
            raise ValueError(f'Unknown dataset name "{self.dataset_name}"')

        train_data, train_targets = train_dataset.data / 255.0, train_dataset.targets
        test_data, test_targets = test_dataset.data / 255.0, test_dataset.targets

        if isinstance(train_data, list):
            train_data, train_targets = torch.tensor(train_data), torch.tensor(train_targets)

        if isinstance(test_data, list):
            test_data, test_targets = torch.tensor(test_data), torch.tensor(test_targets)

        if self.img_d == 1:
            train_data, test_data = torch.unsqueeze(train_data, 1), torch.unsqueeze(test_data, 1)
        else:
            train_data, test_data = torch.moveaxis(train_data, 3, 1), torch.moveaxis(test_data, 3, 1)

        if "exclude_classes" in config:
            train_data, train_targets = self.__exclude_classes(config["exclude_classes"], train_data, train_targets)
            test_data, test_targets = self.__exclude_classes(config["exclude_classes"], test_data, test_targets)

        if "class_map" in config:
            target2label = self.__get_targets_map(config["class_map"])
            train_targets = self.__map_targets(target2label, train_targets)
            test_targets = self.__map_targets(target2label, test_targets)

        if config.get("balanced", False):
            train_data, train_targets = self.__balance_dataset(train_data, train_targets)
            test_data, test_targets = self.__balance_dataset(test_data, test_targets)

        if config.get("save_examples", 0):
            self.__save_examples(train_data, train_targets, test_data, test_targets, config["save_examples"])

        print(f"Train shapes: data={train_data.shape}, targets={train_targets.shape}")
        print(f" Test shapes: data={test_data.shape}, targets={test_targets.shape}")
        print(f"First 10 train targets: {train_targets[:10]}")
        print(f" First 10 test targets: {test_targets[:10]}")

        self.train_dataset = data_utils.TensorDataset(train_data, train_targets)
        self.test_dataset = data_utils.TensorDataset(test_data, test_targets)

    def flat_size(self) -> int:
        return self.img_h * self.img_w * self.img_d

    def __exclude_classes(self, exclude: Set[int], data: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = [i for i, target in enumerate(targets) if target.item() not in exclude]
        data, targets = data[indices], targets[indices]
        return data, targets

    def __map_targets(self, target2label: dict, targets: torch.Tensor) -> torch.Tensor:
        targets = torch.tensor([target2label[target.item()] for target in targets])
        return targets

    def __get_targets_map(self, class_map: dict) -> dict:
        target2label = {}

        for label, targets in class_map.items():
            for target in targets:
                target2label[target] = label

        return target2label

    def __balance_dataset(self, data: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target2count = defaultdict(list)

        for i, target in enumerate(targets):
            target2count[target.item()].append(i)

        min_count = min(len(indices) for indices in target2count.values())
        indices = []

        for target_indices in target2count.values():
            indices.extend(target_indices[:min_count])

        indices.sort()

        return data[indices], targets[indices]

    def __save_examples(self, train_data: torch.tensor, train_targets: torch.tensor, test_data: torch.tensor, test_targets: torch.tensor, count: int) -> None:
        dataset_path = os.path.join("dataset_examples", self.dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        class_names = DATASET_TO_CLASS_NAMES[self.dataset_name]

        for i in range(count):
            save_image(train_data[i], os.path.join(dataset_path, f"train_{i}_{class_names[train_targets[i]]}.png"))
            save_image(test_data[i], os.path.join(dataset_path, f"test_{i}_{class_names[test_targets[i]]}.png"))
