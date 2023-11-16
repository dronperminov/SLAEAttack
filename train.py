import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import transforms

import config
from src.dataset import Dataset
from src.dense_network import DenseNetwork


def train_model(model: DenseNetwork, data_loader: data_utils.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), torch.unsqueeze(target, dim=-1))
        loss.backward()
        optimizer.step()


def evaluate(model: DenseNetwork, data_loader: data_utils.DataLoader, device: torch.device, label: str) -> None:
    model.eval()
    loss = 0
    error = 0
    total = len(data_loader.dataset)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = torch.squeeze(model(data))
            output[output < 0] = -1
            output[output >= 0] = 1

            loss += F.mse_loss(output, target, reduction='sum').item()
            error += (output != target).sum().item()

    loss /= total
    error /= total

    print(f'    {label:>6}: loss: {loss:.4f}, error: {error:.3%}')


def main():
    batch_size = 32
    learning_rate = 0.001
    epochs = 10
    device = torch.device("cuda")

    dataset = Dataset({
        "name": config.DATASET,
        "class_names": {-1: "seven", 1: "other"},
        "exclude_classes": {},
        "class_map": {-1.0: [7], 1.0: [0, 1, 2, 3, 4, 5, 6, 8, 9]},
        "balanced": True,
        "img_w": 28,
        "img_h": 28
    }, transforms.ToTensor())

    dataloader_args = {
        'batch_size': batch_size,
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    }

    data_loaders = {
        "train": torch.utils.data.DataLoader(dataset.train_dataset, **dataloader_args),
        "test": torch.utils.data.DataLoader(dataset.test_dataset, **dataloader_args)
    }

    model = DenseNetwork(dataset.flat_size(), config.SIZES, config.ACTIVATION).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_model(model, data_loaders["train"], optimizer, device)

        print(f"\nEpoch {epoch}:")
        evaluate(model, data_loaders["train"], device, "train")
        evaluate(model, data_loaders["test"], device, "test")
        model.save(f"models/model_epoch{epoch + 1}.pth")


if __name__ == '__main__':
    main()
