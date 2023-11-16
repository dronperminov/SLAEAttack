import torch
import torch.nn.functional
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import transforms

import config
from src.dataset import Dataset
from src.dense_network import DenseNetwork


def train_model(model: DenseNetwork, data_loader: data_utils.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(torch.softmax(output, 1), labels)
        loss.backward()
        optimizer.step()


def evaluate(model: DenseNetwork, data_loader: data_utils.DataLoader, device: torch.device, label: str) -> None:
    loss, error, total = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            predicted = torch.argmax(output, 1)

            loss += criterion(output, labels)
            error += (predicted != labels).sum().item()
            total += labels.shape[0]

    loss /= total
    error /= total

    print(f'    {label:>6}: loss: {loss:.4f}, error: {error:.3%}')


def main() -> None:
    batch_size = 32
    learning_rate = 0.001
    epochs = 10
    device = torch.device("cuda")

    dataset = Dataset({
        "name": config.DATASET,
        "save_examples": 16,
        "img_w": config.IMAGE_WIDTH,
        "img_h": config.IMAGE_HEIGHT,
        "img_d": config.IMAGE_DEPTH
    }, transforms.ToTensor())

    dataloader_args = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True, 'shuffle': True}

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
        model.save(f'models/{config.DATASET}_{"-".join(str(size) for size in config.SIZES)}.pth')


if __name__ == '__main__':
    main()
