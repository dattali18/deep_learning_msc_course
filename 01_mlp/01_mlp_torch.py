"""
This python module is a for creating a PyTorch Neural Network for the MNIST dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("mps" if torch.mps.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        # Keep raw logits for cross_entropy
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MNIST arrives as [B, 1, 28, 28]; flatten to [B, 784]
        x = x.view(x.size(0), -1)
        x = self.sequential(x)
        return x

    def set_optimizer(self, optimizer: optim.Optimizer) -> None:
        self.optimizer = optimizer

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader) -> tuple:
        self.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for _, (images, labels) in tqdm.tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc="Train",
                unit="batch",
                leave=False,
        ):
            images, labels = images.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.forward(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            bs = images.size(0)
            train_loss_sum += loss.item() * bs
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_count += bs

        train_loss = train_loss_sum / train_count
        train_acc = train_correct / train_count

        self.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.forward(images)
                loss = self.loss_function(outputs, labels)

                bs = images.size(0)
                val_loss_sum += loss.item() * bs
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_count += bs

        val_loss = val_loss_sum / val_count
        val_acc = val_correct / val_count
        return train_loss, train_acc, val_loss, val_acc

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10) -> list:
        history = []

        pbar = tqdm.trange(epochs, desc="Training", unit="epoch")
        for epoch in pbar:
            train_loss, train_acc, val_loss, val_acc = self.train_epoch(train_loader, val_loader)

            history.append([epoch + 1, train_loss, val_loss, train_acc, val_acc])
            pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                acc=f"{train_acc:.4f}",
                val_acc=f"{val_acc:.4f}",
            )

        return history

    def eval_model(self, test_loader: DataLoader) -> tuple:
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm.tqdm(test_loader, desc="Evaluating", unit="batch"):
                # put images, labels to device
                images, labels = images.to(device), labels.to(device)

                outputs = self.forward(images)
                total_loss += F.cross_entropy(outputs, labels, reduction="sum").item()
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def save_model(self, path: str = "models/mlp.pth") -> None:
        print(f"Saving model to {path}")
        torch.save(self.state_dict(), path)

    def load_model(self, path: str = "models/mlp.pth") -> None:
        print(f"Loading model from {path}")
        self.load_state_dict(torch.load(path))


def plot_mnist_images(images, labels, n_x: int = 3, n_y: int = 3) -> None:
    # plot a grid of n_x x n_y images using matplotlib
    fig, axes = plt.subplots(n_x, n_y, figsize=(n_y * 2, n_x * 2))
    for i in range(n_x):
        for j in range(n_y):
            random_idx = np.random.randint(0, len(images))
            label = labels[random_idx]
            axes[i, j].imshow(images[random_idx].squeeze(), cmap="gray")
            axes[i, j].set_title(f"Label: {label}")
            axes[i, j].axis("off")

    plt.show()


def plot_training_history(history: list) -> None:
    history = np.array(history)
    epochs = history[:, 0]
    train_loss = history[:, 1]
    val_loss = history[:, 2]
    train_acc = history[:, 3]
    val_acc = history[:, 4]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: loss
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: accuracy
    axes[1].plot(epochs, train_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc, label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    print("Using device:", device)

    # load the data from torch
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    # plot 3 x 3 matrix images
    # load a list of images from the test_dataset
    images, labels = [], []
    for i in range(100):
        image, label = test_dataset[i]
        images.append(image)
        labels.append(label)

    plot_mnist_images(images, labels, 3, 3)

    mlp = MLP().to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    mlp.set_optimizer(optimizer)
    history = mlp.train_model(train_loader, test_loader, epochs=10)
    loss, acc = mlp.eval_model(test_loader)

    # plot the history
    plot_training_history(history)

    print("Accuracy: {:.2f}%".format(acc * 100))

    # save the model
    mlp.save_model("models/mlp.pth")


if __name__ == "__main__":
    main()
