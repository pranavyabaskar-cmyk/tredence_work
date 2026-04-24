import torch
import torchvision
import torchvision.transforms as transforms

from src.utils.config import *
from experiments.run_experiments import run


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=False, download=True, transform=transform
    )

    # 🔥 SUBSET FOR SPEED
    subset_size = 2000
    train_subset = torch.utils.data.Subset(trainset, range(subset_size))

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False
    )

    return trainloader, testloader


if __name__ == "__main__":
    print("🚀 Starting training pipeline...")

    trainloader, testloader = get_data()

    print(f"Train batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")

    run(trainloader, testloader)