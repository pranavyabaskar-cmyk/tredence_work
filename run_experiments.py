import torch
import pandas as pd

from src.model.cnn_model import PrunableCNN
from src.training.train import train
from src.training.evaluate import evaluate
from src.training.sparsity import compute_sparsity
from src.utils.config import *

def run(trainloader, testloader):
    results = []

    for lam in LAMBDA_VALUES:
        print(f"\n Running experiment with lambda = {lam}")

        model = PrunableCNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            train(model, trainloader, optimizer, lam, DEVICE)

        acc = evaluate(model, testloader, DEVICE)
        sparsity = compute_sparsity(model)

        print(f"✅ Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

        results.append([lam, acc, sparsity])
    torch.save(model.state_dict(), f"outputs/models/model_lambda_{lam}.pth")
    df = pd.DataFrame(results, columns=["Lambda", "Accuracy", "Sparsity"])
    df.to_csv("experiments/results.csv", index=False)

    print("\nFinal Results:")
    print(df)