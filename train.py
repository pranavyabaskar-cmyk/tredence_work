import torch
import torch.nn as nn
from src.training.sparsity import sparsity_loss

def train(model, trainloader, optimizer, lambda_sparse, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0

    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        cls_loss = criterion(outputs, labels)
        sp_loss = sparsity_loss(model)

        loss = cls_loss + lambda_sparse * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 progress indicator
        if i % 20 == 0:
            print(f"Batch {i}/{len(trainloader)} | Loss: {loss.item():.4f}")

    return total_loss