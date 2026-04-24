import torch
from src.model.prunable_linear import PrunableLinear


def sparsity_loss(model):
    loss = 0.0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(10 * module.gate_scores)

            # 🔥 normalized loss (important)
            loss += torch.mean(gates)

    return loss


def compute_sparsity(model, threshold=0.5):
    total = 0
    zero = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = (torch.sigmoid(10 * module.gate_scores) > threshold).float()

            total += gates.numel()
            zero += torch.sum(gates == 0).item()

    return 100 * zero / total