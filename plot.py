import matplotlib.pyplot as plt
import torch
from src.model.prunable_linear import PrunableLinear

def plot_gates(model, save_path):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.savefig(save_path)
    plt.close()