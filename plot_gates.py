import torch
import matplotlib.pyplot as plt
import os

from src.model.cnn_model import PrunableCNN
from src.training.sparsity import compute_sparsity
from src.utils.config import DEVICE
from src.model.prunable_linear import PrunableLinear

def plot_gates(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(10 * module.gate_scores)
            gates = gates.detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/gate_distribution.png")
    plt.show()


if __name__ == "__main__":
    # Load your trained model manually if saved
    # OR temporarily reuse last trained model in memory

    model = PrunableCNN().to(DEVICE)

    print("⚠️ NOTE: This loads a fresh model.")
    print("👉 Ideally load trained weights if you saved them.")

    plot_gates(model)