import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # 🔥 Soft gates
        gates = torch.sigmoid(10 * self.gate_scores)

        # 🔥 Hard pruning (binary)
        hard_gates = (gates > 0.5).float()

        # 🔥 Straight-through estimator
        gates = hard_gates.detach() - gates.detach() + gates

        pruned_weights = self.weight * gates

        return F.linear(x, pruned_weights, self.bias)