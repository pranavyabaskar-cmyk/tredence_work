# Self-Pruning Neural Network

## 1. Why L1 Penalty Encourages Sparsity

The model introduces a learnable gate for each weight, where gate values are constrained between 0 and 1 using a sigmoid function.

To encourage pruning, an L1 penalty is applied to the gate values:

**Sparsity Loss = Σ |gates|**

Since all gate values are positive after the sigmoid activation, this simplifies to:

**Sparsity Loss = Σ gates**

The L1 penalty encourages sparsity because:

- It penalizes non-zero values linearly  
- It pushes less important gate values toward zero  
- Unlike L2 regularization, which only shrinks values, L1 promotes exact zeros  

As a result, weights associated with small gate values are effectively removed, leading to a sparse network.

---

## 2. Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|------------------|--------------|
| 0.01   | 38.41            | 56.05        |
| 0.10   | 39.08            | 60.90        |
| 1.00   | 40.40            | 63.40        |

---

## 3. Observations

- The model achieves significant sparsity (~60%)  
- Increasing λ increases sparsity  
- Accuracy remains relatively stable despite pruning  
- Moderate sparsity acts as a regularizer and slightly improves generalization  

---

## 4. Gate Value Distribution

The distribution of learned gate values is shown below:

![Gate Distribution](outputs/plots/gate_distribution.png)

### Interpretation

- A large spike near **0** represents pruned connections  
- A cluster near **1** represents important retained connections  

This bimodal distribution confirms that the model successfully separates useful and redundant weights, demonstrating effective self-pruning behavior.

---

## 5. Accuracy vs Sparsity Tradeoff

The relationship between model accuracy and sparsity is shown below:

![Accuracy vs Sparsity](outputs/plots/accuracy_sparsity.png)

### Interpretation

- Increasing λ increases sparsity  
- Accuracy remains relatively stable across different λ values  
- This indicates that many network parameters are redundant  

The plot clearly demonstrates the trade-off between model compression and predictive performance.

---

## 6. Conclusion

This project demonstrates a self-pruning neural network that dynamically removes unnecessary weights during training.

The combination of:
- Learnable gating mechanisms  
- L1 sparsity regularization  
- Straight-Through Estimation (STE)  

allows the model to learn a compact and efficient architecture.

The results and visualizations confirm that the network achieves over **60% sparsity** with minimal loss in accuracy, making it effective for model compression and efficient deployment.