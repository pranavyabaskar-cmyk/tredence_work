import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("experiments/results.csv")

# Sort by lambda
df = df.sort_values("Lambda")

# Plot
plt.figure()
plt.plot(df["Lambda"], df["Accuracy"], marker='o', label="Accuracy")
plt.plot(df["Lambda"], df["Sparsity"], marker='o', label="Sparsity")

plt.xscale("log")
plt.xlabel("Lambda (log scale)")
plt.ylabel("Value")
plt.title("Accuracy vs Sparsity Tradeoff")
plt.legend()

plt.savefig("outputs/plots/accuracy_sparsity.png")
plt.show()