import torch

BATCH_SIZE = 32
LR = 0.001
EPOCHS = 5   # small for fast run

#  stronger pruning
LAMBDA_VALUES = [0.01, 0.1, 1.0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "./data"
MODEL_SAVE_PATH = "./outputs/models/"
PLOT_PATH = "./outputs/plots/"