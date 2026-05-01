import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import is_regressor

class SklearnCAFF(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=64, n_heads=4, dropout=0.1, lr=0.001, batch_size=32, epochs=100):
        self.hidden = hidden
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

model = SklearnCAFF()
print("Is regressor?", is_regressor(model))
print("Estimator type:", getattr(model, "_estimator_type", None))
