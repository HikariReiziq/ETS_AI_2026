import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import is_regressor

class SklearnCAFF(BaseEstimator, RegressorMixin):
    _estimator_type = 'regressor'
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
