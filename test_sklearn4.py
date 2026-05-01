from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import is_regressor

class SklearnCAFF(BaseEstimator, RegressorMixin):
    _estimator_type = 'regressor'
    def __init__(self):
        pass

model = SklearnCAFF()
print("Is regressor?", is_regressor(model))
print("Estimator type:", getattr(model, "_estimator_type", None))
