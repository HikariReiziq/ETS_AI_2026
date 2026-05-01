from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import is_regressor

class SklearnCAFF_Bad(BaseEstimator, RegressorMixin):
    def __init__(self): pass

class SklearnCAFF_Good(RegressorMixin, BaseEstimator):
    def __init__(self): pass

print("Bad:", is_regressor(SklearnCAFF_Bad()))
print("Good:", is_regressor(SklearnCAFF_Good()))
