class SklearnCAFF:
    _estimator_type = "regressor"
    def __init__(self):
        pass

model = SklearnCAFF()
print("Estimator type:", getattr(model, "_estimator_type", None))
