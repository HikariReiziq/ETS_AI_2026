import nbformat as nbf

with open('Sensor_Analysis_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "class SklearnCAFF(BaseEstimator, RegressorMixin):" in cell.source:
            # Set explicit estimator type
            if "_estimator_type = 'regressor'" not in cell.source:
                cell.source = cell.source.replace(
                    "class SklearnCAFF(BaseEstimator, RegressorMixin):\n    def __init__", 
                    "class SklearnCAFF(BaseEstimator, RegressorMixin):\n    _estimator_type = 'regressor'\n    def __init__"
                )

with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated to fix SklearnCAFF ValueError.")
