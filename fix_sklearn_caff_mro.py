import nbformat as nbf

with open('Sensor_Analysis_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "class SklearnCAFF" in cell.source:
            # Fix the MRO and also remove the hardcoded _estimator_type just to be clean
            new_source = cell.source.replace("class SklearnCAFF(BaseEstimator, RegressorMixin):", "class SklearnCAFF(RegressorMixin, BaseEstimator):")
            cell.source = new_source

with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated to fix SklearnCAFF MRO.")
