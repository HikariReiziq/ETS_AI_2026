import nbformat as nbf

with open('Sensor_Analysis_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "caff_params = {'hidden': 64" in cell.source:
            cell.source = cell.source.replace("'epochs': 200}", "'epochs': 100}")
        if "def tune_caff(" in cell.source:
            cell.source = cell.source.replace("'epochs': 100 # kurangi epochs", "'epochs': 50 # dikurangi drastis agar 1-2 jam selesai")
            cell.source = cell.source.replace("n_trials=10)", "n_trials=5)")
            cell.source = cell.source.replace("best_params['epochs'] = 200", "best_params['epochs'] = 100")

with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("CAFF training time successfully reduced.")
