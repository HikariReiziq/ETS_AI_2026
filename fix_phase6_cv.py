import nbformat as nbf

with open('Sensor_Analysis_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "cross_val_score" in cell.source or "cross_val_predict" in cell.source:
            # Replace cv=skf for LC
            cell.source = cell.source.replace(
                "cv=skf, scoring='r2', n_jobs=-1", 
                "cv=list(skf.split(X_lc_scaled, labels_lc)) if 'lc' in m_name else list(skf.split(X_ps_scaled, labels_ps)), scoring='r2', n_jobs=-1"
            )
            # Actually, the loop uses model_lc for LC and model_ps for PS. Let's do exact replacements.
            cell.source = cell.source.replace(
                "scores_lc = cross_val_score(model_lc, X_lc_scaled, y_lc, cv=skf, scoring='r2', n_jobs=-1)",
                "scores_lc = cross_val_score(model_lc, X_lc_scaled, y_lc, cv=list(skf.split(X_lc_scaled, labels_lc)), scoring='r2', n_jobs=-1)"
            )
            cell.source = cell.source.replace(
                "scores_ps = cross_val_score(model_ps, X_ps_scaled, y_ps, cv=skf, scoring='r2', n_jobs=-1)",
                "scores_ps = cross_val_score(model_ps, X_ps_scaled, y_ps, cv=list(skf.split(X_ps_scaled, labels_ps)), scoring='r2', n_jobs=-1)"
            )
            
            cell.source = cell.source.replace(
                "cross_val_predict(model_lc_best, X_lc_scaled, y_lc, cv=skf",
                "cross_val_predict(model_lc_best, X_lc_scaled, y_lc, cv=list(skf.split(X_lc_scaled, labels_lc))"
            )
            cell.source = cell.source.replace(
                "cross_val_predict(model_ps_best, X_ps_scaled, y_ps, cv=skf",
                "cross_val_predict(model_ps_best, X_ps_scaled, y_ps, cv=list(skf.split(X_ps_scaled, labels_ps))"
            )

with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated to fix cross_val_score and cross_val_predict cv argument.")
