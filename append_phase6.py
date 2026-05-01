import nbformat as nbf

with open('Sensor_Analysis_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

phase6_cells = []

# Markdown Phase 6
phase6_cells.append(nbf.v4.new_markdown_cell("""## Phase 6 — Evaluasi & Komparasi Final
Visualisasi metrik performa semua model, Parity Plot model terbaik, Residual Plot, dan Feature Importance."""))

# 1. Bar Chart Perbandingan R²
phase6_cells.append(nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Menyiapkan DataFrame Hasil
df_res_lc['Sensor'] = 'LC'
df_res_ps['Sensor'] = 'PS'
df_res_lc['Model'] = df_res_lc.index
df_res_ps['Model'] = df_res_ps.index

df_all = pd.concat([df_res_lc, df_res_ps])

plt.figure(figsize=(16, 8))
sns.barplot(data=df_all, x='R2', y='Model', hue='Sensor', palette='viridis')
plt.title('Perbandingan R² Semua Model (LC vs PS)', fontsize=16)
plt.xlabel('R² Score')
plt.ylabel('Model')
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase6_barchart_r2.png', dpi=300)
plt.show()"""))

# 2. Boxplot R² per fold (Top 5 Models)
phase6_cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import cross_val_score

# Kita pilih 5 model tercepat dan terbaik untuk LC & PS agar tidak memakan waktu lama (seperti CAFF)
top_fast_models = ['6. RandomForest', '7. ExtraTrees', '8. XGBoost_Tuned', '9. LightGBM_Tuned', '10. CatBoost_Tuned']

cv_results = []
print("Menghitung R² per fold untuk Top 5 Fast Models...")
for m_name in top_fast_models:
    if m_name in best_models['LC']:
        model_lc = best_models['LC'][m_name]
        scores_lc = cross_val_score(model_lc, X_lc_scaled, y_lc, cv=skf, scoring='r2', n_jobs=-1)
        for s in scores_lc: cv_results.append({'Sensor': 'LC', 'Model': m_name, 'R2': s})
        
    if m_name in best_models['PS']:
        model_ps = best_models['PS'][m_name]
        scores_ps = cross_val_score(model_ps, X_ps_scaled, y_ps, cv=skf, scoring='r2', n_jobs=-1)
        for s in scores_ps: cv_results.append({'Sensor': 'PS', 'Model': m_name, 'R2': s})

df_cv = pd.DataFrame(cv_results)

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_cv, x='Model', y='R2', hue='Sensor', palette='Set2')
plt.title('Sebaran R² per Fold (5-Fold CV) pada Top 5 Fast Models')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase6_boxplot_r2_fold.png', dpi=300)
plt.show()"""))

# 3. Parity Plot & Residual Plot Model Terbaik
phase6_cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import cross_val_predict
import numpy as np

# Ambil nama model terbaik (R2 tertinggi)
best_model_name_lc = df_res_lc['R2'].astype(float).idxmax()
best_model_name_ps = df_res_ps['R2'].astype(float).idxmax()

print(f"Model Terbaik LC: {best_model_name_lc}")
print(f"Model Terbaik PS: {best_model_name_ps}")

model_lc_best = best_models['LC'][best_model_name_lc]
model_ps_best = best_models['PS'][best_model_name_ps]

print("Generating cross_val_predict untuk Parity Plot...")
# n_jobs=None untuk menghindari pickling error jika melibatkan PyTorch
jobs_lc = None if ('CAFF' in best_model_name_lc or 'Stacking' in best_model_name_lc or 'Voting' in best_model_name_lc) else -1
jobs_ps = None if ('CAFF' in best_model_name_ps or 'Stacking' in best_model_name_ps or 'Voting' in best_model_name_ps) else -1

y_pred_log_lc = cross_val_predict(model_lc_best, X_lc_scaled, y_lc, cv=skf, n_jobs=jobs_lc)
y_pred_log_ps = cross_val_predict(model_ps_best, X_ps_scaled, y_ps, cv=skf, n_jobs=jobs_ps)

# Inverse transform
y_true_inv_lc = (10 ** y_lc) - 1
y_pred_inv_lc = np.maximum(0, (10 ** y_pred_log_lc) - 1)

y_true_inv_ps = (10 ** y_ps) - 1
y_pred_inv_ps = np.maximum(0, (10 ** y_pred_log_ps) - 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Parity Plot LC
axes[0].scatter(y_true_inv_lc, y_pred_inv_lc, alpha=0.5, color='blue')
axes[0].plot([0, 1000], [0, 1000], 'r--', lw=2)
axes[0].set_title(f'Parity Plot LC ({best_model_name_lc})')
axes[0].set_xlabel('Actual PPM')
axes[0].set_ylabel('Predicted PPM')
axes[0].grid(True, linestyle='--', alpha=0.7)

# Parity Plot PS
axes[1].scatter(y_true_inv_ps, y_pred_inv_ps, alpha=0.5, color='orange')
axes[1].plot([0, 1000], [0, 1000], 'r--', lw=2)
axes[1].set_title(f'Parity Plot PS ({best_model_name_ps})')
axes[1].set_xlabel('Actual PPM')
axes[1].set_ylabel('Predicted PPM')
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase6_parity_plot.png', dpi=300)
plt.show()"""))

# 4. Residual Plot per PPM Range
phase6_cells.append(nbf.v4.new_code_cell("""def plot_residuals(y_true, y_pred, sensor_name, ax):
    res = y_pred - y_true
    
    # Kategori range ppm
    ranges = []
    for val in y_true:
        if val <= 10: ranges.append('0-10')
        elif val <= 100: ranges.append('10-100')
        else: ranges.append('100-1000')
        
    df_res = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred, 'Residual': res, 'Range': ranges})
    
    sns.boxplot(data=df_res, x='Range', y='Residual', ax=ax, order=['0-10', '10-100', '100-1000'], palette='pastel')
    ax.axhline(0, color='red', linestyle='--', lw=2)
    ax.set_title(f'Residual Distribution ({sensor_name})')
    ax.set_ylabel('Residual (Predicted - Actual) PPM')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_residuals(y_true_inv_lc, y_pred_inv_lc, 'LC', axes[0])
plot_residuals(y_true_inv_ps, y_pred_inv_ps, 'PS', axes[1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase6_residual_plot.png', dpi=300)
plt.show()"""))

# 5. Feature Importance (SHAP)
phase6_cells.append(nbf.v4.new_code_cell("""import shap

print("Membuat Feature Importance menggunakan SHAP pada XGBoost_Tuned...")

if '8. XGBoost_Tuned' in best_models['LC']:
    model_shap_lc = best_models['LC']['8. XGBoost_Tuned']
    model_shap_lc.fit(X_lc_scaled, y_lc) # Fit ulang di seluruh data untuk interpretasi
    explainer_lc = shap.TreeExplainer(model_shap_lc)
    shap_values_lc = explainer_lc.shap_values(X_lc_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Feature Importance - LC (XGBoost)")
    shap.summary_plot(shap_values_lc, X_lc_scaled, feature_names=feature_cols, show=False)
    plt.savefig(OUTPUT_DIR / 'phase6_shap_lc.png', dpi=300, bbox_inches='tight')
    plt.show()

if '8. XGBoost_Tuned' in best_models['PS']:
    model_shap_ps = best_models['PS']['8. XGBoost_Tuned']
    model_shap_ps.fit(X_ps_scaled, y_ps)
    explainer_ps = shap.TreeExplainer(model_shap_ps)
    shap_values_ps = explainer_ps.shap_values(X_ps_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Feature Importance - PS (XGBoost)")
    shap.summary_plot(shap_values_ps, X_ps_scaled, feature_names=feature_cols, show=False)
    plt.savefig(OUTPUT_DIR / 'phase6_shap_ps.png', dpi=300, bbox_inches='tight')
    plt.show()"""))

# 6. Dashboard Ringkasan Final
phase6_cells.append(nbf.v4.new_markdown_cell("""### 🏆 Kesimpulan & Dashboard Ringkasan
- Visualisasi di atas menunjukkan perbandingan performa model LC vs PS.
- Parity plot dan Residual plot membuktikan seberapa baik model terbaik (otomatis dipilih berdasarkan R²) memprediksi pada berbagai rentang konsentrasi.
- Interpretasi SHAP menunjukkan fitur-fitur (dari total 16 fitur Dual-Peak) yang paling berpengaruh dalam prediksi ppm.
- Seluruh grafik telah tersimpan secara otomatis sebagai PNG resolusi tinggi di dalam folder `output/`."""))

# Append ke nb
has_phase6 = any("Phase 6" in cell.source for cell in nb.cells if cell.cell_type == "markdown")
if not has_phase6:
    nb.cells.extend(phase6_cells)

with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Phase 6 appended successfully!")
