import nbformat as nbf

with open('Sensor_Analysis_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Cell 1 Markdown to reflect Phase 1 & 2
for cell in nb.cells:
    if "Phase 1: Load Dataset" in cell.source:
        cell.source = """# PROMPT LANJUTAN UTS AI A — Sensor LC vs PS Cyclic Voltammetry
## Part 4 — Feature Engineering Lanjutan + Advanced Modeling
### Phase 1 & 2: Load Dataset, Preprocessing, & Feature Engineering (16 Fitur)"""

# 2. Fix Imputation Cell to use median per ppm level
impute_cell_source = """# Simpan ke CSV (Raw)
df_lc.to_csv(OUTPUT_DIR / 'dataset_lc_v4_raw.csv', index=False)
df_ps.to_csv(OUTPUT_DIR / 'dataset_ps_v4_raw.csv', index=False)

# Phase 2: Impute features with median PER PPM LEVEL
def impute_missing_per_ppm(df):
    feature_cols = [c for c in df.columns if c.startswith('upper_') or c.startswith('lower_')]
    df_imp = df.copy()
    for col in feature_cols:
        # 1. Impute dengan median dari masing-masing level ppm
        df_imp[col] = df_imp.groupby('ppm')[col].transform(lambda x: x.fillna(x.median()) if not x.isna().all() else x)
        
        # 2. Fallback: jika di suatu level ppm nilainya NaN semua, gunakan global median
        if df_imp[col].isna().any():
            df_imp[col] = df_imp[col].fillna(df_imp[col].median())
            
        # 3. Fallback terakhir: jika seluruh kolom NaN, isi dengan 0
        if df_imp[col].isna().all():
            df_imp[col] = df_imp[col].fillna(0.0)
    return df_imp

print("Melakukan imputasi missing values (NaN) menggunakan median per ppm level...")
df_lc_imputed = impute_missing_per_ppm(df_lc)
df_ps_imputed = impute_missing_per_ppm(df_ps)

# Simpan ke CSV (Imputed)
df_lc_imputed.to_csv(OUTPUT_DIR / 'dataset_lc_v4_imputed.csv', index=False)
df_ps_imputed.to_csv(OUTPUT_DIR / 'dataset_ps_v4_imputed.csv', index=False)

# Tampilkan Statistik LC
print("=== Statistik LC (Raw) ===")
print("Shape:", df_lc.shape)
print("Missing values:\\n", df_lc.isna().sum()[df_lc.isna().sum() > 0])

# Tampilkan Statistik PS
print("\\n=== Statistik PS (Raw) ===")
print("Shape:", df_ps.shape)
print("Missing values:\\n", df_ps.isna().sum()[df_ps.isna().sum() > 0])

print("\\nMissing values setelah imputasi LC:", df_lc_imputed.isna().sum().sum())
print("Missing values setelah imputasi PS:", df_ps_imputed.isna().sum().sum())

# Verifikasi
assert df_lc.shape[0] == 800, f"Expected 800 samples for LC, got {df_lc.shape[0]}"
assert 790 <= df_ps.shape[0] <= 800, f"Expected ~797 samples for PS, got {df_ps.shape[0]}"
print("\\nVERIFIKASI BERHASIL: Phase 1 & 2 terpenuhi.")"""

for cell in nb.cells:
    if "def impute_missing(df):" in cell.source:
        cell.source = impute_cell_source

# 3. Add Phase 3 (EDA) Markdown and Code Cells
phase3_cells = []

phase3_cells.append(nbf.v4.new_markdown_cell("""## Phase 3 — EDA (Exploratory Data Analysis)
Visualisasi distribusi fitur, heatmap korelasi dengan target konsentrasi (`ppm`), dan kurva kalibrasi."""))

# Cell EDA 1: Descriptive Stats
phase3_cells.append(nbf.v4.new_code_cell("""import matplotlib.pyplot as plt
import seaborn as sns

# Statistik deskriptif singkat
print("=== Deskripsi Fitur LC (Setelah Imputasi) ===")
display(df_lc_imputed.describe().T[['mean', 'std', 'min', '50%', 'max']])

print("\\n=== Deskripsi Fitur PS (Setelah Imputasi) ===")
display(df_ps_imputed.describe().T[['mean', 'std', 'min', '50%', 'max']])"""))

# Cell EDA 2: Correlation Heatmap
phase3_cells.append(nbf.v4.new_code_cell("""# Hitung korelasi dengan target (ppm)
corr_lc = df_lc_imputed.corr()[['ppm']].sort_values(by='ppm', ascending=False)
corr_ps = df_ps_imputed.corr()[['ppm']].sort_values(by='ppm', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.heatmap(corr_lc, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
axes[0].set_title('LC: Fitur vs Konsentrasi (PPM)', fontsize=14)

sns.heatmap(corr_ps, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
axes[1].set_title('PS: Fitur vs Konsentrasi (PPM)', fontsize=14)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase3_korelasi.png', dpi=300)
plt.show()"""))

# Cell EDA 3: Boxplot of key features per PPM level
phase3_cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Pilih 2 fitur penting dari upper dan 2 fitur penting dari lower (contoh: Ip_mean dan Area_mean)
sns.boxplot(data=df_lc_imputed, x='ppm', y='upper_Ip_mean', ax=axes[0, 0])
axes[0, 0].set_title('LC: Upper Ip Mean per PPM')
axes[0, 0].tick_params(axis='x', rotation=45)

sns.boxplot(data=df_ps_imputed, x='ppm', y='upper_Ip_mean', ax=axes[0, 1])
axes[0, 1].set_title('PS: Upper Ip Mean per PPM')
axes[0, 1].tick_params(axis='x', rotation=45)

sns.boxplot(data=df_lc_imputed, x='ppm', y='lower_Ip_mean', ax=axes[1, 0])
axes[1, 0].set_title('LC: Lower Ip Mean per PPM')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.boxplot(data=df_ps_imputed, x='ppm', y='lower_Ip_mean', ax=axes[1, 1])
axes[1, 1].set_title('PS: Lower Ip Mean per PPM')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase3_boxplot_ppm.png', dpi=300)
plt.show()"""))

# Cell EDA 4: Calibration Curve
phase3_cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Linear Scale
sns.lineplot(data=df_lc_imputed, x='ppm', y='upper_Ip_mean', marker='o', label='LC', errorbar='sd', ax=axes[0])
sns.lineplot(data=df_ps_imputed, x='ppm', y='upper_Ip_mean', marker='s', label='PS', errorbar='sd', ax=axes[0])
axes[0].set_title('Kurva Kalibrasi (Skala Linear)')
axes[0].set_xlabel('Konsentrasi (ppm)')
axes[0].set_ylabel('Upper Peak Ip Mean (µA)')
axes[0].grid(True, linestyle='--', alpha=0.7)

# Log Scale
sns.lineplot(data=df_lc_imputed, x='ppm', y='upper_Ip_mean', marker='o', label='LC', errorbar='sd', ax=axes[1])
sns.lineplot(data=df_ps_imputed, x='ppm', y='upper_Ip_mean', marker='s', label='PS', errorbar='sd', ax=axes[1])
axes[1].set_xscale('log')
axes[1].set_title('Kurva Kalibrasi (Skala Logaritmik)')
axes[1].set_xlabel('Konsentrasi (ppm) [Log]')
axes[1].set_ylabel('Upper Peak Ip Mean (µA)')
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase3_calibration_curve.png', dpi=300)
plt.show()"""))

# Only append if Phase 3 is not already there
has_phase3 = any("Phase 3 — EDA" in cell.source for cell in nb.cells if cell.cell_type == "markdown")
if not has_phase3:
    nb.cells.extend(phase3_cells)

with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook Sensor_Analysis_v4.ipynb updated with fixed Imputation (per ppm level) and Phase 3 (EDA).")
