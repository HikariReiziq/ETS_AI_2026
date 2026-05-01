Tolong buatkan file `Sensor_Analysis_v3.ipynb` dari nol di workspace berikut:

**Workspace:** `C:\Users\Hikar\Documents\Kuliah\Bahan Belajar Kuliah Semester 4\AI\UTS Kelas AI A`

---

## KONTEKS PROYEK

Proyek UTS AI A: membandingkan sensor elektrokimia **LC (LecSens, custom)** vs **PS (EmStat4 PalmSens, komersial)** untuk deteksi Cd²⁺ menggunakan Cyclic Voltammetry (CV).

- Konsentrasi Cd²⁺: 16 level — KCL(0 ppm), 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000 ppm
- 50 file `.xlsx` per level, masing-masing berisi 10 scan CV
- Sensor LC: sweep dimulai ANODIC (V naik). Sensor PS: sweep dimulai CATHODIC (V turun)
- KCL folder = background elektrolit 0.1M tanpa Cd²⁺ → ppm = 0

**Part 1 sudah selesai** di `Sensor_Analysis.ipynb` (4 fitur, XGBoost R²=0.823 LC / 0.579 PS).
**Part 2 = v3 ini**: 16 fitur dual-peak, 13 model + Optuna + TabICLv2 + CAFF.

---

## ENVIRONMENT

```
Python  : 3.13 di .venv (folder workspace)
Tersedia: numpy, pandas, scipy, sklearn, xgboost, lightgbm, matplotlib, seaborn, openpyxl
Install : pip install catboost optuna shap tabicl torch
JANGAN  : df.style.highlight_max() — jinja2 tidak tersedia
```

---

## PIPELINE (6 Phase, semua dalam 1 notebook)

```
[Phase 1] Load Dataset
[Phase 2] Preprocessing & Feature Engineering
[Phase 3] EDA (Exploratory Data Analysis) — WAJIB ada visualisasi nyata
[Phase 4] Model Training & Hyperparameter Tuning
[Phase 5] Ensemble & Stacking
[Phase 6] Evaluasi & Komparasi Final
```

Setiap phase diawali **Markdown cell** dengan header `## Phase X — Nama Phase`.
Semua code cell harus dijalankan dan menghasilkan output / plot yang terlihat.

---

## KONFIGURASI TEKNIS (gunakan PERSIS nilai ini)

```python
BASE_DIR = Path(r'C:\Users\Hikar\Documents\Kuliah\Bahan Belajar Kuliah Semester 4\AI\UTS Kelas AI A')
LC_DIR   = BASE_DIR / 'Alat Sensor LC'
PS_DIR   = BASE_DIR / 'Alat Sensor PS'
OUTPUT_DIR = BASE_DIR / 'output'

PPM_LEVELS   = [2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]
STABLE_SCANS = list(range(2, 10))   # scan indeks 2–9 (8 scan stabil per file)

# Window peak per sensor
UPPER_V_MIN_LC, UPPER_V_MAX_LC = -0.62, -0.43
UPPER_V_MIN_PS, UPPER_V_MAX_PS = -0.80, -0.43
LOWER_V_MIN_LC, LOWER_V_MAX_LC = -0.80, -0.40
LOWER_V_MIN_PS, LOWER_V_MAX_PS = -0.90, -0.45

# Target transformation
# y = np.log10(ppm + 1)  →  inverse: ppm_pred = 10**y_pred - 1
# log10(0+1) = 0.0 untuk KCL/background

# 16 fitur dual-peak
UPPER_FEATURES = ['upper_Ip_mean','upper_Ip_std','upper_Ep_mean','upper_Ep_std',
                  'upper_Area_mean','upper_FWHM_mean','upper_skewness','upper_kurtosis']
LOWER_FEATURES = ['lower_Ip_mean','lower_Ip_std','lower_Ep_mean','lower_Ep_std',
                  'lower_Area_mean','lower_FWHM_mean','lower_skewness','lower_kurtosis']
FEATURE_COLS = UPPER_FEATURES + LOWER_FEATURES
```

---

## DETAIL PHASE

### Phase 1 — Load Dataset (Cell 1–5)

- **Cell 1 (md):** Judul notebook + deskripsi singkat 6-phase pipeline
- **Cell 2:** Semua import + konfigurasi konstanta di atas
- **Cell 3:** Fungsi `load_cv_file()`, `_detect_starts_anodic()`, `get_anodic_sweep()`, `get_cathodic_sweep()`

```python
def load_cv_file(filepath, sensor_type='LC'):
    try:
        if sensor_type == 'LC':
            df = pd.read_excel(filepath, header=[0,1], sheet_name='Data CV')
        else:
            df = pd.read_excel(filepath, header=[0,1], sheet_name=0)
    except Exception:
        df = pd.read_excel(filepath, header=[0,1], sheet_name=0)
    if isinstance(df, dict):
        df = list(df.values())[0]
    scans = {}
    n_scans = df.shape[1] // 2
    for s in range(n_scans):
        v = df.iloc[:, s*2].dropna().values.astype(float)
        i = df.iloc[:, s*2+1].dropna().values.astype(float)
        if len(v) > 10:
            scans[s] = {'V': v, 'I': i}
    return scans

def _detect_starts_anodic(v):
    return np.mean(np.diff(v[:min(10, len(v))])) > 0

def get_anodic_sweep(v, i):
    if len(v) < 5: return v, i
    if _detect_starts_anodic(v):
        dv = np.diff(v)
        first_cat = [idx for idx in np.where(dv < 0)[0] if idx >= max(5, len(v)//5)]
        if not first_cat: return v, i
        return v[:first_cat[0]+1], i[:first_cat[0]+1]
    else:
        turn = int(np.argmin(v))
        return v[turn:], i[turn:]

def get_cathodic_sweep(v, i):
    if len(v) < 5: return v, i
    if _detect_starts_anodic(v):
        turn = int(np.argmax(v))
        return v[turn:], i[turn:]
    else:
        turn = int(np.argmin(v))
        return v[:turn+1], i[:turn+1]
```

- **Cell 4:** Fungsi `extract_upper_peak_features()` dan `extract_lower_peak_features()` (8 fitur masing-masing, Savitzky-Golay smoothing, baseline correction, FWHM via scipy)
- **Cell 5:** Fungsi `process_single_file_v3()` + `build_dataset_v3()` — load semua folder PPM + KCL, print progress tiap ppm level, return DataFrame

Verifikasi: LC = 800 sampel, PS ≈ 797 sampel.

---

### Phase 2 — Preprocessing & Feature Engineering (Cell 6–8)

- **Cell 6 (md):** Header Phase 2
- **Cell 7:** Analisis NaN — print tabel NaN count + NaN% per fitur sebelum imputation
- **Cell 8:** `impute_features()` — kolom all-NaN → 0, partial-NaN → median.
  Simpan ke `output/dataset_LC_v3_imputed.csv` dan `output/dataset_PS_v3_imputed.csv`.
  Print: jumlah sampel, NaN sebelum/sesudah, distribusi per ppm.

---

### Phase 3 — EDA: Exploratory Data Analysis (Cell 9–17)

**WAJIB: tiap cell menghasilkan plot yang tampil. Simpan ke `output/fig_*.png`. Gunakan `plt.show()`.**

- **Cell 9 (md):** Header Phase 3
- **Cell 10:** Statistik deskriptif + tabel korelasi Pearson fitur vs ppm (LC dan PS)
- **Cell 11:** `fig1_heatmap_korelasi.png` — Heatmap korelasi 16 fitur + ppm (LC & PS side by side, 1×2)
- **Cell 12:** `fig2_calibration_curve.png` — upper_Ip_mean vs ppm, 2 subplot (linear scale & log scale), LC dan PS overlay
- **Cell 13:** `fig3_boxplot_upper_LC.png` — Boxplot 8 upper features per ppm level (LC), subplot 2×4
- **Cell 14:** `fig4_boxplot_upper_PS.png` — Boxplot 8 upper features per ppm level (PS), subplot 2×4
- **Cell 15:** `fig5_boxplot_lower.png` — Boxplot lower_Ip_mean per ppm level (LC vs PS overlay)
- **Cell 16:** `fig6_mutual_info.png` — Mutual Information ranking 16 fitur (barh chart, LC & PS)
- **Cell 17:** `fig7_kde_distribution.png` — KDE plot upper_Ip_mean per sensor pada ppm 0, 10, 100, 1000
- **Cell 18 (md):** Ringkasan temuan EDA (isi berdasarkan hasil plot)

---

### Phase 4 — Model Training & Hyperparameter Tuning (Cell 19–31)

- **Cell 19 (md):** Header Phase 4
- **Cell 20:** Persiapan — y = log10(ppm+1), StandardScaler, StratifiedKFold(5)
- **Cell 21:** Fungsi `evaluate_model()` → R², RMSE, MAE, MAPE di ruang ppm asli (inverse transform)
- **Cell 22:** Model 1–3: LinearRegression, Ridge(alpha=1), ElasticNet(alpha=0.1, l1_ratio=0.5)
- **Cell 23:** Model 4: Polynomial Ridge (degree=2)
- **Cell 24:** Model 5–7: SVR(C=500,gamma='scale'), RandomForest(n=300), ExtraTrees(n=300)
- **Cell 25:** Model 8: XGBoost + Optuna (50 trials)
- **Cell 26:** Model 9: LightGBM + Optuna (50 trials)
- **Cell 27:** Model 10: CatBoost + Optuna (50 trials)
- **Cell 28:** Model 11: TabICLv2 (`from tabicl import TabICLRegressor`)
- **Cell 29:** Model 12: CAFF (PyTorch custom)

```python
import torch, torch.nn as nn
class CAFF(nn.Module):
    def __init__(self, n_features=16, hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1))
    def forward(self, x):
        enc = self.encoder(x).unsqueeze(1)
        out, _ = self.attn(enc, enc, enc)
        return self.decoder(out.squeeze(1)).squeeze(-1)
```

- **Cell 30:** Model 13: CAFF-Optuna (CAFF + Optuna 50 trials: hidden, n_heads, dropout, lr, epochs)
- **Cell 31:** Tabel ringkasan 13 model — R², RMSE, MAE, MAPE (LC dan PS)

Optuna search space: n_estimators[100–500], max_depth[3–10], learning_rate[0.01–0.3 log], subsample[0.6–1.0], colsample[0.6–1.0], reg_alpha/lambda[1e-5–10 log]

---

### Phase 5 — Ensemble & Stacking (Cell 32–34)

- **Cell 32 (md):** Header Phase 5
- **Cell 33:** VotingRegressor (uniform): XGBoost_tuned + LightGBM_tuned + CatBoost_tuned + ExtraTrees
- **Cell 34:** StackingRegressor: base=[XGB, LGB, CatBoost, RF], meta=Ridge(alpha=1.0)

---

### Phase 6 — Evaluasi & Komparasi Final (Cell 35–42)

- **Cell 35 (md):** Header Phase 6
- **Cell 36:** `fig8_r2_comparison.png` — Grouped bar chart R² semua model (LC vs PS)
- **Cell 37:** `fig9_rmse_comparison.png` — Grouped bar chart RMSE semua model
- **Cell 38:** `fig10_parity_plot.png` — Parity plot Actual vs Predicted ppm (model terbaik LC + PS)
- **Cell 39:** `fig11_residual.png` — Residual plot per rentang ppm (0–10, 10–100, 100–1000)
- **Cell 40:** `fig12_shap.png` — SHAP feature importance (atau RF feature importance jika shap error)
- **Cell 41:** Tabel komparasi final LC vs PS + print model terbaik + kesimpulan singkat
- **Cell 42 (md):** Dashboard ringkasan akhir

---

## ATURAN PENTING

1. Jalankan semua cell — tidak boleh ada cell tanpa output
2. Phase 3 EDA WAJIB menghasilkan 7 figure nyata (tampil di output cell)
3. Semua figure disimpan ke folder `output/`
4. Nama notebook: `Sensor_Analysis_v3.ipynb`
5. Gunakan `plt.show()` setelah setiap plot
6. Jangan gunakan `df.style.*` — gunakan `display(df)` atau `print(df.to_string())`
7. Install sebelum mulai: `pip install catboost optuna shap tabicl torch`
