# PROMPT LANJUTAN UTS AI A — Sensor LC vs PS Cyclic Voltammetry
## Sesi Baru: Part 2 — Feature Engineering Lanjutan + Advanced Modeling

---

## KONTEKS PROYEK

**Workspace:** `C:\Users\Hikar\Documents\Kuliah\Bahan Belajar Kuliah Semester 4\AI\UTS Kelas AI A`

**Soal UTS (Soal UTS AI A.pdf):**
Membandingkan kinerja dua sensor elektrokimia untuk deteksi logam berat Kadmium (Cd²⁺) menggunakan teknik Cyclic Voltammetry (CV). Sensor yang dibandingkan:
- **LC (LecSens):** Sensor buatan sendiri (custom)
- **PS (EmStat4):** Sensor komersial (Palmsens EmStat4)

Konsentrasi: 0–1000 ppm (16 level: KCL/0, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000 ppm)
Replikasi: 50 file per level konsentrasi.

**Penjelasan Dataset (Penejelasan data.pdf):**
- Setiap file `.xlsx` berisi 10 scan CV (kolom berpasangan V, I)
- LC: scan dimulai ANODIC (V naik -1.1 → -0.5V), stripping peak Cd²⁺ di ≈ -0.52V
- PS: scan dimulai CATHODIC (V turun -0.5 → -1.4V), anodic return setelah argmin(V)
- KCL folder = background elektrolit 0.1M KCl tanpa Cd²⁺ (ppm=0)
- Scan stabil yang dipakai: scan indeks 2–9 (8 scan per file)

---

## RINGKASAN PEKERJAAN YANG SUDAH DIKERJAKAN (Part 1 — Sensor_Analysis.ipynb)

### Dataset berhasil di-load:
- **LC:** 800 sampel (50 file × 16 level termasuk KCL ppm=0)
- **PS:** 797 sampel (49–50 file × 16 level)

### Preprocessing yang sudah berjalan:
1. `load_cv_file()` — baca Excel multi-sheet, ekstrak scan V,I
2. `get_anodic_sweep()` — adaptif: LC ambil awal scan, PS ambil setelah argmin(V)
3. `extract_peak_features()` — window per sensor:
   - LC: `[-0.62, -0.43]` V (sempit, hindari spurious peak di -0.67V)
   - PS: `[-0.80, -0.43]` V (lebar, akomodasi peak shift -0.50→-0.70V)
   - Baseline correction: 5th-percentile subtraction
4. `process_single_file()` → averaging 8 scan stabil per file

### Fitur yang sudah diekstrak (4 fitur di Part 1):
| Fitur | Deskripsi |
|-------|-----------|
| `Ip_mean` | Mean peak current (µA) — sinyal utama Cd²⁺ |
| `Ip_std` | Std dev Ip antar scan → noise/repeatabilitas |
| `Ep_mean` | Mean peak potential (V) → Cd²⁺ ≈ -0.52V (LC) |
| `Area_mean` | Integral faradaic (µA·V) |

**Target:** `ppm` (0–1000, regresi)

### Hasil Evaluasi Part 1 (5-Fold CV, 7 model):
| Sensor | Model Terbaik | R² | RMSE (ppm) | MAPE |
|--------|--------------|-----|------------|------|
| **LC** | XGBoost | 0.823 | 127.85 | 520% |
| **PS** | XGBoost | 0.579 | 199.54 | 666% |

**Root cause akurasi rendah:**
- Hanya 4 fitur → terlalu sedikit informasi
- MAPE tinggi karena range 0–1000 ppm (error kecil di ppm=2 = MAPE besar)
- PS slope kalibrasi linear = NEGATIF → mekanisme non-linear kompleks
- Tidak ada hyperparameter optimization (default params)

### Visualisasi yang sudah dibuat (13 figur):
fig1: Voltammogram CV comparison | fig2: Calibration curve | fig3: Boxplot Ip/ppm | 
fig4: Correlation matrix | fig5: Metric comparison bar | fig6: R² boxplot per fold | 
fig7: Parity plot | fig8: Residual analysis | fig9: Feature importance | 
fig10: Linear calibration | fig11: RSD% repeatability | fig12: LC vs PS scatter | 
fig13: Dashboard summary

---

## INSPIRASI DARI PAPER DOSEN (CAFF-IBWO, Taufiq Choirul Amri et al., Chemical Engineering Journal Advances 2026)

**Paper:** *"Cross-Attention Feature Fusion network for robust estimation of Cd²⁺ and Pb²⁺ in water samples using Cyclic Voltammetry"*
**File:** `Contoh_lain_paper_sebagai_referensi_ScienceDirect_articles_01May2026_04-15-33.949/Cross-Attention-Feature-Fusion-network-for-robust-esti_2026_Chemical-Enginee.pdf`

**Key findings dari paper:**
- CAFF-IBWO: R²=0.97 untuk Cd²⁺ (vs kita 0.82 saat ini)
- Mereka menggunakan **14 fitur** (7 per peak = upper + lower peak): 
  peak_presence (binary), start_current, peak_current, end_current, peak_area, skewness, kurtosis
- **Dual-stream architecture**: raw CV data (query channel) + extracted peak features (key/value channel)
- **IBWO** untuk hyperparameter optimization (chaotic initialization + adaptive exploration)
- **Stratified 5-fold CV**
- Cross-attention fusion meningkatkan 6-9% MSE vs single-stream
- Robustness dievaluasi menggunakan CIDS (Chemically-Informed Degradation Simulation)

**Takeaway untuk kita:**
- Gunakan **dual-peak approach** seperti paper: ekstrak fitur dari BOTH upper peak (anodic/stripping) DAN lower peak (cathodic/deposition)
- Target: **16 fitur per dataset** (8 upper peak + 8 lower peak) — sama persis dengan pendekatan teman dan mendekati paper (14 fitur)
- Gunakan Optuna sebagai pengganti IBWO (lebih mudah implementasi di Python)
- Stratified KFold penting untuk dataset imbalanced per ppm level
- TabICLv2 tersedia via `pip install tabicl` (sklearn-compatible, gratis, CPU-friendly)
- CAFF bisa diimplementasi dari scratch dengan PyTorch (~50 baris kode)
- CAFF-IBWO → implementasi sebagai CAFF + Optuna tuning (hasil setara)

---

## DISKUSI JUMLAH FITUR (8 vs 16)

**Pertanyaan:** Teman menggunakan 16 fitur, mana yang lebih baik?

**Jawaban: 16 fitur lebih baik** untuk kasus ini. Alasannya:

| Aspek | 8 Fitur (single-peak) | 16 Fitur (dual-peak) |
|-------|----------------------|---------------------|
| Sumber informasi | Hanya anodic stripping peak | Anodic + cathodic peak |
| Kesesuaian paper dosen | Tidak (paper pakai 14 fitur dual) | Ya (mendekati 14 fitur paper) |
| Informasi tambahan | — | Deposition kinetics dari lower peak |
| Robustness | Lebih rentan noise | Lebih redundant, noise tolerant |
| Kompleksitas kode | Sederhana | Moderate (perlu extract 2 windows) |

**Kenapa dual-peak lebih informatif secara elektrokimia:**
- Upper peak (anodic/stripping): mengukur seberapa banyak Cd²⁺ ter-strip → langsung proporsional dengan konsentrasi
- Lower peak (cathodic/deposition): mengukur seberapa banyak Cd²⁺ ter-deposit → informasi komplementer
- Rasio upper/lower peak bisa jadi fitur implisit yang menangkap non-linearitas

**Catatan implementasi:**
- Lower peak bisa lemah/tidak terdeteksi pada ppm rendah (0, 2, 4 ppm)
- Solusi: jika lower peak tidak terdeteksi, isi NaN → impute dengan median (SimpleImputer)
- Jangan drop sampel yang lower peak-nya NaN

---

## KLARIFIKASI MODEL LANJUTAN (CAFF, CAFF-IBWO, TabICLv2)

| Model | Tersedia? | Cara Install | Catatan |
|-------|-----------|-------------|--------|
| **TabICLv2** | ✅ Ya | `pip install tabicl` | Foundation model, sklearn API, download checkpoint sekali |
| **CAFF** | ✅ Implementasi manual | `pip install torch` | Custom PyTorch ~50 baris, dari paper dosen |
| **CAFF-IBWO** | ⚠️ Partial | `pip install torch optuna` | CAFF-Optuna (IBWO diganti Optuna, hasil setara) |
| **TabPFN** | ✅ Ya | `pip install tabpfn` | Alternatif TabICLv2, butuh login PriorLabs |

**Arsitektur CAFF sederhana (PyTorch):**
```python
import torch
import torch.nn as nn

class CAFF(nn.Module):
    def __init__(self, n_features=16, hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        # Encoder: project features to hidden dim
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Cross-attention: Q dari raw features, K/V dari encoded features
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        # Decoder: hidden dim → 1 output
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, x):
        # x: [batch, n_features]
        enc = self.encoder(x).unsqueeze(1)  # [batch, 1, hidden]
        attn_out, _ = self.attn(enc, enc, enc)  # self-attention
        out = self.decoder(attn_out.squeeze(1))  # [batch, 1]
        return out.squeeze(-1)
```

**Waktu estimasi training Part 2 (tanpa GPU):**
- Phase 1 (Load data): ~5 menit
- Phase 2 (Feature engineering): ~15 menit
- Phase 3 (EDA): ~5 menit
- Phase 4 (10 sklearn/xgb models + Optuna): ~30 menit
- Phase 4 (TabICLv2, download pertama kali): ~10 menit (download) + 2 menit (inference)
- Phase 4 (CAFF training 200 epochs): ~10 menit per fold × 5 = ~50 menit
- Phase 4 (CAFF-Optuna 50 trials × 5 folds): ~3-4 jam
- Phase 5-6 (Ensemble + Evaluasi): ~20 menit

⚠️ **REKOMENDASI**: Jika waktu terbatas, skip CAFF-Optuna (ganti dengan CAFF fixed params) untuk hemat 3 jam.

---

## KLARIFIKASI SARAN DARI MAHASISWA

| Saran Mahasiswa | Status | Penjelasan |
|-----------------|--------|------------|
| Cross Validation 5-Fold | ✅ Tetap pakai | Sudah ada di Part 1, pertahankan |
| Silhouette Score | ❌ Tidak dipakai | Ini metrik clustering (unsupervised), kita regresi |
| Epoch training | ✅ Khusus neural net | Hanya relevan untuk MLP/TabNet jika ditambahkan |
| ROC-AUC Curve | ❌ Tidak relevan | Ini untuk klasifikasi, kita regresi kontinu |
| Perbanyak fitur → 16 | ✅ Dilanjutkan | Dual-peak (upper+lower), sesuai paper dosen & teman |
| Ensemble model | ✅ Ditambahkan | Stacking + Voting ensemble |

**Metrik evaluasi regresi yang tepat:**
- R² (coefficient of determination) — makin tinggi makin baik
- RMSE (root mean squared error) — makin rendah makin baik  
- MAE (mean absolute error) — makin rendah makin baik
- MAPE% (mean absolute percentage error) — makin rendah makin baik
- Parity Plot (Actual vs Predicted scatter) — visualisasi utama
- Residual Plot — deteksi bias/heteroscedasticity

---

## RENCANA PART 2 — Pipeline Baru di Notebook Baru

**Buat notebook baru:** `Sensor_Analysis_v2.ipynb`

### PIPELINE LENGKAP:

```
[Phase 1] Load Dataset
    ├── Baca file xlsx dari Alat Sensor LC/ dan Alat Sensor PS/
    ├── Include KCL (ppm=0) sebagai baseline (folder KCL/)
    │   LC KCL: 50 file (KCL_LECSENS_01.xlsx ... KCL_LECSENS_050.xlsx)
    │   PS KCL: 50 file (KCL_EMSTAT_01.xlsx ... KCL_EMSTAT_50.xlsx)
    └── Verifikasi: LC=800 sampel, PS≈797 sampel

[Phase 2] Preprocessing & Feature Engineering (EKSPANSI KE 8 FITUR)
    ├── Savitzky-Golay smoothing (window=9, polyorder=3)
    ├── Adaptif anodic sweep detection:
    │   LC: ambil dari awal hingga turn point (starts_anodic=True)
    │   PS: ambil dari argmin(V) hingga akhir (starts_anodic=False)
    ├── Window per-sensor:
    │   LC: v_min=-0.62, v_max=-0.43 V
    │   PS: v_min=-0.80, v_max=-0.43 V
    ├── Baseline: background = np.percentile(iw, 5), Ip = Ip_raw - background
    ├── Window per-sensor (UPPER/anodic peak):
    │   LC: v_min=-0.62, v_max=-0.43 V
    │   PS: v_min=-0.80, v_max=-0.43 V
    ├── Window per-sensor (LOWER/cathodic peak) — deteksi dari cathodic sweep:
    │   LC: v_min=-0.80, v_max=-0.40 V (return sweep setelah turn point)
    │   PS: v_min=-0.90, v_max=-0.45 V (forward cathodic sweep, sebelum argmin)
    └── Ekstrak 16 fitur per file (rata-rata 8 scan stabil, DUAL-PEAK approach):
        === UPPER PEAK (Anodic/Stripping) ===
        (1)  upper_Ip_mean     — mean peak current anodic across scans
        (2)  upper_Ip_std      — std peak current anodic across scans
        (3)  upper_Ep_mean     — mean peak potential anodic across scans
        (4)  upper_Ep_std      — std peak potential anodic across scans
        (5)  upper_Area_mean   — mean trapezoidal integral above baseline (anodic)
        (6)  upper_FWHM_mean   — mean full-width half-maximum (anodic)
        (7)  upper_skewness    — skewness distribusi arus di window anodic
        (8)  upper_kurtosis    — kurtosis distribusi arus di window anodic
        === LOWER PEAK (Cathodic/Deposition) ===
        (9)  lower_Ip_mean     — mean |peak current| cathodic across scans
        (10) lower_Ip_std      — std peak current cathodic across scans
        (11) lower_Ep_mean     — mean peak potential cathodic across scans
        (12) lower_Ep_std      — std peak potential cathodic across scans
        (13) lower_Area_mean   — mean integral above baseline (cathodic)
        (14) lower_FWHM_mean   — mean full-width half-maximum (cathodic)
        (15) lower_skewness    — skewness distribusi arus di window cathodic
        (16) lower_kurtosis    — kurtosis distribusi arus di window cathodic
        CATATAN: Jika lower peak tidak terdeteksi (low concentration), isi NaN →
        impute dengan median per ppm level sebelum modeling

[Phase 3] EDA (Exploratory Data Analysis)
    ├── Statistik deskriptif per fitur per ppm level
    ├── Heatmap korelasi fitur (8 fitur + ppm)
    ├── Boxplot setiap fitur per ppm level
    └── Calibration curve: Ip_mean vs ppm (linear + log scale)

[Phase 4] Model Training & Hyperparameter Tuning
    ├── TARGET TRANSFORMATION: y_train = log10(ppm + 1)
    │   (inverse: ppm_pred = 10^y_pred - 1)
    │   Alasan: normalisasi skala, MAPE lebih bermakna, model mudah belajar
    ├── Feature scaling: StandardScaler
    ├── Stratified KFold (n_splits=5) berdasarkan ppm bins
    ├── 13 model regresi:
    │   === BASELINE LINEAR ===
    │   (1)  LinearRegression
    │   (2)  Ridge (alpha=1.0)
    │   (3)  ElasticNet (alpha=0.1, l1_ratio=0.5)
    │   (4)  Polynomial Ridge (degree=2)
    │   === KERNEL / SVM ===
    │   (5)  SVR (RBF, C=500, gamma='scale')
    │   === TREE ENSEMBLE ===
    │   (6)  RandomForestRegressor (n_estimators=300)
    │   (7)  ExtraTreesRegressor (n_estimators=300)
    │   (8)  XGBoost + Optuna tuning (50 trials)
    │   (9)  LightGBM + Optuna tuning (50 trials)
    │   (10) CatBoost + Optuna tuning (50 trials)
    │   === FOUNDATION MODEL ===
    │   (11) TabICLv2 (TabICLRegressor, no tuning needed, zero-shot ICL)
    │         pip install tabicl → from tabicl import TabICLRegressor
    │         Catatan: download checkpoint ~300MB dari HuggingFace pada pertama kali
    │   === DEEP LEARNING ===
    │   (12) CAFF (Cross-Attention Feature Fusion) — implementasi PyTorch custom
    │         Arsitektur: FC encoder → Multi-Head Attention → FC decoder
    │         Input: 16 fitur, hidden=64, heads=4, epochs=200
    │   (13) CAFF-Optuna (CAFF + Optuna hyperparameter tuning, pengganti CAFF-IBWO)
    │         Tuning: hidden_size [32,128], heads [2,4,8], lr [1e-4,1e-2], dropout [0,0.3]
    │         50 Optuna trials → pick best val R²
    ├── Optuna search space (untuk XGBoost, LightGBM, CatBoost):
    │   n_estimators: [100, 500]
    │   max_depth: [3, 10]
    │   learning_rate: [0.01, 0.3] (log)
    │   subsample: [0.6, 1.0]
    │   colsample_bytree/feature_fraction: [0.6, 1.0]
    │   reg_alpha, reg_lambda: [1e-5, 10.0] (log)
    ├── Optuna search space (untuk CAFF-Optuna):
    │   hidden_size: [32, 64, 128]
    │   n_heads: [2, 4, 8]
    │   learning_rate: [1e-4, 1e-2] (log)
    │   dropout: [0.0, 0.3]
    │   batch_size: [16, 32, 64]
    └── Simpan best params per model

[Phase 5] Ensemble & Stacking
    ├── VotingRegressor (mean): XGBoost + LightGBM + CatBoost + ExtraTrees
    │   (tuned versions)
    └── StackingRegressor:
        base_estimators = [XGBoost_tuned, LightGBM_tuned, CatBoost_tuned, CAFF_Optuna]
        meta_learner = Ridge(alpha=1.0)
        cv = 5

[Phase 6] Evaluasi & Komparasi Final
    ├── 5-Fold Stratified CV semua 15 model (13 base + 2 ensemble)
    ├── Metrik: R², RMSE (ppm), MAE (ppm), MAPE (%)
    │   CATATAN: semua metrik dihitung di space ppm (setelah inverse transform)
    ├── Visualisasi:
    │   - Bar chart perbandingan R² semua model (LC vs PS)
    │   - Boxplot R² per fold
    │   - Parity Plot model terbaik
    │   - Residual Plot per ppm range (0-10, 10-100, 100-1000)
    │   - Feature Importance (Random Forest + SHAP jika tersedia)
    │   - Tabel komparasi lengkap LC vs PS
    └── Dashboard ringkasan final
```

---

## CATATAN TEKNIS PENTING

### Penanganan ppm=0 (KCL):
```python
# log_ppm untuk KCL = NaN karena log10(0) tidak terdefinisi
# Untuk log-transform target: gunakan log10(ppm + 1) → log10(0+1) = 0 ✓
y = np.log10(df['ppm'] + 1)
# Inverse: ppm_pred = 10**y_pred - 1
```

### Kalkulasi FWHM:
```python
def calc_fwhm(vw, iw):
    peak_idx = np.argmax(iw)
    half_max = iw[peak_idx] / 2.0
    # Cari titik kiri dan kanan di mana arus = half_max
    left = np.where(iw[:peak_idx] <= half_max)[0]
    right = np.where(iw[peak_idx:] <= half_max)[0]
    if len(left) == 0 or len(right) == 0:
        return float('nan')
    v_left = vw[left[-1]]
    v_right = vw[peak_idx + right[0]]
    return abs(v_right - v_left)
```

### Kalkulasi Skewness & Kurtosis:
```python
from scipy import stats
skewness = float(stats.skew(iw))     # distribusi arus di window
kurtosis = float(stats.kurtosis(iw)) # excess kurtosis
```

### Ekstraksi Lower (Cathodic) Peak:
```python
def get_cathodic_sweep(v, i, starts_anodic=True):
    """Ambil bagian cathodic sweep dari CV data."""
    if starts_anodic:
        # LC: starts anodic → cathodic sweep = setelah V mencapai max
        turn_idx = np.argmax(v)
        return v[turn_idx:], i[turn_idx:]
    else:
        # PS: starts cathodic → cathodic sweep = sebelum argmin(V)
        turn_idx = np.argmin(v)
        return v[:turn_idx+1], i[:turn_idx+1]

# Cathodic window per sensor:
# LC: [-0.80, -0.40] V (dalam cathodic return sweep)
# PS: [-0.90, -0.45] V (dalam cathodic forward sweep)
# PENTING: untuk cathodic peak, current NEGATIF → cari argmin (bukan argmax)
def extract_lower_peak(vw, iw):
    """Ekstrak fitur dari cathodic/deposition peak."""
    if len(iw) == 0:
        return {k: np.nan for k in ['lower_Ip_mean','lower_Ip_std','lower_Ep_mean',
                                     'lower_Ep_std','lower_Area_mean','lower_FWHM_mean',
                                     'lower_skewness','lower_kurtosis']}
    peak_idx = np.argmin(iw)  # cathodic peak = minimum (most negative)
    baseline = np.percentile(iw, 95)  # untuk cathodic: 95th percentile sebagai baseline
    iw_corrected = baseline - iw  # flip agar positif untuk kalkulasi
    # ... lanjut seperti upper peak
```

### Library tambahan yang perlu diinstall:
```
pip install catboost optuna shap tabicl torch
```

**Catatan install:**
- `catboost` → CatBoost model
- `optuna` → hyperparameter tuning untuk XGBoost/LightGBM/CatBoost/CAFF
- `shap` → feature importance visualization
- `tabicl` → TabICLv2 foundation model (download checkpoint ~250MB dari HuggingFace saat pertama kali dipakai)
- `torch` → PyTorch untuk CAFF custom implementation
- TabICLv2 butuh internet saat pertama kali (download checkpoint). Setelah itu bisa offline.
- CAFF tidak perlu download apa-apa, diimplementasi dari scratch.

### Environment:
- Python 3.13 di `.venv` folder workspace
- Semua library lain sudah ada (numpy, pandas, sklearn, xgboost, lightgbm, scipy, matplotlib, seaborn)

---

## INSTRUKSI UNTUK AI ASSISTANT

**Kerjakan PELAN-PELAN, satu Phase per sesi.**

**MULAI DARI PHASE 1 DAHULU** (Load Dataset & verifikasi):

1. Buat notebook baru `Sensor_Analysis_v2.ipynb`
2. Cell 1 (Markdown): Judul dan deskripsi pipeline
3. Cell 2: Import library + konfigurasi
4. Cell 3: Fungsi `load_cv_file()` + `get_anodic_sweep()` (copy dari Part 1)
5. Cell 4: Fungsi `extract_peak_features_v2()` dengan 8 fitur baru
6. Cell 5: Fungsi `process_single_file_v2()` dan `build_dataset_v2()`
7. Cell 6: Load dataset LC dan PS
8. Cell 7: Simpan ke CSV, tampilkan statistik, verifikasi

**JANGAN lanjut ke Phase 2 sebelum:**
- Dataset LC = 800 sampel ✓
- Dataset PS ≈ 797 sampel ✓
- Semua 16 fitur diekstrak (upper + lower peak)
- NaN pada lower peak wajar untuk ppm rendah (0, 2, 4 ppm) — impute dengan SimpleImputer
- FWHM upper peak valid untuk ≥ 80% sampel

Setelah Phase 1 selesai dan terverifikasi, baru lanjut ke Phase 2 (EDA).

---

*Prompt ini dibuat: 1 Mei 2026*
*Diupdate: 1 Mei 2026 — tambah 16 fitur dual-peak, TabICLv2, CAFF, CAFF-Optuna*
*Part 1 selesai di: Sensor_Analysis.ipynb*
*Part 2 akan dikerjakan di: Sensor_Analysis_v2.ipynb*
