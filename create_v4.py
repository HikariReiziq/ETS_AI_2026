import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1 (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# PROMPT LANJUTAN UTS AI A — Sensor LC vs PS Cyclic Voltammetry
## Part 4 — Feature Engineering Lanjutan + Advanced Modeling
### Phase 1: Load Dataset & Verifikasi"""))

# Cell 2: Import library + konfigurasi
cells.append(nbf.v4.new_code_cell("""import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import stats
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path('.')
LC_DIR   = BASE_DIR / 'Alat Sensor LC'
PS_DIR   = BASE_DIR / 'Alat Sensor PS'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

PPM_LEVELS   = [2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]
STABLE_SCANS = list(range(2, 10))   # scan indeks 2-9 (8 scan stabil per file)

# Window per-sensor untuk Anodic (Upper) Peak
UPPER_V_MIN_LC, UPPER_V_MAX_LC = -0.62, -0.43
UPPER_V_MIN_PS, UPPER_V_MAX_PS = -0.80, -0.43

# Window per-sensor untuk Cathodic (Lower) Peak
LOWER_V_MIN_LC, LOWER_V_MAX_LC = -0.80, -0.40
LOWER_V_MIN_PS, LOWER_V_MAX_PS = -0.90, -0.45
"""))

# Cell 3: load_cv_file + get_anodic_sweep + get_cathodic_sweep
cells.append(nbf.v4.new_code_cell("""def load_cv_file(filepath, sensor_type='LC'):
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

def get_anodic_sweep(v, i, starts_anodic=True):
    if len(v) < 5: return v, i
    if starts_anodic:
        # LC: starts anodic -> anodic sweep = awal sampai turn point (max)
        turn_idx = np.argmax(v)
        return v[:turn_idx+1], i[:turn_idx+1]
    else:
        # PS: starts cathodic -> anodic sweep = turn point (min) sampai akhir
        turn_idx = np.argmin(v)
        return v[turn_idx:], i[turn_idx:]

def get_cathodic_sweep(v, i, starts_anodic=True):
    if len(v) < 5: return v, i
    if starts_anodic:
        # LC: starts anodic -> cathodic sweep = setelah turn point (max)
        turn_idx = np.argmax(v)
        return v[turn_idx:], i[turn_idx:]
    else:
        # PS: starts cathodic -> cathodic sweep = sebelum turn point (min)
        turn_idx = np.argmin(v)
        return v[:turn_idx+1], i[:turn_idx+1]
"""))

# Cell 4: Fungsi extract_peak_features_v2
cells.append(nbf.v4.new_code_cell("""def calc_fwhm(vw, iw):
    if len(iw) < 3: return float('nan')
    peak_idx = np.argmax(iw)
    half_max = iw[peak_idx] / 2.0
    left = np.where(iw[:peak_idx] <= half_max)[0]
    right = np.where(iw[peak_idx:] <= half_max)[0]
    if len(left) == 0 or len(right) == 0:
        return float('nan')
    v_left = vw[left[-1]]
    v_right = vw[peak_idx + right[0]]
    return abs(v_right - v_left)

def calc_fwhm_cathodic(vw, iw_corrected):
    # For cathodic, iw_corrected is already flipped to be positive
    return calc_fwhm(vw, iw_corrected)

def extract_upper_peak(v, i, v_min, v_max):
    mask = (v >= v_min) & (v <= v_max)
    vw, iw = v[mask], i[mask]
    if len(vw) < 10:
        return {k: np.nan for k in ['upper_Ip_mean','upper_Ip_std','upper_Ep_mean','upper_Ep_std','upper_Area_mean','upper_FWHM_mean','upper_skewness','upper_kurtosis']}
    
    iw = savgol_filter(iw, window_length=9, polyorder=3)
    baseline = np.percentile(iw, 5)
    iw_corr = iw - baseline
    
    peak_idx = np.argmax(iw_corr)
    Ip = iw_corr[peak_idx]
    Ep = vw[peak_idx]
    Area = np.trapz(iw_corr, x=vw) if hasattr(np, 'trapz') else np.trapezoid(iw_corr, x=vw)
    FWHM = calc_fwhm(vw, iw_corr)
    skewness = float(stats.skew(iw))
    kurtosis = float(stats.kurtosis(iw))
    
    return {
        'Ip': Ip, 'Ep': Ep, 'Area': Area, 'FWHM': FWHM, 
        'skewness': skewness, 'kurtosis': kurtosis
    }

def extract_lower_peak(v, i, v_min, v_max):
    mask = (v >= v_min) & (v <= v_max)
    vw, iw = v[mask], i[mask]
    if len(vw) < 10:
        return {k: np.nan for k in ['lower_Ip_mean','lower_Ip_std','lower_Ep_mean','lower_Ep_std','lower_Area_mean','lower_FWHM_mean','lower_skewness','lower_kurtosis']}
    
    iw = savgol_filter(iw, window_length=9, polyorder=3)
    baseline = np.percentile(iw, 95)
    iw_corr = baseline - iw # flip agar positif
    
    # Karena kita sudah flip, peak_idx adalah argmax(iw_corr)
    peak_idx = np.argmax(iw_corr)
    Ip = iw_corr[peak_idx] # ini abs(current)
    Ep = vw[peak_idx]
    
    # Urutkan berdasarkan V untuk trapz agar Area positif jika V tidak urut menaik
    sort_idx = np.argsort(vw)
    Area = np.trapz(iw_corr[sort_idx], x=vw[sort_idx]) if hasattr(np, 'trapz') else np.trapezoid(iw_corr[sort_idx], x=vw[sort_idx])
    Area = abs(Area)
    
    FWHM = calc_fwhm_cathodic(vw, iw_corr)
    skewness = float(stats.skew(iw))
    kurtosis = float(stats.kurtosis(iw))
    
    return {
        'Ip': Ip, 'Ep': Ep, 'Area': Area, 'FWHM': FWHM, 
        'skewness': skewness, 'kurtosis': kurtosis
    }
"""))

# Cell 5: process_single_file_v2 dan build_dataset_v2
cells.append(nbf.v4.new_code_cell("""def process_single_file_v2(filepath, sensor_type, ppm_level):
    scans = load_cv_file(filepath, sensor_type)
    if not scans:
        return None
        
    starts_anodic = True if sensor_type == 'LC' else False
    
    u_v_min, u_v_max = (UPPER_V_MIN_LC, UPPER_V_MAX_LC) if sensor_type == 'LC' else (UPPER_V_MIN_PS, UPPER_V_MAX_PS)
    l_v_min, l_v_max = (LOWER_V_MIN_LC, LOWER_V_MAX_LC) if sensor_type == 'LC' else (LOWER_V_MIN_PS, LOWER_V_MAX_PS)
    
    u_Ips, u_Eps, u_Areas, u_FWHMs, u_skews, u_kurts = [], [], [], [], [], []
    l_Ips, l_Eps, l_Areas, l_FWHMs, l_skews, l_kurts = [], [], [], [], [], []
    
    for s in STABLE_SCANS:
        if s not in scans: continue
        v_raw, i_raw = scans[s]['V'], scans[s]['I']
        
        v_ano, i_ano = get_anodic_sweep(v_raw, i_raw, starts_anodic)
        v_cat, i_cat = get_cathodic_sweep(v_raw, i_raw, starts_anodic)
        
        u_feat = extract_upper_peak(v_ano, i_ano, u_v_min, u_v_max)
        l_feat = extract_lower_peak(v_cat, i_cat, l_v_min, l_v_max)
        
        if not np.isnan(u_feat.get('Ip', np.nan)):
            u_Ips.append(u_feat['Ip'])
            u_Eps.append(u_feat['Ep'])
            u_Areas.append(u_feat['Area'])
            u_FWHMs.append(u_feat['FWHM'])
            u_skews.append(u_feat['skewness'])
            u_kurts.append(u_feat['kurtosis'])
            
        if not np.isnan(l_feat.get('Ip', np.nan)):
            l_Ips.append(l_feat['Ip'])
            l_Eps.append(l_feat['Ep'])
            l_Areas.append(l_feat['Area'])
            l_FWHMs.append(l_feat['FWHM'])
            l_skews.append(l_feat['skewness'])
            l_kurts.append(l_feat['kurtosis'])
            
    if len(u_Ips) == 0:
        return None
        
    return {
        'ppm': ppm_level,
        'upper_Ip_mean': np.nanmean(u_Ips) if len(u_Ips) > 0 else np.nan,
        'upper_Ip_std': np.nanstd(u_Ips) if len(u_Ips) > 0 else np.nan,
        'upper_Ep_mean': np.nanmean(u_Eps) if len(u_Eps) > 0 else np.nan,
        'upper_Ep_std': np.nanstd(u_Eps) if len(u_Eps) > 0 else np.nan,
        'upper_Area_mean': np.nanmean(u_Areas) if len(u_Areas) > 0 else np.nan,
        'upper_FWHM_mean': np.nanmean(u_FWHMs) if len(u_FWHMs) > 0 else np.nan,
        'upper_skewness': np.nanmean(u_skews) if len(u_skews) > 0 else np.nan,
        'upper_kurtosis': np.nanmean(u_kurts) if len(u_kurts) > 0 else np.nan,
        
        'lower_Ip_mean': np.nanmean(l_Ips) if len(l_Ips) > 0 else np.nan,
        'lower_Ip_std': np.nanstd(l_Ips) if len(l_Ips) > 0 else np.nan,
        'lower_Ep_mean': np.nanmean(l_Eps) if len(l_Eps) > 0 else np.nan,
        'lower_Ep_std': np.nanstd(l_Eps) if len(l_Eps) > 0 else np.nan,
        'lower_Area_mean': np.nanmean(l_Areas) if len(l_Areas) > 0 else np.nan,
        'lower_FWHM_mean': np.nanmean(l_FWHMs) if len(l_FWHMs) > 0 else np.nan,
        'lower_skewness': np.nanmean(l_skews) if len(l_skews) > 0 else np.nan,
        'lower_kurtosis': np.nanmean(l_kurts) if len(l_kurts) > 0 else np.nan,
    }

def build_dataset_v2(sensor_dir, sensor_type):
    data = []
    
    # KCL (0 ppm)
    kcl_dir = sensor_dir / 'KCL'
    if kcl_dir.exists():
        for f in kcl_dir.glob('*.xlsx'):
            res = process_single_file_v2(f, sensor_type, 0)
            if res: data.append(res)
            
    # Other PPM levels
    for ppm in PPM_LEVELS:
        ppm_dir = sensor_dir / f'{ppm} ppm'
        if not ppm_dir.exists(): ppm_dir = sensor_dir / f'{ppm} PPM'
        if not ppm_dir.exists(): ppm_dir = sensor_dir / f'{ppm}ppm'
        if not ppm_dir.exists(): ppm_dir = sensor_dir / str(ppm)
            
        if ppm_dir.exists():
            for f in ppm_dir.glob('*.xlsx'):
                res = process_single_file_v2(f, sensor_type, ppm)
                if res: data.append(res)
        else:
            print(f"Warning: Directory not found for {ppm} ppm in {sensor_dir}")
    
    return pd.DataFrame(data)
"""))

# Cell 6: Load dataset LC dan PS
cells.append(nbf.v4.new_code_cell("""print("Memproses dataset LC...")
df_lc = build_dataset_v2(LC_DIR, 'LC')
print(f"Dataset LC selesai: {len(df_lc)} sampel")

print("Memproses dataset PS...")
df_ps = build_dataset_v2(PS_DIR, 'PS')
print(f"Dataset PS selesai: {len(df_ps)} sampel")
"""))

# Cell 7: Simpan ke CSV, tampilkan statistik, verifikasi
cells.append(nbf.v4.new_code_cell("""# Simpan ke CSV
df_lc.to_csv(OUTPUT_DIR / 'dataset_lc_v4_raw.csv', index=False)
df_ps.to_csv(OUTPUT_DIR / 'dataset_ps_v4_raw.csv', index=False)

# Impute features with median per column
def impute_missing(df):
    feature_cols = [c for c in df.columns if c.startswith('upper_') or c.startswith('lower_')]
    for col in feature_cols:
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

df_lc_imputed = impute_missing(df_lc.copy())
df_ps_imputed = impute_missing(df_ps.copy())

df_lc_imputed.to_csv(OUTPUT_DIR / 'dataset_lc_v4_imputed.csv', index=False)
df_ps_imputed.to_csv(OUTPUT_DIR / 'dataset_ps_v4_imputed.csv', index=False)

# Tampilkan Statistik LC
print("=== Statistik LC ===")
print("Shape:", df_lc.shape)
print("Missing values di raw LC:\\n", df_lc.isna().sum()[df_lc.isna().sum() > 0])

# Tampilkan Statistik PS
print("\\n=== Statistik PS ===")
print("Shape:", df_ps.shape)
print("Missing values di raw PS:\\n", df_ps.isna().sum()[df_ps.isna().sum() > 0])

# Verifikasi
assert df_lc.shape[0] == 800, f"Expected 800 samples for LC, got {df_lc.shape[0]}"
assert 790 <= df_ps.shape[0] <= 800, f"Expected ~797 samples for PS, got {df_ps.shape[0]}"
print("\\nVERIFIKASI BERHASIL: Semua kriteria Phase 1 terpenuhi.")
"""))

nb['cells'] = cells
with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook Sensor_Analysis_v4.ipynb created successfully.")
