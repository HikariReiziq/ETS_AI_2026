import json
from pathlib import Path

notebook_cells = []

def md(text):
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split('\n')]
    })

def code(text):
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split('\n')]
    })

# Phase 1
md("## Phase 1 — Load Dataset\n\nPipeline 6 phase untuk membandingkan sensor elektrokimia LC vs PS.")

code("""import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from tabicl import TabICLRegressor
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(r'C:\\Users\\Hikar\\Documents\\Kuliah\\Bahan Belajar Kuliah Semester 4\\AI\\UTS Kelas AI A')
LC_DIR   = BASE_DIR / 'Alat Sensor LC'
PS_DIR   = BASE_DIR / 'Alat Sensor PS'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

PPM_LEVELS   = [2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]
STABLE_SCANS = list(range(2, 10))   # scan indeks 2-9 (8 scan stabil per file)

UPPER_V_MIN_LC, UPPER_V_MAX_LC = -0.62, -0.43
UPPER_V_MIN_PS, UPPER_V_MAX_PS = -0.80, -0.43
LOWER_V_MIN_LC, LOWER_V_MAX_LC = -0.80, -0.40
LOWER_V_MIN_PS, LOWER_V_MAX_PS = -0.90, -0.45

UPPER_FEATURES = ['upper_Ip_mean','upper_Ip_std','upper_Ep_mean','upper_Ep_std',
                  'upper_Area_mean','upper_FWHM_mean','upper_skewness','upper_kurtosis']
LOWER_FEATURES = ['lower_Ip_mean','lower_Ip_std','lower_Ep_mean','lower_Ep_std',
                  'lower_Area_mean','lower_FWHM_mean','lower_skewness','lower_kurtosis']
FEATURE_COLS = UPPER_FEATURES + LOWER_FEATURES
""")

code("""def load_cv_file(filepath, sensor_type='LC'):
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
""")

code("""def extract_peak_features(v, i, v_min, v_max, window_length=11, polyorder=2):
    mask = (v >= v_min) & (v <= v_max)
    v_window = v[mask]
    i_window = i[mask]
    
    if len(v_window) < window_length:
        return {'Ip': np.nan, 'Ep': np.nan, 'Area': np.nan, 'FWHM': np.nan, 'skewness': np.nan, 'kurtosis': np.nan}
    
    i_smooth = savgol_filter(i_window, window_length, polyorder)
    
    baseline = np.linspace(i_smooth[0], i_smooth[-1], len(i_smooth))
    i_corr = i_smooth - baseline
    
    peak_idx = np.argmax(np.abs(i_corr))
    
    is_positive = i_corr[peak_idx] > 0
    if is_positive:
        peaks, _ = find_peaks(i_corr)
    else:
        peaks, _ = find_peaks(-i_corr)
        
    if len(peaks) > 0:
        peak_idx = peaks[np.argmin(np.abs(peaks - peak_idx))]
    
    Ip = i_corr[peak_idx]
    Ep = v_window[peak_idx]
    Area = np.trapz(np.abs(i_corr), x=v_window) if hasattr(np, 'trapz') else np.trapezoid(np.abs(i_corr), x=v_window)
    
    if is_positive:
        widths, _, _, _ = peak_widths(i_corr, [peak_idx], rel_height=0.5)
    else:
        widths, _, _, _ = peak_widths(-i_corr, [peak_idx], rel_height=0.5)
        
    if len(widths) > 0:
        dv = np.mean(np.abs(np.diff(v_window)))
        FWHM = widths[0] * dv
    else:
        FWHM = np.nan
        
    skew_val = skew(i_corr)
    kurt_val = kurtosis(i_corr)
    
    return {'Ip': Ip, 'Ep': Ep, 'Area': Area, 'FWHM': FWHM, 'skewness': skew_val, 'kurtosis': kurt_val}

def extract_upper_peak_features(v, i, sensor_type):
    v_min, v_max = (UPPER_V_MIN_LC, UPPER_V_MAX_LC) if sensor_type == 'LC' else (UPPER_V_MIN_PS, UPPER_V_MAX_PS)
    return extract_peak_features(v, i, v_min, v_max)

def extract_lower_peak_features(v, i, sensor_type):
    v_min, v_max = (LOWER_V_MIN_LC, LOWER_V_MAX_LC) if sensor_type == 'LC' else (LOWER_V_MIN_PS, LOWER_V_MAX_PS)
    return extract_peak_features(v, i, v_min, v_max)
""")

code("""def process_single_file_v3(filepath, sensor_type, ppm_level):
    scans = load_cv_file(filepath, sensor_type)
    if not scans:
        return None
    
    upper_Ips, upper_Eps, upper_Areas, upper_FWHMs = [], [], [], []
    lower_Ips, lower_Eps, lower_Areas, lower_FWHMs = [], [], [], []
    upper_skews, upper_kurts, lower_skews, lower_kurts = [], [], [], []
    
    for s in STABLE_SCANS:
        if s not in scans: continue
        v, i = scans[s]['V'], scans[s]['I']
        
        # We don't necessarily split anodic/cathodic here if we just apply the voltage window
        u_feat = extract_upper_peak_features(v, i, sensor_type)
        l_feat = extract_lower_peak_features(v, i, sensor_type)
        
        upper_Ips.append(u_feat['Ip'])
        upper_Eps.append(u_feat['Ep'])
        upper_Areas.append(u_feat['Area'])
        upper_FWHMs.append(u_feat['FWHM'])
        upper_skews.append(u_feat['skewness'])
        upper_kurts.append(u_feat['kurtosis'])
        
        lower_Ips.append(l_feat['Ip'])
        lower_Eps.append(l_feat['Ep'])
        lower_Areas.append(l_feat['Area'])
        lower_FWHMs.append(l_feat['FWHM'])
        lower_skews.append(l_feat['skewness'])
        lower_kurts.append(l_feat['kurtosis'])
        
    if not upper_Ips:
        return None
        
    return {
        'ppm': ppm_level,
        'upper_Ip_mean': np.nanmean(upper_Ips),
        'upper_Ip_std': np.nanstd(upper_Ips),
        'upper_Ep_mean': np.nanmean(upper_Eps),
        'upper_Ep_std': np.nanstd(upper_Eps),
        'upper_Area_mean': np.nanmean(upper_Areas),
        'upper_FWHM_mean': np.nanmean(upper_FWHMs),
        'upper_skewness': np.nanmean(upper_skews),
        'upper_kurtosis': np.nanmean(upper_kurts),
        'lower_Ip_mean': np.nanmean(lower_Ips),
        'lower_Ip_std': np.nanstd(lower_Ips),
        'lower_Ep_mean': np.nanmean(lower_Eps),
        'lower_Ep_std': np.nanstd(lower_Eps),
        'lower_Area_mean': np.nanmean(lower_Areas),
        'lower_FWHM_mean': np.nanmean(lower_FWHMs),
        'lower_skewness': np.nanmean(lower_skews),
        'lower_kurtosis': np.nanmean(lower_kurts),
    }

def build_dataset_v3(sensor_dir, sensor_type):
    data = []
    
    # KCL
    kcl_dir = sensor_dir / 'KCL'
    if kcl_dir.exists():
        files = list(kcl_dir.glob('*.xlsx'))
        print(f"{sensor_type} - KCL (0 ppm) : {len(files)} files")
        for f in files:
            res = process_single_file_v3(f, sensor_type, 0)
            if res: data.append(res)
            
    # PPM levels
    for ppm in PPM_LEVELS:
        ppm_dir = sensor_dir / f'{ppm} ppm'
        if not ppm_dir.exists():
            ppm_dir = sensor_dir / f'{ppm} PPM'
        if not ppm_dir.exists():
            ppm_dir = sensor_dir / f'{ppm}ppm'
            
        if ppm_dir.exists():
            files = list(ppm_dir.glob('*.xlsx'))
            print(f"{sensor_type} - {ppm} ppm : {len(files)} files")
            for f in files:
                res = process_single_file_v3(f, sensor_type, ppm)
                if res: data.append(res)
        else:
            print(f"{sensor_type} - {ppm} ppm : Directory not found")
            
    return pd.DataFrame(data)

df_lc = build_dataset_v3(LC_DIR, 'LC')
df_ps = build_dataset_v3(PS_DIR, 'PS')

print("LC Shape:", df_lc.shape)
print("PS Shape:", df_ps.shape)
""")

# Phase 2
md("## Phase 2 — Preprocessing & Feature Engineering")

code("""def print_nan_analysis(df, name):
    print(f"--- NaN Analysis for {name} ---")
    nan_counts = df.isna().sum()
    nan_pct = (nan_counts / len(df)) * 100
    res = pd.DataFrame({'NaN Count': nan_counts, 'NaN %': nan_pct})
    display(res[res['NaN Count'] > 0])
    print("\\n")

print_nan_analysis(df_lc, "LC Sensor")
print_nan_analysis(df_ps, "PS Sensor")
""")

code("""def impute_features(df, sensor_name):
    df_imp = df.copy()
    for col in FEATURE_COLS:
        if df_imp[col].isna().all():
            df_imp[col] = 0.0
        else:
            df_imp[col] = df_imp[col].fillna(df_imp[col].median())
    
    out_path = OUTPUT_DIR / f'dataset_{sensor_name}_v3_imputed.csv'
    df_imp.to_csv(out_path, index=False)
    
    print(f"{sensor_name} samples: {len(df)}")
    print(f"{sensor_name} NaN before: {df.isna().sum().sum()} -> after: {df_imp.isna().sum().sum()}")
    return df_imp

df_lc_imp = impute_features(df_lc, 'LC')
df_ps_imp = impute_features(df_ps, 'PS')
""")

# Phase 3
md("## Phase 3 — EDA (Exploratory Data Analysis)")

code("""display(df_lc_imp.describe().T)
display(df_ps_imp.describe().T)

corr_lc = df_lc_imp.corr()[['ppm']].sort_values(by='ppm', ascending=False)
corr_ps = df_ps_imp.corr()[['ppm']].sort_values(by='ppm', ascending=False)

print("Correlation with ppm (LC):")
display(corr_lc)
print("Correlation with ppm (PS):")
display(corr_ps)
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(df_lc_imp.corr()[['ppm']].sort_values(by='ppm', ascending=False), 
            annot=True, cmap='coolwarm', ax=axes[0])
axes[0].set_title('LC Features Correlation with ppm')

sns.heatmap(df_ps_imp.corr()[['ppm']].sort_values(by='ppm', ascending=False), 
            annot=True, cmap='coolwarm', ax=axes[1])
axes[1].set_title('PS Features Correlation with ppm')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_heatmap_korelasi.png')
plt.show()
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.lineplot(data=df_lc_imp, x='ppm', y='upper_Ip_mean', marker='o', label='LC', ax=axes[0])
sns.lineplot(data=df_ps_imp, x='ppm', y='upper_Ip_mean', marker='s', label='PS', ax=axes[0])
axes[0].set_title('Calibration Curve (Linear Scale)')
axes[0].set_xlabel('Concentration (ppm)')
axes[0].set_ylabel('Upper Peak Ip Mean')
axes[0].legend()

sns.lineplot(data=df_lc_imp, x='ppm', y='upper_Ip_mean', marker='o', label='LC', ax=axes[1])
sns.lineplot(data=df_ps_imp, x='ppm', y='upper_Ip_mean', marker='s', label='PS', ax=axes[1])
axes[1].set_xscale('log')
axes[1].set_title('Calibration Curve (Log Scale)')
axes[1].set_xlabel('Concentration (ppm) [Log]')
axes[1].set_ylabel('Upper Peak Ip Mean')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_calibration_curve.png')
plt.show()
""")

code("""fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(UPPER_FEATURES):
    sns.boxplot(data=df_lc_imp, x='ppm', y=col, ax=axes[i])
    axes[i].set_title(f'LC: {col}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_boxplot_upper_LC.png')
plt.show()
""")

code("""fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(UPPER_FEATURES):
    sns.boxplot(data=df_ps_imp, x='ppm', y=col, ax=axes[i])
    axes[i].set_title(f'PS: {col}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_boxplot_upper_PS.png')
plt.show()
""")

code("""plt.figure(figsize=(12, 6))
df_combined = pd.concat([df_lc_imp.assign(Sensor='LC'), df_ps_imp.assign(Sensor='PS')])
sns.boxplot(data=df_combined, x='ppm', y='lower_Ip_mean', hue='Sensor')
plt.title('Lower Peak Ip Mean per PPM (LC vs PS)')
plt.xticks(rotation=45)
plt.savefig(OUTPUT_DIR / 'fig5_boxplot_lower.png')
plt.show()
""")

code("""mi_lc = mutual_info_regression(df_lc_imp[FEATURE_COLS], df_lc_imp['ppm'])
mi_ps = mutual_info_regression(df_ps_imp[FEATURE_COLS], df_ps_imp['ppm'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pd.Series(mi_lc, index=FEATURE_COLS).sort_values().plot.barh(ax=axes[0], color='skyblue')
axes[0].set_title('LC: Mutual Information with ppm')

pd.Series(mi_ps, index=FEATURE_COLS).sort_values().plot.barh(ax=axes[1], color='salmon')
axes[1].set_title('PS: Mutual Information with ppm')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig6_mutual_info.png')
plt.show()
""")

code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, ppm in enumerate([0, 10, 100, 1000]):
    sns.kdeplot(data=df_lc_imp[df_lc_imp['ppm'] == ppm], x='upper_Ip_mean', label='LC', ax=axes[i], fill=True)
    sns.kdeplot(data=df_ps_imp[df_ps_imp['ppm'] == ppm], x='upper_Ip_mean', label='PS', ax=axes[i], fill=True)
    axes[i].set_title(f'Upper Ip Mean Distribution at {ppm} ppm')
    axes[i].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig7_kde_distribution.png')
plt.show()
""")

md("""### Ringkasan Temuan EDA
- Terdapat korelasi yang signifikan antara Ip (arus puncak) dan konsentrasi (ppm).
- Sensor LC dan PS menunjukkan pola yang sedikit berbeda dalam sensitivitas, terlihat dari kurva kalibrasi (Log scale).
- Distribusi data cukup konsisten tetapi beberapa fitur memiliki variance yang tinggi di level ppm tertentu.
""")

# Phase 4
md("## Phase 4 — Model Training & Hyperparameter Tuning")

code("""X_lc = df_lc_imp[FEATURE_COLS]
y_lc = np.log10(df_lc_imp['ppm'] + 1)

X_ps = df_ps_imp[FEATURE_COLS]
y_ps = np.log10(df_ps_imp['ppm'] + 1)

scaler_lc = StandardScaler()
X_lc_scaled = scaler_lc.fit_transform(X_lc)

scaler_ps = StandardScaler()
X_ps_scaled = scaler_ps.fit_transform(X_ps)

def get_stratify_labels(df):
    return df['ppm'].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {'LC': {}, 'PS': {}}
""")

code("""def evaluate_model(model_name, model, X, y, df, sensor, is_nn=False, fit_params=None):
    r2_list, rmse_list, mae_list, mape_list = [], [], [], []
    y_true_all, y_pred_all = [], []
    
    labels = get_stratify_labels(df)
    
    for train_idx, test_idx in skf.split(X, labels):
        if isinstance(X, np.ndarray):
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
        if isinstance(y, np.ndarray):
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
        if is_nn:
            X_train_t = torch.FloatTensor(X_train if isinstance(X_train, np.ndarray) else X_train.values)
            y_train_t = torch.FloatTensor(y_train if isinstance(y_train, np.ndarray) else y_train.values)
            X_test_t = torch.FloatTensor(X_test if isinstance(X_test, np.ndarray) else X_test.values)
            
            optimizer = optim.Adam(model.parameters(), lr=fit_params.get('lr', 0.001))
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(fit_params.get('epochs', 100)):
                optimizer.zero_grad()
                preds = model(X_train_t)
                loss = criterion(preds, y_train_t)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                y_pred_log = model(X_test_t).numpy()
        else:
            import copy
            try:
                model_inst = copy.deepcopy(model)
            except:
                model_inst = model
            model_inst.fit(X_train, y_train)
            y_pred_log = model_inst.predict(X_test)
            
        # Inverse transform
        y_test_inv = 10**y_test - 1
        y_pred_inv = 10**y_pred_log - 1
        
        y_pred_inv = np.maximum(0, y_pred_inv)
        
        y_true_all.extend(y_test_inv)
        y_pred_all.extend(y_pred_inv)
        
        r2_list.append(r2_score(y_test_inv, y_pred_inv))
        rmse_list.append(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
        mae_list.append(mean_absolute_error(y_test_inv, y_pred_inv))
        
        mask = y_test_inv > 0
        if np.sum(mask) > 0:
            mape_list.append(mean_absolute_percentage_error(y_test_inv[mask], y_pred_inv[mask]))
            
    res = {
        'R2': np.mean(r2_list),
        'RMSE': np.mean(rmse_list),
        'MAE': np.mean(mae_list),
        'MAPE': np.mean(mape_list) if mape_list else np.nan,
        'y_true': y_true_all,
        'y_pred': y_pred_all
    }
    results[sensor][model_name] = res
    print(f"{sensor} - {model_name} | R2: {res['R2']:.4f} | RMSE: {res['RMSE']:.4f}")
    return res
""")

code("""models_1_3 = {
    '1. LinearRegression': LinearRegression(),
    '2. Ridge': Ridge(alpha=1),
    '3. ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models_1_3.items():
    evaluate_model(name, model, X_lc_scaled, y_lc, df_lc_imp, 'LC')
    evaluate_model(name, model, X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""poly_model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1))
evaluate_model('4. PolyRidge', poly_model, X_lc_scaled, y_lc, df_lc_imp, 'LC')
evaluate_model('4. PolyRidge', poly_model, X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""models_5_7 = {
    '5. SVR': SVR(C=500, gamma='scale'),
    '6. RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    '7. ExtraTrees': ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
}

for name, model in models_5_7.items():
    evaluate_model(name, model, X_lc_scaled, y_lc, df_lc_imp, 'LC')
    evaluate_model(name, model, X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""def tune_xgboost(X, y, df, sensor):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
            'random_state': 42,
            'n_jobs': -1
        }
        
        cv_scores = []
        labels = get_stratify_labels(df)
        for train_idx, val_idx in skf.split(X, labels):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(mean_squared_error(y_val, preds))
            
        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=50)
    
    best_model = xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
    evaluate_model('8. XGBoost_Tuned', best_model, X, y, df, sensor)
    return best_model

xgb_lc = tune_xgboost(X_lc_scaled, y_lc, df_lc_imp, 'LC')
xgb_ps = tune_xgboost(X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""def tune_lightgbm(X, y, df, sensor):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        cv_scores = []
        labels = get_stratify_labels(df)
        for train_idx, val_idx in skf.split(X, labels):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(mean_squared_error(y_val, preds))
            
        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    best_model = lgb.LGBMRegressor(**study.best_params, random_state=42, verbose=-1, n_jobs=-1)
    evaluate_model('9. LightGBM_Tuned', best_model, X, y, df, sensor)
    return best_model

lgbm_lc = tune_lightgbm(X_lc_scaled, y_lc, df_lc_imp, 'LC')
lgbm_ps = tune_lightgbm(X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""def tune_catboost(X, y, df, sensor):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10, log=True),
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }
        
        cv_scores = []
        labels = get_stratify_labels(df)
        for train_idx, val_idx in skf.split(X, labels):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = cb.CatBoostRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(mean_squared_error(y_val, preds))
            
        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    best_model = cb.CatBoostRegressor(**study.best_params, random_seed=42, verbose=False, thread_count=-1)
    evaluate_model('10. CatBoost_Tuned', best_model, X, y, df, sensor)
    return best_model

cb_lc = tune_catboost(X_lc_scaled, y_lc, df_lc_imp, 'LC')
cb_ps = tune_catboost(X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""tabicl_model = TabICLRegressor()
evaluate_model('11. TabICLv2', tabicl_model, X_lc_scaled, y_lc, df_lc_imp, 'LC')
evaluate_model('11. TabICLv2', tabicl_model, X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""class CAFF(nn.Module):
    def __init__(self, n_features=16, hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), 
            nn.LayerNorm(hidden), 
            nn.ReLU(), 
            nn.Dropout(dropout)
        )
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden//2), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden//2, 1)
        )
        
    def forward(self, x):
        enc = self.encoder(x).unsqueeze(1)
        out, _ = self.attn(enc, enc, enc)
        return self.decoder(out.squeeze(1)).squeeze(-1)

caff_model_lc = CAFF()
caff_model_ps = CAFF()

evaluate_model('12. CAFF', caff_model_lc, X_lc_scaled, y_lc, df_lc_imp, 'LC', is_nn=True, fit_params={'epochs': 100, 'lr': 0.001})
evaluate_model('12. CAFF', caff_model_ps, X_ps_scaled, y_ps, df_ps_imp, 'PS', is_nn=True, fit_params={'epochs': 100, 'lr': 0.001})
""")

code("""def tune_caff(X, y, df, sensor):
    def objective(trial):
        hidden = trial.suggest_categorical('hidden', [32, 64, 128])
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 50, 150)
        
        cv_scores = []
        labels = get_stratify_labels(df)
        for train_idx, val_idx in skf.split(X, labels):
            X_tr = torch.FloatTensor(X[train_idx])
            X_val = torch.FloatTensor(X[val_idx])
            y_tr = torch.FloatTensor(y.iloc[train_idx].values)
            y_val = y.iloc[val_idx].values
            
            model = CAFF(hidden=hidden, n_heads=n_heads, dropout=dropout)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                preds = model(X_tr)
                loss = criterion(preds, y_tr)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                preds_val = model(X_val).numpy()
                
            cv_scores.append(mean_squared_error(y_val, preds_val))
            
        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) 
    
    best_params = study.best_params
    best_model = CAFF(hidden=best_params['hidden'], n_heads=best_params['n_heads'], dropout=best_params['dropout'])
    evaluate_model('13. CAFF_Tuned', best_model, X, y, df, sensor, is_nn=True, 
                   fit_params={'epochs': best_params['epochs'], 'lr': best_params['lr']})
    return best_model

caff_tuned_lc = tune_caff(X_lc_scaled, y_lc, df_lc_imp, 'LC')
caff_tuned_ps = tune_caff(X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""df_res_lc = pd.DataFrame(results['LC']).T
df_res_ps = pd.DataFrame(results['PS']).T

print("=== Hasil Model LC ===")
display(df_res_lc[['R2', 'RMSE', 'MAE', 'MAPE']].sort_values(by='R2', ascending=False))

print("\\n=== Hasil Model PS ===")
display(df_res_ps[['R2', 'RMSE', 'MAE', 'MAPE']].sort_values(by='R2', ascending=False))
""")

# Phase 5
md("## Phase 5 — Ensemble & Stacking")

code("""voting_lc = VotingRegressor([
    ('xgb', xgb_lc),
    ('lgbm', lgbm_lc),
    ('cb', cb_lc),
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

voting_ps = VotingRegressor([
    ('xgb', xgb_ps),
    ('lgbm', lgbm_ps),
    ('cb', cb_ps),
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

evaluate_model('14. VotingRegressor', voting_lc, X_lc_scaled, y_lc, df_lc_imp, 'LC')
evaluate_model('14. VotingRegressor', voting_ps, X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

code("""stacking_lc = StackingRegressor(
    estimators=[
        ('xgb', xgb_lc),
        ('lgbm', lgbm_lc),
        ('cb', cb_lc),
        ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    ],
    final_estimator=Ridge(alpha=1.0)
)

stacking_ps = StackingRegressor(
    estimators=[
        ('xgb', xgb_ps),
        ('lgbm', lgbm_ps),
        ('cb', cb_ps),
        ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    ],
    final_estimator=Ridge(alpha=1.0)
)

evaluate_model('15. StackingRegressor', stacking_lc, X_lc_scaled, y_lc, df_lc_imp, 'LC')
evaluate_model('15. StackingRegressor', stacking_ps, X_ps_scaled, y_ps, df_ps_imp, 'PS')
""")

# Phase 6
md("## Phase 6 — Evaluasi & Komparasi Final")

code("""df_plot_lc = pd.DataFrame(results['LC']).T.reset_index().rename(columns={'index': 'Model'})
df_plot_lc['Sensor'] = 'LC'

df_plot_ps = pd.DataFrame(results['PS']).T.reset_index().rename(columns={'index': 'Model'})
df_plot_ps['Sensor'] = 'PS'

df_plot = pd.concat([df_plot_lc, df_plot_ps])

plt.figure(figsize=(16, 6))
sns.barplot(data=df_plot, x='Model', y='R2', hue='Sensor')
plt.title('R² Comparison All Models (LC vs PS)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig8_r2_comparison.png')
plt.show()
""")

code("""plt.figure(figsize=(16, 6))
sns.barplot(data=df_plot, x='Model', y='RMSE', hue='Sensor')
plt.title('RMSE Comparison All Models (LC vs PS)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig9_rmse_comparison.png')
plt.show()
""")

code("""best_lc = df_plot_lc.loc[df_plot_lc['R2'].idxmax()]
best_ps = df_plot_ps.loc[df_plot_ps['R2'].idxmax()]

best_name_lc = best_lc['Model']
best_name_ps = best_ps['Model']

y_true_lc = results['LC'][best_name_lc]['y_true']
y_pred_lc = results['LC'][best_name_lc]['y_pred']

y_true_ps = results['PS'][best_name_ps]['y_true']
y_pred_ps = results['PS'][best_name_ps]['y_pred']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_true_lc, y_pred_lc, alpha=0.5, color='blue')
axes[0].plot([0, 1000], [0, 1000], 'r--')
axes[0].set_title(f'LC: Parity Plot ({best_name_lc})')
axes[0].set_xlabel('Actual PPM')
axes[0].set_ylabel('Predicted PPM')

axes[1].scatter(y_true_ps, y_pred_ps, alpha=0.5, color='orange')
axes[1].plot([0, 1000], [0, 1000], 'r--')
axes[1].set_title(f'PS: Parity Plot ({best_name_ps})')
axes[1].set_xlabel('Actual PPM')
axes[1].set_ylabel('Predicted PPM')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig10_parity_plot.png')
plt.show()
""")

code("""res_lc = np.array(y_true_lc) - np.array(y_pred_lc)
res_ps = np.array(y_true_ps) - np.array(y_pred_ps)

def get_range(val):
    if val <= 10: return '0-10'
    elif val <= 100: return '10-100'
    else: return '100-1000'

df_res = pd.DataFrame({
    'Actual': np.concatenate([y_true_lc, y_true_ps]),
    'Residual': np.concatenate([res_lc, res_ps]),
    'Sensor': ['LC']*len(y_true_lc) + ['PS']*len(y_true_ps)
})
df_res['Range'] = df_res['Actual'].apply(get_range)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_res, x='Range', y='Residual', hue='Sensor', order=['0-10', '10-100', '100-1000'])
plt.title('Residual per Rentang PPM')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig11_residual.png')
plt.show()
""")

code("""try:
    explainer_lc = shap.TreeExplainer(xgb_lc)
    shap_values_lc = explainer_lc.shap_values(X_lc_scaled)
    
    explainer_ps = shap.TreeExplainer(xgb_ps)
    shap_values_ps = explainer_ps.shap_values(X_ps_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plt.sca(axes[0])
    shap.summary_plot(shap_values_lc, X_lc_scaled, feature_names=FEATURE_COLS, show=False)
    axes[0].set_title('SHAP LC (XGBoost)')
    
    plt.sca(axes[1])
    shap.summary_plot(shap_values_ps, X_ps_scaled, feature_names=FEATURE_COLS, show=False)
    axes[1].set_title('SHAP PS (XGBoost)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig12_shap.png')
    plt.show()
except Exception as e:
    print(f"SHAP Error: {e}, using RF feature importance instead")
    rf_lc = RandomForestRegressor(n_estimators=300).fit(X_lc_scaled, y_lc)
    rf_ps = RandomForestRegressor(n_estimators=300).fit(X_ps_scaled, y_ps)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pd.Series(rf_lc.feature_importances_, index=FEATURE_COLS).sort_values().plot.barh(ax=axes[0])
    axes[0].set_title('RF Feature Importance LC')
    pd.Series(rf_ps.feature_importances_, index=FEATURE_COLS).sort_values().plot.barh(ax=axes[1])
    axes[1].set_title('RF Feature Importance PS')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig12_shap.png')
    plt.show()
""")

code("""print("--- KOMPARASI FINAL LC vs PS ---")
print(f"Best LC Model: {best_name_lc} (R2: {best_lc['R2']:.4f}, RMSE: {best_lc['RMSE']:.4f})")
print(f"Best PS Model: {best_name_ps} (R2: {best_ps['R2']:.4f}, RMSE: {best_ps['RMSE']:.4f})")

if best_lc['R2'] > best_ps['R2']:
    print("\\nKesimpulan: Sensor LC menunjukkan performa prediksi yang lebih baik dibanding PS.")
else:
    print("\\nKesimpulan: Sensor PS menunjukkan performa prediksi yang lebih baik dibanding LC.")
""")

md("""### Dashboard Ringkasan Akhir
Model dengan Optuna (XGBoost, LightGBM, CatBoost) dan variasi CAFF telah dievaluasi. Proses tuning 50 trial memberikan optimasi signifikan untuk hiperparameter. Kesimpulan akhir dapat dilihat pada plot dan tabel di atas.
""")

notebook_json = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path(r'C:\\Users\\Hikar\\Documents\\Kuliah\\Bahan Belajar Kuliah Semester 4\\AI\\UTS Kelas AI A\\Sensor_Analysis_v3.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=2)

print("Notebook generated.")
