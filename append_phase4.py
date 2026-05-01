import nbformat as nbf
from pathlib import Path

with open('Sensor_Analysis_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

phase4_cells = []

# Markdown Phase 4
phase4_cells.append(nbf.v4.new_markdown_cell("""## Phase 4 — Model Training & Hyperparameter Tuning
Target di-transformasi menjadi logaritmik: `y = log10(ppm + 1)`.
Evaluasi menggunakan Stratified 5-Fold CV berdasarkan kategori ppm."""))

# Imports
phase4_cells.append(nbf.v4.new_code_cell("""# Import library ML
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

try:
    from tabicl import TabICLRegressor
    HAS_TABICL = True
except ImportError:
    print("Warning: tabicl tidak terinstall. Jalankan: pip install tabicl")
    HAS_TABICL = False

import warnings
warnings.filterwarnings('ignore', category=UserWarning)"""))

# Data Preparation
phase4_cells.append(nbf.v4.new_code_cell("""# Pastikan tidak ada NaN. Jika ada, fill dengan 0
feature_cols = [c for c in df_lc_imputed.columns if c.startswith('upper_') or c.startswith('lower_')]

X_lc = df_lc_imputed[feature_cols].fillna(0).values
y_lc = np.log10(df_lc_imputed['ppm'].values + 1)

X_ps = df_ps_imputed[feature_cols].fillna(0).values
y_ps = np.log10(df_ps_imputed['ppm'].values + 1)

# Skala fitur
scaler_lc = StandardScaler()
X_lc_scaled = scaler_lc.fit_transform(X_lc)

scaler_ps = StandardScaler()
X_ps_scaled = scaler_ps.fit_transform(X_ps)

# Stratifikasi berdasarkan kategori ppm
def get_stratify_labels(ppm_array):
    return ppm_array.astype(str)

labels_lc = get_stratify_labels(df_lc_imputed['ppm'].values)
labels_ps = get_stratify_labels(df_ps_imputed['ppm'].values)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary untuk menyimpan hasil
results = {'LC': {}, 'PS': {}}
best_models = {'LC': {}, 'PS': {}}"""))

# Evaluation Framework
phase4_cells.append(nbf.v4.new_code_cell("""def evaluate_sklearn_model(name, model, X, y, labels, sensor):
    r2_list, rmse_list, mae_list, mape_list = [], [], [], []
    
    for train_idx, val_idx in skf.split(X, labels):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)
        preds_log = m.predict(X_val)
        
        # Inverse transform
        y_val_inv = (10 ** y_val) - 1
        preds_inv = np.maximum(0, (10 ** preds_log) - 1)
        
        r2_list.append(r2_score(y_val_inv, preds_inv))
        rmse_list.append(np.sqrt(mean_squared_error(y_val_inv, preds_inv)))
        mae_list.append(mean_absolute_error(y_val_inv, preds_inv))
        
        mask = y_val_inv > 0
        if np.sum(mask) > 0:
            mape_list.append(mean_absolute_percentage_error(y_val_inv[mask], preds_inv[mask]))
            
    res = {
        'R2': np.mean(r2_list), 'RMSE': np.mean(rmse_list),
        'MAE': np.mean(mae_list), 'MAPE': np.mean(mape_list) if mape_list else np.nan
    }
    results[sensor][name] = res
    print(f"{sensor} - {name:20s} | R2: {res['R2']:.4f} | RMSE: {res['RMSE']:.2f}")
    best_models[sensor][name] = model # Simpan referensi model (belum di-fit di seluruh data)"""))

# Model 1-7
phase4_cells.append(nbf.v4.new_code_cell("""# --- 1-4: Baseline Linear ---
linear_models = {
    '1. LinearRegression': LinearRegression(),
    '2. Ridge': Ridge(alpha=1.0),
    '3. ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    '4. PolyRidge': make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0))
}

# --- 5: Kernel / SVM ---
svm_models = {
    '5. SVR': SVR(kernel='rbf', C=500, gamma='scale')
}

# --- 6-7: Tree Ensemble ---
tree_models = {
    '6. RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    '7. ExtraTrees': ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
}

all_baselines = {**linear_models, **svm_models, **tree_models}

print("Mengevaluasi Model Baseline (1-7)...")
for name, model in all_baselines.items():
    evaluate_sklearn_model(name, model, X_lc_scaled, y_lc, labels_lc, 'LC')
    evaluate_sklearn_model(name, model, X_ps_scaled, y_ps, labels_ps, 'PS')
print("Selesai mengevaluasi baseline.")"""))

# Model 8: XGBoost Optuna
phase4_cells.append(nbf.v4.new_code_cell("""def tune_xgboost(X, y, labels, sensor):
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
        for train_idx, val_idx in skf.split(X, labels):
            model = xgb.XGBRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            # Optuna minimize MSE in log-space
            cv_scores.append(mean_squared_error(y[val_idx], preds))
        return np.mean(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    best_model = xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
    evaluate_sklearn_model('8. XGBoost_Tuned', best_model, X, y, labels, sensor)
    return best_model

print("Tuning XGBoost...")
xgb_lc = tune_xgboost(X_lc_scaled, y_lc, labels_lc, 'LC')
xgb_ps = tune_xgboost(X_ps_scaled, y_ps, labels_ps, 'PS')"""))

# Model 9: LightGBM Optuna
phase4_cells.append(nbf.v4.new_code_cell("""def tune_lightgbm(X, y, labels, sensor):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        cv_scores = []
        for train_idx, val_idx in skf.split(X, labels):
            model = lgb.LGBMRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            cv_scores.append(mean_squared_error(y[val_idx], preds))
        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    best_model = lgb.LGBMRegressor(**study.best_params, random_state=42, verbose=-1, n_jobs=-1)
    evaluate_sklearn_model('9. LightGBM_Tuned', best_model, X, y, labels, sensor)
    return best_model

print("Tuning LightGBM...")
lgb_lc = tune_lightgbm(X_lc_scaled, y_lc, labels_lc, 'LC')
lgb_ps = tune_lightgbm(X_ps_scaled, y_ps, labels_ps, 'PS')"""))

# Model 10: CatBoost Optuna
phase4_cells.append(nbf.v4.new_code_cell("""def tune_catboost(X, y, labels, sensor):
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
        for train_idx, val_idx in skf.split(X, labels):
            model = cb.CatBoostRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            cv_scores.append(mean_squared_error(y[val_idx], preds))
        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    best_model = cb.CatBoostRegressor(**study.best_params, random_seed=42, verbose=False, thread_count=-1)
    evaluate_sklearn_model('10. CatBoost_Tuned', best_model, X, y, labels, sensor)
    return best_model

print("Tuning CatBoost...")
cb_lc = tune_catboost(X_lc_scaled, y_lc, labels_lc, 'LC')
cb_ps = tune_catboost(X_ps_scaled, y_ps, labels_ps, 'PS')"""))

# Model 11: TabICL
phase4_cells.append(nbf.v4.new_code_cell("""if HAS_TABICL:
    print("Mengevaluasi TabICLv2 (Zero-shot)...")
    tabicl_model = TabICLRegressor()
    evaluate_sklearn_model('11. TabICLv2', tabicl_model, X_lc_scaled, y_lc, labels_lc, 'LC')
    evaluate_sklearn_model('11. TabICLv2', tabicl_model, X_ps_scaled, y_ps, labels_ps, 'PS')
else:
    print("TabICLv2 di-skip karena belum terinstall.")"""))

# Model 12: CAFF & Model 13: CAFF-Optuna
phase4_cells.append(nbf.v4.new_code_cell("""class CAFF(nn.Module):
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
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, x):
        enc = self.encoder(x).unsqueeze(1)
        attn_out, _ = self.attn(enc, enc, enc)
        out = self.decoder(attn_out.squeeze(1))
        return out.squeeze(-1)

def train_eval_caff(X_train, y_train, X_val, y_val, params):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size', 32), shuffle=True)
    
    model = CAFF(n_features=X_train.shape[1], 
                 hidden=params.get('hidden', 64), 
                 n_heads=params.get('n_heads', 4), 
                 dropout=params.get('dropout', 0.1))
                 
    optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(params.get('epochs', 200)):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds_val = model(torch.FloatTensor(X_val)).numpy()
        
    return preds_val

def evaluate_caff(name, X, y, labels, sensor, params):
    r2_list, rmse_list, mae_list, mape_list = [], [], [], []
    for train_idx, val_idx in skf.split(X, labels):
        preds_log = train_eval_caff(X[train_idx], y[train_idx], X[val_idx], y[val_idx], params)
        
        y_val_inv = (10 ** y[val_idx]) - 1
        preds_inv = np.maximum(0, (10 ** preds_log) - 1)
        
        r2_list.append(r2_score(y_val_inv, preds_inv))
        rmse_list.append(np.sqrt(mean_squared_error(y_val_inv, preds_inv)))
        mae_list.append(mean_absolute_error(y_val_inv, preds_inv))
        mask = y_val_inv > 0
        if np.sum(mask) > 0:
            mape_list.append(mean_absolute_percentage_error(y_val_inv[mask], preds_inv[mask]))
            
    res = {
        'R2': np.mean(r2_list), 'RMSE': np.mean(rmse_list),
        'MAE': np.mean(mae_list), 'MAPE': np.mean(mape_list) if mape_list else np.nan
    }
    results[sensor][name] = res
    print(f"{sensor} - {name:20s} | R2: {res['R2']:.4f} | RMSE: {res['RMSE']:.2f}")

print("Mengevaluasi 12. CAFF (Fixed Params)...")
caff_params = {'hidden': 64, 'n_heads': 4, 'dropout': 0.1, 'lr': 0.001, 'batch_size': 32, 'epochs': 200}
evaluate_caff('12. CAFF', X_lc_scaled, y_lc, labels_lc, 'LC', caff_params)
evaluate_caff('12. CAFF', X_ps_scaled, y_ps, labels_ps, 'PS', caff_params)"""))

# CAFF-Optuna
phase4_cells.append(nbf.v4.new_code_cell("""def tune_caff(X, y, labels, sensor):
    def objective(trial):
        params = {
            'hidden': trial.suggest_categorical('hidden', [32, 64, 128]),
            'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'epochs': 100 # kurangi epochs sedikit agar tidak terlalu lama saat tuning
        }
        
        cv_scores = []
        for train_idx, val_idx in skf.split(X, labels):
            preds = train_eval_caff(X[train_idx], y[train_idx], X[val_idx], y[val_idx], params)
            cv_scores.append(mean_squared_error(y[val_idx], preds))
        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # Dibatasi 10 trials dulu untuk menghemat waktu (bisa diubah menjadi 50 jika ada waktu)
    
    best_params = study.best_params
    best_params['epochs'] = 200 # gunakan 200 epochs untuk evaluasi akhir
    
    print(f"Best Params {sensor}: {best_params}")
    evaluate_caff('13. CAFF_Tuned', X, y, labels, sensor, best_params)

print("Tuning CAFF (Model 13)...")
tune_caff(X_lc_scaled, y_lc, labels_lc, 'LC')
tune_caff(X_ps_scaled, y_ps, labels_ps, 'PS')"""))

# Summary Results Phase 4
phase4_cells.append(nbf.v4.new_code_cell("""import pandas as pd
df_res_lc = pd.DataFrame(results['LC']).T
df_res_ps = pd.DataFrame(results['PS']).T

print("=== Peringkat Model Regresi (Sensor LC) ===")
display(df_res_lc.sort_values(by='R2', ascending=False))

print("\\n=== Peringkat Model Regresi (Sensor PS) ===")
display(df_res_ps.sort_values(by='R2', ascending=False))"""))

has_phase4 = any("Phase 4 — Model Training" in cell.source for cell in nb.cells if cell.cell_type == "markdown")
if not has_phase4:
    nb.cells.extend(phase4_cells)

with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Phase 4 cells appended to notebook.")
