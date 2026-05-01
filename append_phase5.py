import nbformat as nbf

with open('Sensor_Analysis_v4.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

phase5_cells = []

# Markdown Phase 5
phase5_cells.append(nbf.v4.new_markdown_cell("""## Phase 5 — Ensemble & Stacking
Menggabungkan kekuatan model-model terbaik (XGBoost, LightGBM, CatBoost, ExtraTrees, dan CAFF)
menggunakan `VotingRegressor` dan `StackingRegressor`."""))

# Code Phase 5
phase5_cells.append(nbf.v4.new_code_cell("""from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# 1. Wrapper CAFF untuk kompatibilitas dengan Scikit-Learn
class SklearnCAFF(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=64, n_heads=4, dropout=0.1, lr=0.001, batch_size=32, epochs=100):
        self.hidden = hidden
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def fit(self, X, y):
        train_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = CAFF(n_features=X.shape[1], 
                          hidden=self.hidden, 
                          n_heads=self.n_heads, 
                          dropout=self.dropout)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model belum di-fit!")
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.FloatTensor(X)).numpy()
        return preds

# Ambil hyperparameter terbaik dari output tuning Phase 4
best_caff_lc = {'hidden': 64, 'n_heads': 4, 'lr': 0.005297624608166901, 'dropout': 0.0155202402336867, 'batch_size': 32, 'epochs': 100}
best_caff_ps = {'hidden': 128, 'n_heads': 2, 'lr': 0.0022842692416162056, 'dropout': 0.20109255860421849, 'batch_size': 16, 'epochs': 100}

# Instansiasi CAFF Optuna
caff_optuna_lc = SklearnCAFF(**best_caff_lc)
caff_optuna_ps = SklearnCAFF(**best_caff_ps)
"""))

phase5_cells.append(nbf.v4.new_code_cell("""# 2. Voting Regressor
print("Mengevaluasi 14. VotingRegressor...")
voting_lc = VotingRegressor([
    ('xgb', xgb_lc),
    ('lgbm', lgb_lc),
    ('cb', cb_lc),
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

voting_ps = VotingRegressor([
    ('xgb', xgb_ps),
    ('lgbm', lgb_ps),
    ('cb', cb_ps),
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

evaluate_sklearn_model('14. VotingRegressor', voting_lc, X_lc_scaled, y_lc, labels_lc, 'LC')
evaluate_sklearn_model('14. VotingRegressor', voting_ps, X_ps_scaled, y_ps, labels_ps, 'PS')
"""))

phase5_cells.append(nbf.v4.new_code_cell("""# 3. Stacking Regressor
print("\\nMengevaluasi 15. StackingRegressor...")

# Catatan: n_jobs di StackingRegressor diset None (1 core) agar terhindar dari PicklingError PyTorch di Windows.
stacking_lc = StackingRegressor(
    estimators=[
        ('xgb', xgb_lc),
        ('lgbm', lgb_lc),
        ('cb', cb_lc),
        ('caff', caff_optuna_lc)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=None 
)

stacking_ps = StackingRegressor(
    estimators=[
        ('xgb', xgb_ps),
        ('lgbm', lgb_ps),
        ('cb', cb_ps),
        ('caff', caff_optuna_ps)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=None
)

evaluate_sklearn_model('15. StackingRegressor', stacking_lc, X_lc_scaled, y_lc, labels_lc, 'LC')
evaluate_sklearn_model('15. StackingRegressor', stacking_ps, X_ps_scaled, y_ps, labels_ps, 'PS')

print("\\nPhase 5 Selesai!")
"""))

has_phase5 = any("Phase 5 — Ensemble" in cell.source for cell in nb.cells if cell.cell_type == "markdown")
if not has_phase5:
    nb.cells.extend(phase5_cells)

with open('Sensor_Analysis_v4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Phase 5 appended successfully!")
