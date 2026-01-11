# scripts/train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
TRAIN_FEATURES = os.path.join(DATASET_DIR, "train_features.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_FEATURES)
# assume label exists; drop label and an id-like column if present
labels = df['label'].values
cols_to_drop = ['label']
if 'filename' in df.columns:
    cols_to_drop.append('filename')
if 'id' in df.columns:
    cols_to_drop.append('id')
X = df.drop(columns=cols_to_drop).values.astype(float)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# RandomForest
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
rf_val_pred = rf.predict(X_val_s)
rf_mae = mean_absolute_error(y_val, rf_val_pred)
print("RandomForest Val MAE:", rf_mae)

# SVR
svm = SVR()
svm.fit(X_train_s, y_train)
svm_val_pred = svm.predict(X_val_s)
svm_mae = mean_absolute_error(y_val, svm_val_pred)
print("SVM Val MAE:", svm_mae)

# Optionally ensemble: average predictions
ensemble_val = 0.5 * rf_val_pred + 0.5 * svm_val_pred
ensemble_mae = mean_absolute_error(y_val, ensemble_val)
print("Ensemble Val MAE:", ensemble_mae)

# Lightweight calibration: fit y_true ≈ a * y_pred + b on validation
try:
    a, b = np.polyfit(ensemble_val, y_val, deg=1)
    calib = {"a": float(a), "b": float(b)}
except Exception:
    calib = {"a": 1.0, "b": 0.0}

# Save models & scaler
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(svm, os.path.join(MODELS_DIR, "svm_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(calib, os.path.join(MODELS_DIR, "calibration.pkl"))

# Save training metrics
metrics = {
    "rf_val_mae": float(rf_mae),
    "svm_val_mae": float(svm_mae),
    "ensemble_val_mae": float(ensemble_mae)
}
pd.Series(metrics).to_csv(os.path.join(MODELS_DIR, "train_metrics.csv"))
print("Models and scaler saved to", MODELS_DIR)

