# scripts/predict.py
import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
TEST_FEATURES = os.path.join(DATASET_DIR, "test_features.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_SUB = os.path.join(DATASET_DIR, "submission.csv")

df = pd.read_csv(TEST_FEATURES)
filenames = df['filename'].values
X = df.drop(columns=['filename']).values.astype(float)

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
svm = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))

X_s = scaler.transform(X)
rf_pred = rf.predict(X_s)
svm_pred = svm.predict(X_s)
ensemble = 0.5 * rf_pred + 0.5 * svm_pred

# Clip ensemble to expected range if needed (for SHL problem it's 1-5)
ensemble = np.clip(ensemble, 1.0, 5.0)

out_df = pd.DataFrame({
    "id": [os.path.splitext(x)[0] for x in filenames],
    "grammar_accuracy": ensemble
})
out_df.to_csv(OUT_SUB, index=False)
print("Saved submission:", OUT_SUB)
