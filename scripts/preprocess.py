# scripts/preprocess.py
import os
import pandas as pd
from tqdm import tqdm
from scripts.utils import ensure_wav, extract_audio_features_file
import numpy as np

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
TRAIN_AUDIO_DIR = os.path.join(DATASET_DIR, "audios_train")
TEST_AUDIO_DIR = os.path.join(DATASET_DIR, "audios_test")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

OUT_TRAIN_FEATURES = os.path.join(DATASET_DIR, "train_features.csv")
OUT_TEST_FEATURES = os.path.join(DATASET_DIR, "test_features.csv")

def process_training_data():
    df = pd.read_csv(TRAIN_CSV)
    X = []
    y = []
    ids = []
    missing = []
    print("Extracting features for training set...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = str(row['filename'])
        label = row['label']   # your train.csv uses 'label' column (1-5)
        wav_path = os.path.join(TRAIN_AUDIO_DIR, fname)
        if not os.path.exists(wav_path):
            # try mp3
            alt_mp3 = wav_path.replace(".wav", ".mp3")
            if os.path.exists(alt_mp3):
                wav_path = ensure_wav(alt_mp3)
            else:
                missing.append(fname)
                continue
        feat = extract_audio_features_file(wav_path)
        X.append(feat)
        y.append(label)
        ids.append(fname)
    X = np.array(X)
    df_feat = pd.DataFrame(X)
    df_feat['label'] = y
    df_feat['filename'] = ids
    df_feat.to_csv(OUT_TRAIN_FEATURES, index=False)
    print("Saved", OUT_TRAIN_FEATURES)
    if missing:
        print("Missing files:", missing[:10])

def process_test_data():
    df = pd.read_csv(TEST_CSV)
    X = []
    ids = []
    missing = []
    print("Extracting features for test set...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = str(row['filename'])
        wav_path = os.path.join(TEST_AUDIO_DIR, fname)
        if not os.path.exists(wav_path):
            alt_mp3 = wav_path.replace(".wav", ".mp3")
            if os.path.exists(alt_mp3):
                wav_path = ensure_wav(alt_mp3)
            else:
                missing.append(fname)
                continue
        feat = extract_audio_features_file(wav_path)
        X.append(feat)
        ids.append(fname)
    X = np.array(X)
    df_feat = pd.DataFrame(X)
    df_feat['filename'] = ids
    df_feat.to_csv(OUT_TEST_FEATURES, index=False)
    print("Saved", OUT_TEST_FEATURES)
    if missing:
        print("Missing files:", missing[:10])

if __name__ == "__main__":
    process_training_data()
    process_test_data()
