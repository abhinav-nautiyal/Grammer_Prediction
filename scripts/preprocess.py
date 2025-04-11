import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_features(file_path, sr=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        return np.hstack([mfcc, chroma, spectral_contrast])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(n_mfcc + 12 + 7)

def process_dataset(audio_dir, csv_path, is_train=True):
    df = pd.read_csv(csv_path)
    
    X = []
    y = [] if is_train else None
    ids = []
    
    print(f"Processing {'training' if is_train else 'test'} data...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(audio_dir, row['filename'])
        features = extract_features(file_path)
        X.append(features)
        if is_train:
            y.append(row['label'])
        ids.append(row['filename'].replace('.wav', ''))
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(len(X[0]))]
    feature_df = pd.DataFrame(X, columns=feature_cols)
    feature_df['id'] = ids
    if is_train:
        feature_df['label'] = y
    
    # Save results
    output_dir = os.path.join(BASE_DIR, 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    output_file = 'train_features.csv' if is_train else 'test_features.csv'
    output_path = os.path.join(output_dir, output_file)
    feature_df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")

if __name__ == "__main__":
    # Set paths
    train_audio_dir = os.path.join(BASE_DIR, 'dataset', 'audios_train')
    train_csv = os.path.join(BASE_DIR, 'dataset', 'train.csv')
    test_audio_dir = os.path.join(BASE_DIR, 'dataset', 'audios_test')
    test_csv = os.path.join(BASE_DIR, 'dataset', 'test.csv')
    
    # Process data
    process_dataset(train_audio_dir, train_csv, is_train=True)
    process_dataset(test_audio_dir, test_csv, is_train=False)