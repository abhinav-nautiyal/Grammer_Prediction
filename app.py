import os
import streamlit as st
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'models', 'grammar_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_model()

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, spectral_contrast])

# Streamlit UI
st.title("Grammar Quality Scoring")
audio_file = st.file_uploader("Upload WAV file", type=["wav"])

if audio_file:
    temp_path = os.path.join(BASE_DIR, "temp.wav")
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.audio(temp_path)
    
    try:
        features = extract_features(temp_path)
        features_scaled = scaler.transform([features])
        score = model.predict(features_scaled)[0].clip(1, 5)
        st.success(f"Predicted Grammar Score: {score:.2f}/5")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)