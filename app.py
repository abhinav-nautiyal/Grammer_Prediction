# app.py
import streamlit as st
import tempfile
import os
from scripts.utils import ensure_wav, extract_audio_features_file, audiofile_to_text, call_gemini_grammar_correct, highlight_text_html
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Grammar Quality + Suggestions", layout="centered")
st.title("Grammar Quality Scoring & Suggestions")
model_choice = st.selectbox(
    "Grammar model provider",
    ["Auto (Groq → Gemini)", "Groq", "Gemini"],
    index=0,
    help="Choose which LLM to use for grammar analysis"
)
provider = "auto" if model_choice.startswith("Auto") else ("groq" if model_choice == "Groq" else "gemini")


MODELS_DIR = "models"
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
svm = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
calib_path = os.path.join(MODELS_DIR, "calibration.pkl")
calib = joblib.load(calib_path) if os.path.exists(calib_path) else {"a": 1.0, "b": 0.0}

option = st.radio("Input:", ("Record (microphone)", "Upload file"))

def record_to_file(duration=5, sr=16000):
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wavfile
    except Exception as e:
        st.error("Recording dependencies not installed. Please install 'sounddevice' and 'scipy' or use 'Upload file'.")
        raise
    st.info(f"Recording {duration}s...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    # Ensure PCM 16-bit WAV for SpeechRecognition compatibility
    audio_int16 = (audio.squeeze() * 32767.0).astype(np.int16)
    wavfile.write(tmp.name, sr, audio_int16)
    return tmp.name

if option == "Record (microphone)":
    duration = st.slider("Duration (seconds)", 3, 12, 5)
    if st.button("Start Recording"):
        path = record_to_file(duration=duration)
        st.audio(path)
        st.info("Running analysis...")
        # ML score
        feat = extract_audio_features_file(path)
        scaled = scaler.transform(feat.reshape(1, -1))
        rf_score = rf.predict(scaled)[0]
        svm_score = svm.predict(scaled)[0]
        base = (rf_score + svm_score) / 2.0
        cal = calib.get("a", 1.0) * base + calib.get("b", 0.0)
        avg_score = float(np.clip(cal, 1.0, 5.0))
        st.metric("Predicted Grammar Score (1-5)", f"{avg_score:.2f}")
        # ASR -> text
        text = audiofile_to_text(path)
        st.subheader("Recognized text")
        st.write(text if text else "_(could not transcribe)_")
        # Gemini grammar corrections
        if text:
            parsed = call_gemini_grammar_correct(text, provider=provider)
            corrected = parsed.get("corrected_text", "")
            mistakes = parsed.get("mistakes", [])
            st.subheader("Corrected Text")
            st.write(corrected)
            # Highlighted HTML
            highlighted = highlight_text_html(text, mistakes)
            st.markdown(f"**Original with highlights:**  \n\n{highlighted}", unsafe_allow_html=True)
            # Show each mistake nicely
            if mistakes:
                st.subheader("Mistakes & Suggestions")
                for m in mistakes:
                    st.markdown(f"- **Original:** `{m.get('original','')}`  \n  **Suggestion:** `{m.get('suggestion','')}`  \n  **Why:** {m.get('explanation','')}")
            else:
                st.info("No explicit grammar mistakes returned by the model.")
                if 'raw' in parsed and parsed['raw']:
                    with st.expander("Model response (debug)"):
                        st.code(parsed['raw'])
        os.remove(path)

else:
    uploaded = st.file_uploader("Upload audio (.wav or .mp3)", type=["wav", "mp3"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read())
        tmp.close()
        wav_path = ensure_wav(tmp.name)
        st.audio(wav_path)
        st.info("Running analysis...")
        feat = extract_audio_features_file(wav_path)
        scaled = scaler.transform(feat.reshape(1, -1))
        rf_score = rf.predict(scaled)[0]
        svm_score = svm.predict(scaled)[0]
        base = (rf_score + svm_score) / 2.0
        cal = calib.get("a", 1.0) * base + calib.get("b", 0.0)
        avg_score = float(np.clip(cal, 1.0, 5.0))
        st.metric("Predicted Grammar Score (1-5)", f"{avg_score:.2f}")
        text = audiofile_to_text(wav_path)
        st.subheader("Recognized text")
        st.write(text if text else "_(could not transcribe)_")
        if text:
            parsed = call_gemini_grammar_correct(text, provider=provider)
            corrected = parsed.get("corrected_text", "")
            mistakes = parsed.get("mistakes", [])
            st.subheader("Corrected Text")
            st.write(corrected)
            highlighted = highlight_text_html(text, mistakes)
            st.markdown(f"**Original with highlights:**  \n\n{highlighted}", unsafe_allow_html=True)
            if mistakes:
                st.subheader("Mistakes & Suggestions")
                for m in mistakes:
                    st.markdown(f"- **Original:** `{m.get('original','')}`  \n  **Suggestion:** `{m.get('suggestion','')}`  \n  **Why:** {m.get('explanation','')}")
        # cleanup
        os.remove(tmp.name)
        if wav_path != tmp.name and wav_path.endswith(".wav"):
            # if conversion created a wav (ensure_wav), and it's in temp dir, remove it
            try:
                os.remove(wav_path)
            except Exception:
                pass
