# Grammar Quality Scoring System
The Grammar Quality Scoring System is a machine learning application that analyzes spoken audio content and predicts grammar quality on a scale from 1 to 5. This project combines audio processing, feature extraction, and machine learning to automatically assess the grammatical accuracy of spoken language.

## Table of Content
- [Project Structure](#Project_Structure)
- [Prerequisites & Installation](#Prerequisites_and_Installation)
- [Usage](#Usage)
- [Technical Details](#Technical_Details)
- [Troubleshooting](#Troubleshooting)
- [Results](#Results)

## Project_Structure
```Bash
Grammar-Quality-Scoring/
├── dataset/
│   ├── audios_train/         # Training audio files (.wav)
│   ├── audios_test/          # Test audio files (.wav)
│   ├── train.csv             # Training labels
│   ├── test.csv              # Test file names
│   ├── train_features.csv    # Generated features (preprocess)
│   └── test_features.csv     # Generated features (preprocess)
├── models/
│   ├── scaler.pkl
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── calibration.pkl       # Optional linear calibration
│   └── train_metrics.csv
├── scripts/
│   ├── preprocess.py         # Feature extraction
│   ├── train.py              # Model training (RF + SVR)
│   ├── predict.py            # Prediction generation (RF+SVR ensemble)
│   └── utils.py              # Shared helpers (audio, ASR, LLM providers)
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
└── .env                      # API keys (GOOGLE_API_KEY, GROQ_API_KEY)
```

## Prerequisites_and_Installation
### Prerequisites
- Python 3.11 (recommended)
- pip package manager
- ffmpeg (required for MP3 → WAV conversion via pydub)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/grammar-quality-scoring.git
cd grammar-quality-scoring
```
2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use "venv\Scripts\activate"
```
3. Install dependencies (versions pinned for model compatibility):
```bash
pip install -r requirements.txt
# key pins: scikit-learn==1.2.2, numpy==1.24.4, scipy==1.10.1
```
4. (Optional) Create `.env` to enable grammar suggestions (LLM providers):
```bash
echo 'GOOGLE_API_KEY="<your_gemini_key>"' >> .env
echo 'GROQ_API_KEY="gsk_...<your_groq_key>"' >> .env
```

## Usage
### Data Processing Pipeline
Run scripts as modules from the project root so relative paths resolve correctly:
1. Extract audio features:
```bash
python -m scripts.preprocess
```
2. Train the models (RF + SVR) and save artifacts:
```bash
python -m scripts.train
```
3. Generate predictions (RF+SVR ensemble):
```bash
python -m scripts.predict
```

### Web Interface
Run the Streamlit app:
```bash
streamlit run app.py
```
Open the URL Streamlit prints (e.g., http://localhost:8501).

- Use "Upload file" and select a WAV/MP3 to test.
- In the sidebar, choose the Grammar model provider:
  - Auto (Groq → Gemini), Groq, or Gemini
- If MP3 uploads fail, install ffmpeg or use WAV.

## Technical_Details
### Feature Extraction
The system currently computes the following statistics using Librosa:
- MFCC mean (13)
- Chroma mean (12)
- Spectral contrast mean (7)

### Machine Learning Model
- Training: RandomForestRegressor and SVR, with StandardScaler
- Inference: Ensemble average of RF and SVR predictions
- Metric: Mean Absolute Error (MAE)
- Score Range: 1 (poor) to 5 (excellent grammar)

### Grammar Analysis (LLM)
- Providers supported: Groq (Llama/Mixtral) and Gemini
- Select provider via the "Grammar model provider" control in the app
- Requires corresponding API keys in `.env` (see Installation)

## Troubleshooting
- **Pickle/NumPy/Sklearn mismatch**
  - Ensure pinned versions from `requirements.txt` are installed
  - Reinstall: `pip install --upgrade --force-reinstall -r requirements.txt`
- **MP3 upload conversion fails**
  - Install ffmpeg, or upload WAV files instead
- **LLM returns no mistakes**
  - Try switching provider (Groq/Gemini) in the UI
  - Expand "Model response (debug)" to see raw output
- **Gemini quota/invalid key**
  - Verify project enablement/quota; switch to Groq as needed
- **Microphone recording on macOS**
  - Grant microphone permission to Terminal/IDE and your browser

## Results
The model provides:
- Grammar quality predictions with visual feedback
- Audio playback capability
- Grammar suggestions (LLM) with highlights and explanations
