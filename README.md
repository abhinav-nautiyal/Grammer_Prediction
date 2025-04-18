# Grammar Quality Scoring System
The Grammar Quality Scoring System is a machine learning application that analyzes spoken audio content and predicts grammar quality on a scale from 1 to 5. This project combines audio processing, feature extraction, and machine learning to automatically assess the grammatical accuracy of spoken language.

## Table of Content
- [Project Structure](#Project_Structure)
- [Prerequisites & Installation](#Prerequisites_and_Installation)
- [Usage](#Usage)
- [Technical Details](#Technical_Details)
- [Result](#Results)

## Project_Structure
```Bash
Grammar-Quality-Scoring/
├── dataset/
│   ├── audios_train/         # Training audio files (.wav)
│   ├── audios_test/          # Test audio files (.wav)
│   ├── train.csv             # Training labels
│   ├── test.csv              # Test file names
├── scripts/
│   ├── preprocess.py         # Feature extraction
│   ├── train.py              # Model training
│   └── predict.py            # Prediction generation
├── app.py                    # Streamlit web interface
└── requirements.txt          # Python dependencies
```

## Prerequisites_and_Installation
### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/grammar-quality-scoring.git
cd grammar-quality-scoring
```
2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Data Processing Pipeline
1. Extract audio features:
```bash
python scripts/preprocess.py
```
2. Train the model:
```bash
python scripts/train.py
```
3. Generate predictions:
```bash
python scripts/predict.py
```
### Web Interface
Run the Streamlit app:
```bash
streamlit run app.py
```

## Technical_Details
### Feature Extraction
The system extracts the following audio features using Librosa:
- MFCCs (Mel-frequency cepstral coefficients)
- Chroma features
- Spectral contrast
- Zero-crossing rate
- Spectral centroid
- RMS energy

### Machine Learning Model
- Algorithm: Random Forest Regressor
- Evaluation Metric: Mean Absolute Error (MAE)
- Score Range: 1 (poor) to 5 (excellent grammar)

## Results
The model provides:
- Grammar quality predictions with visual feedback
- Audio playback capability
- Clear quality assessment indicators
