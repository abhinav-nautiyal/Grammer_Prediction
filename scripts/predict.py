import os
import pandas as pd
import joblib

# Set base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def predict_test_set():
    # Paths
    features_path = os.path.join(BASE_DIR, 'dataset', 'test_features.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'grammar_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    output_path = os.path.join(BASE_DIR, 'dataset', 'submission.csv')
    
    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    test_df = pd.read_csv(features_path)
    
    # Prepare data
    X_test = test_df.drop('id', axis=1)
    X_test_scaled = scaler.transform(X_test)
    
    # Predict and save
    predictions = model.predict(X_test_scaled)
    submission = pd.DataFrame({
        'filename': test_df['id'] + '.wav',
        'label': predictions.clip(1, 5)  # Ensure scores are between 1-5
    })
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    predict_test_set()