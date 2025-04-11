import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# Set base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    # Paths
    features_path = os.path.join(BASE_DIR, 'dataset', 'train_features.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'grammar_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    
    # Create models directory if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(features_path)
    X = train_df.drop(['id', 'label'], axis=1)
    y = train_df['label']
    
    # Split and scale data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.4f}")
    print(f"Validation MAE: {mean_absolute_error(y_val, val_pred):.4f}")
    
    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()