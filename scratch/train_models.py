import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import joblib
import os

# --- Configuration ---
TRAIN_FILE = 'data/train_augmented.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Define columns to exclude from training
EXCLUDE_COLS = ['Data_Type', 'Run_Name', 'Time_Step', 'run_id', 'Fault_Name', 'Is_Synthetic', 'Synthesis_Method', 'TIME', 'Time', 'Step Number']

def build_autoencoder(input_dim):
    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))
    return Autoencoder()

def train():
    print("Loading augmented training data...")
    df = pd.read_csv(TRAIN_FILE)
    
    # Feature selection
    # Spectral features often have dots in them (e.g. '250.0')
    features = [c for c in df.columns if c not in EXCLUDE_COLS]
    
    # Clean features: Ensure they are numeric
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Fault_Name']
    
    print(f"Training with {len(features)} features.")
    
    # Scaler fitting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f'{MODEL_DIR}/scaler.joblib')
    
    # --- 1. Autoencoder Training (Normal Data Only) ---
    print("Training Autoencoder on Normal data...")
    X_normal = X_scaled[y == 'Normal']
    if len(X_normal) == 0:
        # Fallback if labels are slightly different
        X_normal = X_scaled[y.str.lower() == 'normal']
        
    X_normal_tensor = torch.FloatTensor(X_normal)
    
    model_ae = build_autoencoder(len(features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_ae.parameters(), lr=0.001)
    
    # Simple training loop
    for epoch in range(100):
        optimizer.zero_grad()
        output = model_ae(X_normal_tensor)
        loss = criterion(output, X_normal_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"AE Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")
            
    # Calculate threshold (95th percentile of normal reconstruction error)
    with torch.no_grad():
        reconstructed = model_ae(X_normal_tensor)
        errors = torch.mean((X_normal_tensor - reconstructed)**2, axis=1).numpy()
        threshold = np.percentile(errors, 95)
        print(f"AE Anomaly Threshold set to: {threshold:.6f}")
        
    torch.save({
        'model_state_dict': model_ae.state_dict(),
        'threshold': threshold,
        'features': features
    }, f'{MODEL_DIR}/autoencoder.pth')

    # --- 2. LightGBM Training (All Data) ---
    print("Training LightGBM Classifier...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, f'{MODEL_DIR}/label_encoder.joblib')
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        objective='multiclass',
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_scaled, y_encoded)
    joblib.dump(lgb_model, f'{MODEL_DIR}/lightgbm_model.joblib')
    
    print(f"Success. Models saved in '{MODEL_DIR}/'.")

if __name__ == "__main__":
    train()
