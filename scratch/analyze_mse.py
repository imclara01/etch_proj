import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import Autoencoder

def analyze():
    # 1. Load models and data
    checkpoint = torch.load('models/autoencoder.pth', map_location='cpu')
    features = checkpoint['features']
    input_dim = len(features)
    
    model = Autoencoder(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = joblib.load('models/scaler.joblib')
    threshold = checkpoint['threshold']
    
    train_df = pd.read_csv('data/train_tstr.csv')
    test_df = pd.read_csv('data/test_tstr.csv')
    
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    
    # Extract Normal data
    synth_norm = train_df[train_df['Fault_Name'] == 'Normal']
    real_norm = test_df[test_df['Fault_Name'] == 'Normal']
    
    # 2. Calculate MSE
    def get_mse(df):
        X = df[features].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            recon = model(X_tensor)
            mse = torch.mean((X_tensor - recon)**2, dim=1).numpy()
        return mse

    synth_mse = get_mse(synth_norm)
    real_mse = get_mse(real_norm)
    
    print(f"--- MSE Statistics ---")
    print(f"Threshold: {threshold:.6f}")
    print(f"Synthetic Normal MSE: Mean={np.mean(synth_mse):.6f}, 95th={np.percentile(synth_mse, 95):.6f}, Max={np.max(synth_mse):.6f}")
    print(f"Real Normal MSE:      Mean={np.mean(real_mse):.6f}, 95th={np.percentile(real_mse, 95):.6f}, Max={np.max(real_mse):.6f}")
    
    fpr = np.mean(real_mse > threshold)
    print(f"False Positive Rate on Real Normal: {fpr:.2%}")
    
    # 3. Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(synth_mse, bins=50, alpha=0.5, label='Synthetic Normal', density=True)
    plt.hist(real_mse, bins=50, alpha=0.5, label='Real Normal', density=True)
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title('MSE Distribution: Synthetic vs Real Normal')
    plt.xlabel('MSE')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('scratch/mse_dist.png')
    print("Chart saved to scratch/mse_dist.png")

if __name__ == '__main__':
    analyze()
