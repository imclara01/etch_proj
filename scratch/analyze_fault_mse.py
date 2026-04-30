import pandas as pd
import numpy as np
import torch
import joblib
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
    
    test_df = pd.read_csv('data/test_tstr.csv')
    test_df.columns = test_df.columns.str.strip()
    
    print(f"--- Per-Fault MSE Statistics (Real Test Data) ---")
    print(f"Base Threshold: {threshold:.6f}")
    
    results = []
    for fault in sorted(test_df['Fault_Name'].unique()):
        df_sub = test_df[test_df['Fault_Name'] == fault]
        X = df_sub[features].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            recon = model(X_tensor)
            mse = torch.mean((X_tensor - recon)**2, dim=1).numpy()
        
        det_rate = np.mean(mse > threshold)
        print(f"{fault:15}: Mean MSE={np.mean(mse):.6f}, 95th={np.percentile(mse, 95):.6f}, Det Rate={det_rate:.2%}")
        results.append({'Fault': fault, 'Mean_MSE': np.mean(mse), 'Det_Rate': det_rate})

if __name__ == '__main__':
    analyze()
