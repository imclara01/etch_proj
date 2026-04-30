import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import Autoencoder

def check_synth_faults():
    checkpoint = torch.load('models/autoencoder.pth', map_location='cpu')
    features = checkpoint['features']
    model = Autoencoder(len(features))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = joblib.load('models/scaler.joblib')
    
    df = pd.read_csv('data/train_tstr.csv')
    df.columns = df.columns.str.strip()
    
    synth_f = df[(df['Is_Synthetic'] == 1) & (df['Fault_Name'] != 'Normal')]
    
    X = synth_f[features].values
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        recon = model(X_tensor)
        mse = torch.mean((X_tensor - recon)**2, dim=1).numpy()
    
    print(f"Synthetic Fault MSE (in train_tstr.csv):")
    print(f"Mean={np.mean(mse):.6f}, Min={np.min(mse):.6f}, Max={np.max(mse):.6f}")
    
    # Per-fault in synth
    for fault in sorted(synth_f['Fault_Name'].unique()):
        f_sub = synth_f[synth_f['Fault_Name'] == fault]
        X_sub = f_sub[features].values
        X_s_sub = scaler.transform(X_sub)
        X_t_sub = torch.FloatTensor(X_s_sub)
        with torch.no_grad():
            r_sub = model(X_t_sub)
            m_sub = torch.mean((X_t_sub - r_sub)**2, dim=1).numpy()
        print(f"  {fault:15}: Mean MSE={np.mean(m_sub):.6f}")

if __name__ == '__main__':
    check_synth_faults()
