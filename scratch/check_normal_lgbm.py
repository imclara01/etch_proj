import pandas as pd
import numpy as np
import joblib
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_normal():
    lgb_model = joblib.load('models/lightgbm_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    le = joblib.load('models/label_encoder.joblib')
    
    checkpoint = torch.load('models/autoencoder.pth', map_location='cpu')
    features = checkpoint['features']
    
    test_df = pd.read_csv('data/test_tstr.csv')
    test_df.columns = test_df.columns.str.strip()
    
    normal = test_df[test_df['Fault_Name'] == 'Normal']
    X = normal[features].values
    X_scaled = scaler.transform(X)
    
    probs = lgb_model.predict_proba(X_scaled)
    max_probs = np.max(probs, axis=1)
    preds = le.inverse_transform(np.argmax(probs, axis=1))
    
    # Check how many normal samples are predicted as faults with high confidence
    fault_mask = (preds != 'Normal')
    high_conf_fault_mask = (preds != 'Normal') & (max_probs > 0.9)
    
    print(f"Normal samples misclassified as faults: {np.mean(fault_mask):.2%}")
    print(f"Normal samples misclassified as faults with >0.9 confidence: {np.mean(high_conf_fault_mask):.2%}")
    
    if np.any(high_conf_fault_mask):
        print(f"Sample of misclassified labels: {pd.Series(preds[high_conf_fault_mask]).value_counts().head()}")

if __name__ == '__main__':
    check_normal()
