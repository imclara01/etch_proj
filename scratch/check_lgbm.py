import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_lgbm():
    lgb_model = joblib.load('models/lightgbm_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    le = joblib.load('models/label_encoder.joblib')
    
    checkpoint = torch.load('models/autoencoder.pth', map_location='cpu')
    features = checkpoint['features']
    
    test_df = pd.read_csv('data/test_tstr.csv')
    test_df.columns = test_df.columns.str.strip()
    
    print(f"--- LGBM Prediction on Missed Faults ---")
    
    for fault in sorted(test_df['Fault_Name'].unique()):
        if fault == 'Normal': continue
        
        df_sub = test_df[test_df['Fault_Name'] == fault]
        X = df_sub[features].values
        X_scaled = scaler.transform(X)
        
        probs = lgb_model.predict_proba(X_scaled)
        max_probs = np.max(probs, axis=1)
        preds = le.inverse_transform(np.argmax(probs, axis=1))
        
        correct_mask = (preds == fault)
        print(f"{fault:15}: Accuracy={np.mean(correct_mask):.2%}, Avg Confidence={np.mean(max_probs):.4f}")

if __name__ == '__main__':
    import torch # Needed for checkpoint load
    check_lgbm()
