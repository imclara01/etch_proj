import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
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

class InferenceEngine:
    def __init__(self, model_dir='models'):
        # Load AE
        ae_path = os.path.join(model_dir, 'autoencoder.pth')
        ae_data = torch.load(ae_path, weights_only=False)
        self.features = ae_data['features']
        self.threshold = ae_data['threshold']
        
        self.model_ae = Autoencoder(len(self.features))
        self.model_ae.load_state_dict(ae_data['model_state_dict'])
        self.model_ae.eval()
        
        # Load Scaler, LGBM, LabelEncoder
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        self.lgb_model = joblib.load(os.path.join(model_dir, 'lightgbm_model.joblib'))
        self.le = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
        
    def predict(self, metrics_dict):
        """
        Two-stage inference:
        1. Autoencoder for anomaly detection (MSE > threshold)
        2. LightGBM for fault classification
        """
        # Preprocess
        row_dict = {f: float(metrics_dict.get(f, 0.0)) for f in self.features}
        X_df = pd.DataFrame([row_dict])[self.features]
        X_scaled = self.scaler.transform(X_df)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # 1. Anomaly Detection (AE)
        with torch.no_grad():
            recon = self.model_ae(X_tensor)
            mse = torch.mean((X_tensor - recon)**2).item()
        
        is_anomaly = mse > self.threshold
        
        # 2. Fault Classification (LightGBM)
        probs = self.lgb_model.predict_proba(X_scaled)[0]
        max_prob = float(np.max(probs))
        pred_idx = int(np.argmax(probs))
        pred_label = self.le.inverse_transform([pred_idx])[0]
        
        # 3. Final Decision Logic
        final_status = pred_label
        if is_anomaly:
            # Logic: If AE flags it but LGBM is low confidence or says Normal, it's an Unknown Fault
            if max_prob < 0.6 or pred_label == 'Normal':
                final_status = "UNKNOWN FAULT"
        else:
            final_status = "Normal"
            
        return {
            'status': final_status,
            'mse': mse,
            'confidence': max_prob,
            'is_anomaly': is_anomaly,
            'predicted_label': pred_label,
            'scaled_features': X_scaled
        }

if __name__ == "__main__":
    # Simple test with a sample row from test_split.csv
    engine = InferenceEngine()
    test_df = pd.read_csv('data/test_split.csv')
    sample = test_df.iloc[0].to_dict()
    result = engine.predict(sample)
    print(f"Sample Test Results: {result['status']} (MSE: {result['mse']:.4f}, Conf: {result['confidence']:.2f})")
