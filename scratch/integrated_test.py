import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import time
import threading
import shap
import os
import warnings
from confluent_kafka import Producer, Consumer
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Load Models ---
MODEL_DIR = 'models'
ae_data = torch.load(f'{MODEL_DIR}/autoencoder.pth', weights_only=False)
scaler = joblib.load(f'{MODEL_DIR}/scaler.joblib')
lgb_model = joblib.load(f'{MODEL_DIR}/lightgbm_model.joblib')
le = joblib.load(f'{MODEL_DIR}/label_encoder.joblib')
FEATURES = ae_data['features']
THRESHOLD = ae_data['threshold']

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

model_ae = Autoencoder(len(FEATURES))
model_ae.load_state_dict(ae_data['model_state_dict'])
model_ae.eval()

explainer = shap.TreeExplainer(lgb_model)

TOPIC = 'etching-data-test'
BOOTSTRAP_SERVERS = 'localhost:9092'

def start_consumer():
    conf = {'bootstrap.servers': BOOTSTRAP_SERVERS, 'group.id': f'test_group_{time.time()}', 'auto.offset.reset': 'earliest'}
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])
    print("🧠 [Consumer] Two-Stage Agent ready. Monitoring stream...")
    
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error(): continue
            
            data = json.loads(msg.value().decode('utf-8'))
            
            # Preprocess
            row_dict = {f: float(data['metrics'].get(f, 0.0)) for f in FEATURES}
            X_df = pd.DataFrame([row_dict])[FEATURES]
            X_scaled = scaler.transform(X_df)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # 1. AE Anomaly Score
            with torch.no_grad():
                recon = model_ae(X_tensor)
                mse = torch.mean((X_tensor - recon)**2).item()
            
            is_anomaly = mse > THRESHOLD
            
            # 2. LightGBM Prediction
            probs = lgb_model.predict_proba(X_scaled)[0]
            max_prob = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            pred_label = le.inverse_transform([pred_idx])[0]
            
            # 3. Detection Logic
            if is_anomaly:
                final_status = pred_label
                # Logic: Unknown if low confidence OR predicted Normal despite AE flagging it
                if max_prob < 0.6 or pred_label == 'Normal':
                    final_status = "!! UNKNOWN FAULT !!"
                
                # 4. SHAP Explanation
                # For LGBM multiclass, shap_values returns list of [n_samples, n_features]
                shap_vals = explainer.shap_values(X_scaled)
                # Correct indexing for predicted class
                if isinstance(shap_vals, list):
                    class_shap = shap_vals[pred_idx][0]
                else:
                    # Some shap versions return [samples, features, classes]
                    class_shap = shap_vals[0, :, pred_idx]
                
                top_indices = np.argsort(np.abs(class_shap))[-3:][::-1]
                top_sensors = [FEATURES[i] for i in top_indices]
                
                print(f"🚨 [DETECTION] {data['run_name']} | Status: {final_status:18} | MSE: {mse:.4f} | Conf: {max_prob:.2f} | Root Sensors: {top_sensors}")
            else:
                # Log normal runs occasionally or as dots
                if int(time.time() * 10) % 50 == 0:
                    print(f"✅ [NORMAL] {data['run_name']} | MSE: {mse:.4f}")
    except Exception as e: 
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally: consumer.close()

def start_producer():
    p = Producer({'bootstrap.servers': BOOTSTRAP_SERVERS})
    test_df = pd.read_csv('data/test_split.csv')
    print(f"🚀 [Producer] Streaming {len(test_df)} data points...")
    
    for _, row in test_df.iterrows():
        payload = {"run_name": row['Run_Name'], "metrics": row.to_dict()}
        p.produce(TOPIC, value=json.dumps(payload))
        p.flush()
        time.sleep(0.02) # Faster stream

if __name__ == "__main__":
    t = threading.Thread(target=start_consumer, daemon=True)
    t.start()
    time.sleep(2)
    start_producer()
