import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

# --- 1. 모델 및 데이터 로드 ---
MODEL_DIR = 'models'
ae_data = torch.load(f'{MODEL_DIR}/autoencoder.pth', weights_only=False)
scaler = joblib.load(f'{MODEL_DIR}/scaler.joblib')
FEATURES = ae_data['features']
OLD_THRESHOLD = ae_data['threshold']

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

model = Autoencoder(len(FEATURES))
model.load_state_dict(ae_data['model_state_dict'])
model.eval()

print("📊 Validation 데이터셋 분석 중...")
val_df = pd.read_csv('data/val_split.csv')
X_scaled = scaler.transform(val_df[FEATURES].fillna(0))
y_true = (val_df['Fault_Name'] != 'Normal').astype(int)

# --- 2. MSE 계산 ---
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    recon = model(X_tensor)
    mse_list = torch.mean((X_tensor - recon)**2, dim=1).numpy()

val_df['mse'] = mse_list

# --- 3. 보수적 임계값 검색 (Target FPR: 10%) ---
# 정상 데이터의 MSE만 추출
normal_mse = mse_list[y_true == 0]
# FPR 10%를 위한 백분위수 (90th percentile of Normal MSE)
CONSERVATIVE_THRESHOLD = np.percentile(normal_mse, 90)

print(f"✅ 기존 임계값 (Train 95%): {OLD_THRESHOLD:.4f}")
print(f"🚀 보수적 임계값 (Target FPR 10%): {CONSERVATIVE_THRESHOLD:.4f}")

# --- 4. 결과 시각화 ---
plt.figure(figsize=(12, 6))

# Histogram
sns.histplot(data=val_df, x='mse', hue='Fault_Name', element='step', palette='viridis', bins=50)
plt.axvline(OLD_THRESHOLD, color='red', linestyle='--', label=f'Old Threshold ({OLD_THRESHOLD:.3f})')
plt.axvline(CONSERVATIVE_THRESHOLD, color='blue', linestyle='-', label=f'Conservative Threshold ({CONSERVATIVE_THRESHOLD:.3f})')

plt.title('Validation MSE Distribution (Conservative Tuning)', fontsize=15)
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Frequency')
plt.legend()
plt.yscale('log')
plt.grid(alpha=0.3)

report_path = 'threshold_conservative_report.png'
plt.savefig(report_path)
print(f"📸 보수적 분석 결과 도표 저장 완료: {report_path}")

# --- 5. 모델 업데이트 ---
ae_data['threshold'] = CONSERVATIVE_THRESHOLD
torch.save(ae_data, f'{MODEL_DIR}/autoencoder.pth')
print("💾 모델 파일에 보수적 임계값 업데이트 완료.")

# 성능 지표 출력
y_pred = (mse_list > CONSERVATIVE_THRESHOLD).astype(int)
cm = confusion_matrix(y_true, y_pred)
actual_fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
print(f"\n[보수적 최적화 후 성능 지표]")
print(f"Target FPR:  10.00%")
print(f"Actual FPR:  {actual_fpr*100:.2f}%")
print(f"Recall:      {(cm[1,1]/(cm[1,0]+cm[1,1]))*100:.2f}%")
print(f"Confusion Matrix:\n{cm}")
