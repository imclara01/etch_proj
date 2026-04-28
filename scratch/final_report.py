import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# --- 1. 모델 및 데이터 로드 ---
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

print(f"📊 최종 보고서 생성 중... (임계값: {THRESHOLD:.4f})")
test_df = pd.read_csv('data/test_split.csv')
X_scaled = scaler.transform(test_df[FEATURES].fillna(0))

# --- 2. 2단계 추론 루프 ---
results = []
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    recon = model_ae(X_tensor)
    mse_list = torch.mean((X_tensor - recon)**2, dim=1).numpy()
    
    probs = lgb_model.predict_proba(X_scaled)
    max_probs = np.max(probs, axis=1)
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = le.inverse_transform(pred_indices)

for i in range(len(test_df)):
    is_anomaly = mse_list[i] > THRESHOLD
    pred_label = pred_labels[i]
    max_conf = max_probs[i]
    
    final_status = pred_label
    if is_anomaly:
        if max_conf < 0.6 or pred_label == 'Normal':
            final_status = "UNKNOWN FAULT"
    
    results.append({
        'Run_Name': test_df.iloc[i]['Run_Name'],
        'Actual': test_df.iloc[i]['Fault_Name'],
        'Predicted': final_status,
        'MSE': mse_list[i],
        'Confidence': max_conf
    })

res_df = pd.DataFrame(results)

# --- 3. 통계 계산 ---
# 미지 결함 탐지율 (Unknown Faults in test set)
unknown_runs = res_df[res_df['Actual'].str.contains('Unknown', na=False)]
if len(unknown_runs) == 0:
    # 데이터셋에 Unknown 이라는 글자가 없을 수 있으니 Run_Name으로 필터링 (s3341.int 등)
    # 실제 split에서 5개 Unknown Run을 격리했었음
    # s3341.int, s3354.int, s3366.int 등 (OES 데이터에서 선정됨)
    unknown_mask = res_df['Actual'] != 'Normal' # Test set에는 정상과 미지 결함만 있음
    unknown_runs = res_df[unknown_mask]

detection_rate = (unknown_runs['Predicted'] == 'UNKNOWN FAULT').mean() * 100
normal_runs = res_df[res_df['Actual'] == 'Normal']
fpr = (normal_runs['Predicted'] != 'Normal').mean() * 100

print("\n" + "="*40)
print("🚀 [최종 탐지 성적표]")
print("="*40)
print(f"1. 미지 결함 탐지율: {detection_rate:.2f}%")
print(f"2. 정상 데이터 오탐율: {fpr:.2f}%")
print("-" * 40)
print("3. Run별 탐지 상세 (미지 결함):")
run_summary = res_df.groupby(['Run_Name', 'Actual'])['Predicted'].apply(lambda x: (x == 'UNKNOWN FAULT').mean() * 100).reset_index()
print(run_summary[run_summary['Actual'] != 'Normal'])

# --- 4. 시각화 ---
plt.figure(figsize=(15, 7))
sns.scatterplot(data=res_df, x=res_df.index, y='MSE', hue='Actual', style='Predicted', alpha=0.6)
plt.axhline(THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD:.3f})')
plt.title('Final Test Set Detection Results (AE + LightGBM)', fontsize=16)
plt.xlabel('Data Point Index')
plt.ylabel('Anomaly Score (MSE)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()

report_img = 'final_detection_report.png'
plt.savefig(report_img)
print(f"\n📸 최종 시각화 보고서 저장 완료: {report_img}")

# CSV 요약 저장
res_df.to_csv('final_results_summary.csv', index=False)
