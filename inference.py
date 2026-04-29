# 머신러닝과 딥러닝을 결합하여 센서 데이터의 '이상 여부'와 고장 종류를 판별하는 핵심 추론 엔진
# 1단계로는 오토인코더를 통해 이상치를 탐지하고, 2단계로 LightGBM을 통해 세부 분류 수행

# 고도화 진행한 코드
    # 1. 로깅(Logging) 시스템 설정: 에러와 시스템 상태를 파일과 화면에 동시 기록
    # 2. 하드웨어 가속(GPU) 자동 할당
    # 3. 방어적 프로그래밍: 데이터가 없거나 에러가 발생해도 서버가 죽지 않도록 보호
    # 4. 속도 최적화: Pandas를 거치지 않고 Numpy 배열로 직접 변환 (실시간 처리 핵심)
    # 5. 하드코딩 제거: 초기화 시 설정한 lgbm_confidence_threshold 사용
    # 6. 동적 임계치 알고리즘 적용


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
# [추가] 운영 환경을 위한 로깅, 타입 힌트, 동적 임계치용 큐 라이브러리 추가
import logging
from typing import Dict, Any
from collections import deque 

# [추가] 에러 기록 및 시스템 상태 모니터링을 위한 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InferenceEngine")

class Autoencoder(nn.Module):
    """이상치 탐지를 위한 오토인코더 모델 (구조는 기존과 동일하게 유지)"""
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
    # [기존] def __init__(self, model_dir='models'):
    # [변경] 모델 경로 외에 동적 임계치 파라미터와 신뢰도 기준값을 외부에서 받도록 확장
    def __init__(self, model_dir='models', lgbm_confidence_threshold=0.6, window_size=100, std_multiplier=3.0):
        self.model_dir = model_dir
        self.lgbm_confidence_threshold = lgbm_confidence_threshold
        
        # [추가] 동적 임계치 알고리즘 (Sliding Window) 세팅
        self.window_size = window_size         # 최근 몇 개의 데이터를 기준으로 삼을지 (예: 100개)
        self.std_multiplier = std_multiplier   # 표준편차 가중치 (보통 3-Sigma 규칙 사용)
        self.mse_history = deque(maxlen=self.window_size) # 최근 오차(MSE)를 저장할 공간 (오래된 건 자동 삭제됨)

        # [추가] CPU/GPU 자동 할당 (대용량 처리용)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"추론 엔진 초기화 됨. 할당된 디바이스: {self.device}")

        # [추가] 모델 로드 시 에러 방어 로직
        try:
            self._load_models()
        except Exception as e:
            logger.critical(f"모델을 불러오는 중 치명적 오류 발생: {e}")
            raise

    def _load_models(self):
        # [기존] ae_data = torch.load(ae_path, weights_only=False)
        # [변경] GPU/CPU 디바이스 환경에 맞춰서 텐서를 로드하도록 map_location 추가
        ae_path = os.path.join(self.model_dir, 'autoencoder.pth')
        ae_data = torch.load(ae_path, map_location=self.device, weights_only=False)
        self.features = ae_data['features']
        
        # [기존] self.threshold = ae_data['threshold']
        # [변경] 고정 임계치는 '최소 방어선(base_threshold)'으로만 사용
        self.base_threshold = ae_data['threshold']
        
        # [기존] self.model_ae = Autoencoder(len(self.features))
        # [변경] 모델을 로드하자마자 설정된 디바이스(GPU/CPU)로 보냄
        self.model_ae = Autoencoder(len(self.features)).to(self.device)
        self.model_ae.load_state_dict(ae_data['model_state_dict'])
        self.model_ae.eval()
        
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
        self.lgb_model = joblib.load(os.path.join(self.model_dir, 'lightgbm_model.joblib'))
        self.le = joblib.load(os.path.join(self.model_dir, 'label_encoder.joblib'))

    def predict(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """단일 센서 데이터 이상 탐지 및 분류 (동적 임계치 적용)"""
        # [추가] 센서 통신 끊김 등으로 빈 데이터가 올 때 서버 다운 방지
        if not metrics_dict:
            return {"status": "ERROR", "message": "입력 데이터가 없습니다."}

        try:
            # [기존] X_df = pd.DataFrame([row_dict])[self.features] -> 느린 Pandas 방식
            # [변경] Pandas 데이터프레임 생성을 없애고, 초고속 Numpy 배열로 직행
            row_list = [float(metrics_dict.get(f, 0.0)) for f in self.features]
            X_array = np.array(row_list).reshape(1, -1)
            X_scaled = self.scaler.transform(X_array)
            
            # [기존] X_tensor = torch.FloatTensor(X_scaled)
            # [변경] 텐서를 디바이스로 보냄
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            with torch.no_grad():
                recon = self.model_ae(X_tensor)
                mse = torch.mean((X_tensor - recon)**2).item()
            
            # --- [추가/핵심] 동적 임계치 알고리즘 (Sliding Window) ---
            capped_mse = min(mse, self.base_threshold * 2.0)  # 상한선(Cap) 씌우기
            self.mse_history.append(capped_mse)              # 안전하게 기록장에 추가
            
            # 기록장에 데이터가 어느 정도(최소 10개) 쌓이면 동적 계산 시작
            if len(self.mse_history) >= 10: 
                current_mean = np.mean(self.mse_history) # 최근 MSE의 평균
                current_std = np.std(self.mse_history)   # 최근 MSE의 편차(흔들림)
                # 평균 + (표준편차 * 가중치)를 새로운 실시간 임계치로 설정
                dynamic_threshold = current_mean + (self.std_multiplier * current_std)
                
                # 단, 장비가 완전히 고장나서 동적 임계치 자체가 너무 높아지는 것을 막기 위해 기존 고정값과 비교
                current_threshold = max(dynamic_threshold, self.base_threshold)
            else:
                # 초기 구동 시에는 기존의 고정 임계치 사용
                current_threshold = self.base_threshold
            # --------------------------------------------------------
            
            # [기존] is_anomaly = mse > self.threshold
            # [변경] 위에서 계산한 '실시간 동적 임계치'와 현재 오차를 비교
            is_anomaly = mse > current_threshold
            
            probs = self.lgb_model.predict_proba(X_scaled)[0]
            max_prob = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            pred_label = self.le.inverse_transform([pred_idx])[0]
            
            final_status = pred_label
            if is_anomaly:
                # [기존] if max_prob < 0.6 or pred_label == 'Normal':
                # [변경] 하드코딩된 숫자 제거
                if max_prob < self.lgbm_confidence_threshold or pred_label == 'Normal':
                    final_status = "UNKNOWN FAULT"
            else:
                final_status = "Normal"
                
            return {
                'status': final_status,
                'mse': mse,
                'current_threshold': current_threshold, # [추가] 모니터링을 위해 현재 임계치도 반환
                'confidence': max_prob,
                'is_anomaly': is_anomaly,
                'predicted_label': pred_label
            }
            
        # [추가] 처리 중 예외 발생 시 로그를 남기고 시스템 지속
        except Exception as e:
            logger.error(f"예측 중 예외 발생: {str(e)}")
            return {"status": "ERROR", "message": "내부 서버 오류"}

if __name__ == "__main__":
    # 고도화된 엔진 테스트
    engine = InferenceEngine(window_size=50) # 50개 데이터를 윈도우로 사용
    
    # 더미 데이터로 10번 연속 테스트 (동적 임계치 변화 관찰)
    logger.info("연속 데이터 스트리밍 테스트 시작...")
    for i in range(10):
        # 10번째 데이터에 인위적으로 거대한 이상치(노이즈) 주입
        noise = 5.0 if i == 9 else 0.0 
        sample_data = {f"sensor_{j}": np.random.rand() + noise for j in range(16)} 
        
        result = engine.predict(sample_data)
        if result.get("status") != "ERROR":
            print(f"[{i+1}회차] 상태: {result['status']}, MSE: {result['mse']:.4f}, 현재임계치: {result['current_threshold']:.4f}")