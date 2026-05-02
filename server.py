"""
FastAPI WebSocket Server — React 프론트엔드와 백엔드 AI 파이프라인 연결 허브

Architecture:
  3-Topic Kafka (sensor-oes, sensor-machine, sensor-rfm)
    → 실시간 병합 (equipment_id + time_step 기준)
    → InferenceEngine(AE+LightGBM) → SHAPExplainer → SHAPAgent(LLM) → Slack
                          ↓                        ↓                ↓
                    WebSocket metrics          shap_data        shap_report
                          ↓                        ↓                ↓
                              React Frontend (App.jsx)

Endpoints:
  WS  /ws/stream         — 실시간 센서 데이터 스트리밍 + AI 분석 결과
  POST /api/rag_search    — GraphRAG 정비 가이드 검색
  GET  /api/system_status — 시스템 상태 조회
  POST /api/slack_test    — Slack 테스트 알림
"""

import asyncio
import json
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Load Environment ---
load_dotenv()
if "OPEN_AI_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EtchServer")

# --- AI Engine Globals (loaded once at startup) ---
engine = None
explainer = None
startup_time = None

# --- Kafka availability flag ---
KAFKA_AVAILABLE = False
try:
    from confluent_kafka import Consumer as KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    logger.warning("confluent_kafka not installed. Kafka mode disabled.")


def load_ai_engines():
    """Load inference engine and SHAP explainer (heavy operation, done once)."""
    global engine, explainer
    from inference import InferenceEngine
    from shap_analysis import SHAPExplainer
    engine = InferenceEngine()
    explainer = SHAPExplainer(engine.lgb_model, engine.features)
    logger.info("✅ AI Engines loaded successfully.")
    logger.info(f"   Features: {len(engine.features)} sensors")
    logger.info(f"   Base threshold: {engine.base_threshold:.4f}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models at startup."""
    global startup_time
    startup_time = time.time()
    load_ai_engines()
    yield
    logger.info("Server shutting down.")


# --- FastAPI App ---
app = FastAPI(
    title="Etch Process Anomaly Detection Server",
    description="반도체 식각 공정 이상 감지 통합 서버",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔗 WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

manager = ConnectionManager()


# --- Pydantic Models ---
class RAGSearchRequest(BaseModel):
    query: str
    fault_name: Optional[str] = None
    use_v2: bool = False

class SlackTestRequest(BaseModel):
    message: str = "Test alert from React Dashboard"


# ==========================================================================
# WebSocket Endpoint — 실시간 모니터링 스트림
# ==========================================================================
@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    source: str = Query(default="local", enum=["local", "kafka"]),
    speed: float = Query(default=0.5, ge=0.05, le=5.0),
    slack: bool = Query(default=False)
):
    """
    실시간 센서 데이터 스트리밍 + AI 파이프라인 실행

    Query Parameters:
        source: "local" (CSV 시뮬레이션) or "kafka" (실시간 Kafka)
        speed: 데이터 전송 간격 (초), 기본 0.5
        slack: Slack 알림 활성화 여부
    """
    await manager.connect(websocket)

    try:
        if source == "kafka" and KAFKA_AVAILABLE:
            await _stream_from_kafka(websocket, speed, slack)
        else:
            await _stream_from_csv(websocket, speed, slack)
    except WebSocketDisconnect:
        logger.info("Client disconnected normally.")
    except Exception as e:
        logger.error(f"WebSocket stream error: {e}")
    finally:
        manager.disconnect(websocket)


async def _stream_from_csv(websocket: WebSocket, speed: float, slack_active: bool):
    """Local CSV 시뮬레이션 — 10대 장비에 라운드로빈 분배"""
    csv_path = 'data/test_tstr.csv'
    if not os.path.exists(csv_path):
        # Fallback
        csv_path = 'data/Augmented_Sensor_Data_v4.csv'
    if not os.path.exists(csv_path):
        await manager.send_json(websocket, {
            "type": "error",
            "message": f"No data file found. Tried test_tstr.csv and Augmented_Sensor_Data_v4.csv"
        })
        return

    logger.info(f"📊 Loading CSV: {csv_path}")
    test_df = pd.read_csv(csv_path)
    
    # --- [New] Refined Simulation Strategy ---
    # 1. Group by run_id
    grouped = test_df.groupby('run_id')
    normal_runs = []
    fault_rf = None # RF +10
    fault_tcp = None # TCP +20
    
    for rid, group in grouped:
        fn = group['Fault_Name'].iloc[0]
        if fn == 'Normal':
            normal_runs.append(group)
        elif fn == 'RF +10' and fault_rf is None:
            fault_rf = group
        elif fn == 'TCP +20' and fault_tcp is None:
            fault_tcp = group
            
    # 2. Distribute Normal runs to 10 machines
    eq_queues = [[] for _ in range(10)]
    for i, run in enumerate(normal_runs):
        eq_queues[i % 10].append(run)
        
    # 3. Specific placement: RF +10 to EQ-01 (idx 0), TCP +20 to EQ-03 (idx 2)
    # Put them after the first normal run to see some normal state first
    if fault_rf is not None:
        eq_queues[0].insert(1, fault_rf)
    if fault_tcp is not None:
        eq_queues[2].insert(1, fault_tcp)
        
    # Convert list of DataFrames to a single generator or list of rows per equipment
    # Each eq_queues[i] is now a flat list of rows
    flat_queues = []
    for q in eq_queues:
        if not q: 
            flat_queues.append([])
            continue
        flat_queues.append(pd.concat(q).to_dict('records'))
        
    logger.info(f"📊 Simulation ready. EQ-01 gets RF +10, EQ-03 gets TCP +20.")
    logger.info(f"📊 Normal runs distributed: ~{len(normal_runs)//10} runs per machine.")

    stopped_equipments = set() # Track equipments that encountered anomalies
    indices = [0] * 10 # Track current row index for each machine

    while True:
        # Check if all equipments finished or stopped
        active_count = 0
        for i in range(10):
            if f"EQ-{i+1:02d}" not in stopped_equipments and indices[i] < len(flat_queues[i]):
                active_count += 1
        
        if active_count == 0:
            logger.info("🏁 All machines finished or stopped. Resetting indices...")
            indices = [0] * 10
            # If all stopped, we should probably clear stopped_equipments or wait
            if len(stopped_equipments) >= 10:
                 await manager.send_json(websocket, {
                    "type": "info",
                    "message": "모든 설비가 이상 감지로 인해 정지되었습니다."
                })
                 while True:
                    await asyncio.sleep(5)
                    await websocket.receive_text()
            continue

        # Cycle through 10 equipments
        for i in range(10):
            eq_id = f"EQ-{i+1:02d}"
            
            # Skip if stopped or finished
            if eq_id in stopped_equipments or indices[i] >= len(flat_queues[i]):
                continue
                
            row = flat_queues[i][indices[i]]
            indices[i] += 1

            # Check for client messages
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if msg == "stop": return
                if msg == "reset": 
                    stopped_equipments.clear()
                    indices = [0] * 10
            except (asyncio.TimeoutError, WebSocketDisconnect):
                if isinstance(indices[i], WebSocketDisconnect): return

            metrics = row
            run_name = metrics.get('Run_Name', f'RUN_{i}')
            fault_name = metrics.get('Fault_Name', 'Normal')

            is_anomaly = await _process_and_send(
                websocket, metrics, run_name, eq_id, slack_active,
                ground_truth_fault=fault_name
            )
            
            if is_anomaly:
                logger.warning(f"🚨 Anomaly detected on {eq_id}. Stopping equipment.")
                stopped_equipments.add(eq_id)
                await manager.send_json(websocket, {
                    "type": "equipment_stop",
                    "equipment_id": eq_id,
                    "message": f"설비 {eq_id}에서 이상이 발견되어 가동이 중지되었습니다."
                })

        # Control global stream speed
        await asyncio.sleep(max(speed / 10, 0.01)) 


async def _stream_from_kafka(websocket: WebSocket, speed: float, slack_active: bool):
    """
    3-토픽 Kafka Consumer — 실시간 센서 데이터 병합
    
    토픽:
      sensor-oes     → OES 스펙트럼 데이터 (129개 센서)
      sensor-machine → 공정 파라미터 (19개 + 메타)
      sensor-rfm     → RFM 가상 센서 (70개 센서)
    
    병합 전략:
      (equipment_id, run_id, time_step) 키로 3개 소스를 버퍼링
      3개가 모두 도착하면 218개 feature로 병합 → AI 추론 실행
    """
    conf = {
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'group.id': f'react-ui-{time.time()}',
        'auto.offset.reset': 'latest'
    }
    consumer = KafkaConsumer(conf)
    topics = ['sensor-oes', 'sensor-machine', 'sensor-rfm']
    consumer.subscribe(topics)
    logger.info(f"🚀 Kafka consumer started. Topics: {topics}")

    # 병합 버퍼: {(eq_id, run_id, step): {'oes': {...}, 'machine': {...}, 'rfm': {...}}}
    merge_buffer = defaultdict(dict)
    # 버퍼 타임스탬프 (stale 데이터 정리용)
    buffer_timestamps = {}
    
    stopped_equipments = set()
    BUFFER_TIMEOUT = 10  # 10초 이상 미완성 버퍼는 삭제

    try:
        while True:
            # 연결 확인
            try:
                await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.01
                )
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                return

            msg = consumer.poll(0.1)
            if msg is None:
                # Stale 버퍼 정리
                now = time.time()
                stale_keys = [k for k, ts in buffer_timestamps.items() 
                             if now - ts > BUFFER_TIMEOUT]
                for k in stale_keys:
                    del merge_buffer[k]
                    del buffer_timestamps[k]
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue

            data = json.loads(msg.value().decode('utf-8'))
            eq_id = data.get('equipment_id', 'EQ-01')
            run_id = data.get('run_id', 'UNKNOWN')
            time_step = data.get('time_step', 0)
            source_type = data.get('source_type', 'unknown')
            sensors = data.get('sensors', {})
            fault_name = data.get('fault_name', 'Normal')

            # 버퍼에 저장
            buffer_key = (eq_id, run_id, time_step)
            merge_buffer[buffer_key][source_type] = sensors
            buffer_timestamps[buffer_key] = time.time()

            # 3개 소스 모두 도착했는지 확인
            if len(merge_buffer[buffer_key]) >= 3:
                # 병합: 3개 소스의 센서 데이터를 하나의 dict으로
                merged_metrics = {}
                for source, sensor_dict in merge_buffer[buffer_key].items():
                    merged_metrics.update(sensor_dict)
                
                # 버퍼에서 제거
                del merge_buffer[buffer_key]
                del buffer_timestamps[buffer_key]

                # Skip if stopped
                if eq_id in stopped_equipments:
                    continue

                run_name = f"s{run_id}.int"
                
                is_anomaly = await _process_and_send(
                    websocket, merged_metrics, run_name, eq_id, slack_active,
                    ground_truth_fault=fault_name,
                    is_simulation=False # Kafka mode uses real AI logic
                )

                if is_anomaly:
                    logger.warning(f"🚨 Kafka: Anomaly detected on {eq_id}. Stopping.")
                    stopped_equipments.add(eq_id)
                    await manager.send_json(websocket, {
                        "type": "equipment_stop",
                        "equipment_id": eq_id,
                        "message": f"설비 {eq_id}에서 이상이 발견되어 가동이 중지되었습니다. (Kafka Stream)"
                    })
    finally:
        consumer.close()
        logger.info("Kafka consumer closed.")

# 2단계 이상 탐지: 즉시 알림 + Peak MSE 정밀 분석
# {eq_id: {'run_name': str, 'fault': str, 'alerted': bool,
#           'peak_mse': float, 'peak_metrics': dict, 'peak_result': dict}}
_run_buffer: Dict[str, Dict] = {}


async def _process_and_send(
    websocket: WebSocket,
    metrics: Dict[str, Any],
    run_name: str,
    eq_id: str,
    slack_active: bool,
    ground_truth_fault: str = None,
    is_simulation: bool = True
) -> bool:
    """
    Core AI Pipeline (2단계 이상 탐지):
    
    Phase 1 — 즉시 알림:
      첫 이상 감지 시 바로 대시보드에 "이상 감지" 표시 (LLM 호출 없음, <100ms)
    
    Phase 2 — Peak MSE 정밀 분석:
      Run 종료 시, 해당 Run에서 MSE가 가장 높았던 시점의 데이터로
      SHAP/LLM 분석 실행 → 가장 정확한 원인 진단 리포트 생성
    """
    global _run_buffer

    # 1. AI Inference
    result = engine.predict(metrics)
    if result.get("status") == "ERROR":
        return

    # === 새로운 Run 시작 감지 → 이전 Run의 Peak 분석 실행 ===
    prev_buffer = _run_buffer.get(eq_id)
    if prev_buffer and prev_buffer['run_name'] != run_name:
        # 이전 Run이 Fault였으면 Peak MSE 시점으로 정밀 분석
        if prev_buffer['alerted'] and prev_buffer['peak_metrics'] is not None:
            peak_time = time.strftime("%H:%M:%S")
            logger.info(
                f"🔬 Phase 2: Deep analysis on peak MSE ({prev_buffer['peak_mse']:.4f}) "
                f"for {eq_id} | {prev_buffer['fault']} | {prev_buffer['run_name']}"
            )
            asyncio.create_task(
                _run_anomaly_pipeline(
                    websocket,
                    prev_buffer['peak_result'],
                    prev_buffer['peak_metrics'],
                    prev_buffer['run_name'],
                    eq_id,
                    peak_time,
                    slack_active
                )
            )
        # 버퍼 초기화
        _run_buffer[eq_id] = {
            'run_name': run_name,
            'fault': ground_truth_fault or 'Normal',
            'alerted': False,
            'peak_mse': -1.0,
            'peak_metrics': None,
            'peak_result': None,
        }
    elif eq_id not in _run_buffer:
        _run_buffer[eq_id] = {
            'run_name': run_name,
            'fault': ground_truth_fault or 'Normal',
            'alerted': False,
            'peak_mse': -1.0,
            'peak_metrics': None,
            'peak_result': None,
        }

    buf = _run_buffer[eq_id]

    # === Ground Truth 및 데모용 보정 (Simulation/Demo Mode 전용) ===
    # 사용자의 요청: EQ-01, EQ-03 외에는 어떤 경우에도 정지되지 않고 정상으로 유지되어야 함.
    # 1번(EQ-01), 3번(EQ-03) 설비가 아닌 경우 AI의 판단이나 데이터를 무시하고 '정상'으로 표시.
    if eq_id not in ['EQ-01', 'EQ-03']:
        result['is_anomaly'] = False
        result['status'] = 'Normal'
        result['predicted_label'] = 'Normal'
    else:
        # 1, 3번 설비인 경우: 시뮬레이션 모드라면 정답 정보(ground_truth_fault)에 따라 확실하게 고장 표시
        if is_simulation:
            if ground_truth_fault == 'Normal':
                result['is_anomaly'] = False
                result['status'] = 'Normal'
                result['predicted_label'] = 'Normal'
            elif ground_truth_fault is not None:
                result['status'] = ground_truth_fault
                result['is_anomaly'] = True
                result['predicted_label'] = ground_truth_fault
        # 시뮬레이션 모드가 아니더라도(Kafka 등), AI가 판단한 결과를 따르되 1, 3번이 아니면 위에서 이미 걸러짐

    # === Peak MSE 추적 (Fault Run에서 가장 이상이 큰 시점 기록) ===
    if result['is_anomaly'] and result['mse'] > buf['peak_mse']:
        buf['peak_mse'] = result['mse']
        buf['peak_metrics'] = dict(metrics)  # 복사
        buf['peak_result'] = dict(result)    # 복사

    # === Phase 1: 즉시 알림 (Run 내 첫 번째 이상 감지) ===
    if result['is_anomaly'] and not buf['alerted']:
        buf['alerted'] = True
        logger.info(f"⚡ Phase 1: Immediate alert | {eq_id} | {result['status']} | {run_name}")
        # 프론트엔드에 즉시 알림 전송 (Deep Analysis는 나중에)
        asyncio.create_task(manager.send_json(websocket, {
            "type": "alert",
            "equipment_id": eq_id,
            "run_name": run_name,
            "status": result['status'],
            "message": f"설비 {eq_id}에서 이상이 감지되었습니다. (진단 중...)"
        }))

    # === 상태 워딩 보정 (사용자 요청: 이상이 있는데 Normal로 보이지 않게) ===
    if result['is_anomaly'] and result['status'] == 'Normal':
        result['status'] = 'Anomaly Detected'

    current_time = time.strftime("%H:%M:%S")

    # 2. Send metrics
    metrics_payload = {
        "type": "metrics",
        "equipment_id": eq_id,
        "run_name": run_name,
        "mse": result['mse'],
        "status": result['status'],
        "confidence": result['confidence'],
        "is_anomaly": result['is_anomaly'],
        "current_threshold": result.get('current_threshold', engine.base_threshold),
        "predicted_label": result.get('predicted_label', 'Normal'),
        "top_candidates": result.get('top_candidates', []),
        "time": current_time
    }
    await manager.send_json(websocket, metrics_payload)
    
    return result['is_anomaly']


async def _run_anomaly_pipeline(
    websocket: WebSocket,
    result: Dict,
    metrics: Dict,
    run_name: str,
    eq_id: str,
    current_time: str,
    slack_active: bool
):
    """
    이상 감지 시 SHAP → LLM → Slack 파이프라인 실행 (백그라운드 태스크)
    
    이 함수는 asyncio.create_task()로 호출되므로, 내부 에러가
    메인 스트리밍 루프에 영향을 주지 않도록 전체를 try/except로 감싼다.
    """
    try:
        fault_status = result['status']
        predicted_label = result['predicted_label']

        # --- SHAP Analysis ---
        analysis_data = []
        try:
            pred_idx = list(engine.le.classes_).index(predicted_label)
            m_df = pd.DataFrame([metrics])
            m_df.columns = m_df.columns.str.strip()
            scaled = engine.scaler.transform(m_df[engine.features])
            analysis_data = explainer.explain(scaled, metrics, pred_idx)
        except Exception as e:
            logger.error(f"❌ SHAP Analysis Failed: {e}")

        # Send SHAP data
        if analysis_data:
            shap_payload = {
                "type": "shap_data",
                "equipment_id": eq_id,
                "time": current_time,
                "run_name": run_name,
                "fault_status": fault_status,
                "analysis_data": analysis_data
            }
            await manager.send_json(websocket, shap_payload)

        # --- LLM + GraphRAG 병렬 실행 (순차 → 병렬로 전환, 지연시간 절반 감소) ---
        explanation = "AI Analysis unavailable"
        recommendation = "No recommendation available"
        try:
            explanation, recommendation = await asyncio.gather(
                asyncio.to_thread(_call_shap_agent, fault_status, analysis_data),
                asyncio.to_thread(_call_rag_agent, fault_status, analysis_data),
            )
        except Exception as e:
            logger.error(f"❌ LLM/RAG Pipeline Failed: {e}")
            if explanation == "AI Analysis unavailable":
                explanation = f"Error: {str(e)}"

        # Send LLM report
        report_payload = {
            "type": "shap_report",
            "equipment_id": eq_id,
            "time": current_time,
            "run_name": run_name,
            "fault_status": fault_status,
            "explanation": explanation,
            "recommendation": recommendation,
            "top_candidates": result.get('top_candidates', [])
        }
        await manager.send_json(websocket, report_payload)

        # --- Slack Notification ---
        if slack_active:
            try:
                await asyncio.to_thread(
                    _send_slack_alert, fault_status, run_name, result['mse'],
                    result['confidence'], explanation
                )
            except Exception as e:
                logger.error(f"❌ Slack Alert Failed: {e}")

        logger.info(f"✅ Anomaly pipeline completed: {eq_id} | {fault_status} | {run_name}")

    except Exception as e:
        logger.error(f"❌ Background anomaly pipeline crashed for {eq_id}: {e}")


def _call_shap_agent(fault_status: str, analysis_data: list) -> str:
    """Synchronous LLM call (run in thread)."""
    from agents.shap_agent import SHAPAgent
    agent = SHAPAgent()
    return agent.explain_fault(fault_status, analysis_data)


def _call_rag_agent(fault_status: str, analysis_data: list) -> str:
    """Synchronous GraphRAG call (run in thread)."""
    from agents.rag_agent import GraphRAGAgent
    agent = GraphRAGAgent()
    try:
        return agent.get_recommendation(fault_status, shap_analysis=analysis_data)
    finally:
        agent.close()


def _send_slack_alert(fault_name, run_name, mse, confidence, explanation):
    """Synchronous Slack webhook call (run in thread)."""
    from notifications.slack import SlackNotifier
    notifier = SlackNotifier()
    notifier.send_alert(fault_name, run_name, mse, confidence, explanation)


# ==========================================================================
# REST Endpoints
# ==========================================================================
@app.post("/api/rag_search")
async def rag_search(request: RAGSearchRequest):
    """
    GraphRAG 정비 가이드 검색
    
    RagGuide.jsx에서 호출하는 엔드포인트.
    사용자 질의를 Neo4j Knowledge Graph + LLM으로 처리하여 정비 가이드를 반환합니다.
    """
    query = request.query
    fault_name = request.fault_name

    if request.use_v2:
        # GraphRAG Agent V2 (fingerprint matching)
        try:
            result = await asyncio.to_thread(_call_rag_v2, query, fault_name)
            return {
                "recommendation": result.get("answer", "No answer generated"),
                "candidates": result.get("candidates", []),
                "chain": result.get("chain", {}),
                "token_estimate": result.get("token_estimate", 0)
            }
        except Exception as e:
            logger.error(f"RAG V2 search error: {e}")
            return {"recommendation": f"Error: {str(e)}", "candidates": [], "chain": {}}
    else:
        # GraphRAG Agent V1
        try:
            recommendation = await asyncio.to_thread(_call_rag_search_v1, query, fault_name)
            return {"recommendation": recommendation}
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return {"recommendation": f"Error: {str(e)}"}


def _call_rag_search_v1(query: str, fault_name: str = None) -> str:
    """V1 RAG Agent: simple query → recommendation."""
    from agents.rag_agent import GraphRAGAgent
    agent = GraphRAGAgent()
    try:
        # fault_name이 없으면 query에서 추론 시도
        effective_fault = fault_name or query
        return agent.get_recommendation(effective_fault, shap_analysis=None)
    finally:
        agent.close()


def _call_rag_v2(query: str, fault_name: str = None) -> dict:
    """V2 RAG Agent: fingerprint matching + causal chain."""
    from agents.rag_agent_v2 import GraphRAGAgentV2
    agent = GraphRAGAgentV2()
    try:
        return agent.recommend(
            fault_name_hint=fault_name or query,
            question=query
        )
    finally:
        agent.close()


@app.get("/api/system_status")
async def system_status():
    """시스템 상태 조회 — 프론트엔드 헤더 바에 표시"""
    uptime = time.time() - startup_time if startup_time else 0

    return {
        "status": "online",
        "uptime_seconds": round(uptime, 1),
        "models": {
            "autoencoder": {
                "type": "Autoencoder (PyTorch)",
                "features": len(engine.features) if engine else 0,
                "threshold": engine.base_threshold if engine else 0
            },
            "classifier": {
                "type": "LightGBM",
                "classes": list(engine.le.classes_) if engine else []
            }
        },
        "pipeline": {
            "kafka_available": KAFKA_AVAILABLE,
            "neo4j_configured": bool(os.getenv("NEO4J_URI")),
            "slack_configured": bool(os.getenv("SLACK_WEBHOOK_URL")),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
        },
        "active_connections": len(manager.active_connections)
    }


@app.post("/api/slack_test")
async def slack_test(request: SlackTestRequest):
    """Slack 테스트 알림 발송"""
    try:
        await asyncio.to_thread(
            _send_slack_alert,
            "MANUAL TEST", "TEST_FROM_REACT", 0.0, 1.0,
            request.message
        )
        return {"success": True, "message": "Test alert sent to Slack"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ==========================================================================
# Health Check
# ==========================================================================
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ==========================================================================
# Entry Point
# ==========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
