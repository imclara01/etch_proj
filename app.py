import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from confluent_kafka import Consumer
import threading
from concurrent.futures import ThreadPoolExecutor

from inference import InferenceEngine
from shap_analysis import SHAPExplainer
from agents.shap_agent import SHAPAgent
from agents.rag_agent import GraphRAGAgent
from notifications.slack import SlackNotifier

# --- Initialize Environment ---
load_dotenv()
if "OPEN_AI_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]

# --- Page Config ---
st.set_page_config(
    page_title="Gemini AI | Semiconductor Guardian",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    
    /* Simplified Header */
    .main-header {
        font-size: 1.8rem; font-weight: 700; color: #4285F4; margin-bottom: 1rem;
    }
    
    /* Equipment Card */
    .eq-card {
        background-color: #1A1C23; padding: 1rem; border-radius: 12px;
        border: 1px solid #2D2F39; margin-bottom: 0.5rem;
    }
    
    /* Status Colors */
    .status-normal { color: #34A853; font-weight: bold; }
    .status-fault { color: #EA4335; font-weight: bold; }
    .status-unknown { color: #FBBC05; font-weight: bold; }
    
    /* Metrics */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_engines():
    engine = InferenceEngine()
    explainer = SHAPExplainer(engine.lgb_model, engine.features)
    return engine, explainer

engine, explainer = load_engines()

# --- Sidebar Logic ---
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d473530393318e42.svg", width=50)
    st.markdown("<h2 style='color: white;'>Gemini Config</h2>", unsafe_allow_html=True)
    
    api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    if api_key: os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    simulation_active = st.toggle("Start Monitoring", value=False)
    data_source = st.radio("Pipeline Source", ["Local Simulation", "Kafka Stream"])
    slack_active = st.toggle("Enable Slack Notification", value=False)
    if slack_active:
        if not os.getenv("SLACK_WEBHOOK_URL"):
            st.warning("⚠️ Webhook URL missing in .env")
        else:
            st.success("🔔 Slack Alerts: Active")
            if st.button("Send Test Alert"):
                notifier = SlackNotifier()
                notifier.send_alert("MANUAL TEST", "TEST_RUN_001", 0.0, 1.0, "Test message from Dashboard.")
                st.toast("Test alert sent!")
    
    st.markdown("---")
    threshold_override = st.slider("Anomaly Sensitivity", 0.0, 1.0, float(engine.threshold), 0.01)
    engine.threshold = threshold_override
    
    st.caption("Agent Status: Online 🟢")

# --- UI State Management ---
if 'history' not in st.session_state: st.session_state.history = []
if 'mse_trend' not in st.session_state: st.session_state.mse_trend = {} # Per equipment
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}
if 'executor' not in st.session_state: st.session_state.executor = ThreadPoolExecutor(max_workers=5)
if 'active_dialog' not in st.session_state: st.session_state.active_dialog = None

@st.cache_data(show_spinner=False)
def perform_ai_analysis(fault_status, predicted_label, metrics_dict, run_name, api_key, slack_active):
    """Cached function for heavy AI analysis to avoid redundant LLM calls"""
    print(f"🚀 Starting AI Analysis for {run_name} ({fault_status})...")
    
    # 1. SHAP Analysis
    try:
        # Use predicted_label for SHAP even if status is UNKNOWN
        pred_idx = list(engine.le.classes_).index(predicted_label)
        analysis_data = explainer.explain(engine.scaler.transform(pd.DataFrame([metrics_dict])[engine.features]), metrics_dict, pred_idx)
    except Exception as e:
        print(f"❌ SHAP Analysis Failed for {predicted_label}: {e}")
        analysis_data = []

    # 2. LLM SHAP Agent
    explanation = "AI Analysis disabled (No API Key)"
    if api_key:
        try:
            agent = SHAPAgent()
            explanation = agent.explain_fault(fault_status, analysis_data)
        except Exception as e:
            print(f"❌ SHAP Agent Failed: {e}")
            explanation = f"Error during AI analysis: {str(e)}"
    
    # 3. GraphRAG Agent
    recommendation = "No recommendation available"
    try:
        rag_agent = GraphRAGAgent()
        recommendation = rag_agent.get_recommendation(fault_status, shap_analysis=analysis_data)
        rag_agent.close()
    except Exception as e:
        print(f"❌ GraphRAG Failed: {e}")
        recommendation = f"Knowledge retrieval error: {str(e)}"
        
    # 4. Slack Notification
    if slack_active:
        print(f"📨 Attempting to send Slack alert for {fault_status}...")
        try:
            notifier = SlackNotifier()
            notifier.send_alert(fault_status, run_name, 0.0, 1.0, explanation) # MSE/Conf simplified here
        except Exception as e:
            print(f"❌ Slack Alert Failed: {e}")
            
    print(f"✅ AI Analysis Completed for {run_name}")
    return {
        "explanation": explanation,
        "recommendation": recommendation,
        "analysis_data": analysis_data
    }

@st.dialog("🚨 Anomaly Diagnosis")
def show_analysis_dialog(analysis):
    i_col1, i_col2 = st.columns([1, 1])
    with i_col1:
        st.markdown("### 🤖 Root Cause (SHAP)")
        analysis_data = analysis.get('analysis_data', [])
        if analysis_data:
            stats_df = pd.DataFrame(analysis_data)
            display_df = stats_df[['sensor', 'current_value', 'mean_value', 'status']].copy()
            display_df.columns = ['Sensor', 'Value', 'Normal', 'Status']
            st.table(display_df)
        st.markdown(f"<div style='font-size: 0.9rem;'>{analysis['explanation']}</div>", unsafe_allow_html=True)
    
    with i_col2:
        st.markdown("### 🛠 Maintenance Guide (GraphRAG)")
        st.markdown(f"<div style='font-size: 0.9rem;'>{analysis['recommendation']}</div>", unsafe_allow_html=True)

# --- Main Interface ---
st.markdown("<h1 class='main-header'>Semiconductor Anomaly Command Center</h1>", unsafe_allow_html=True)

# Grid Layout for 10 Equipments (2x5)
eq_placeholders = {}
cols = st.columns(2)
for i in range(10):
    eq_name = f"EQ-{i+1:02d}"
    with cols[i % 2]:
        container = st.container()
        eq_placeholders[eq_name] = {
            "status": container.empty(),
            "metrics": container.empty(),
            "chart": container.empty(),
            "button": container.empty()
        }

log_expander = st.expander("📝 Recent Event Logs", expanded=False)

# --- Core Processing Loop ---
if simulation_active:
    # Setup Data Source
    if data_source == "Local Simulation":
        test_df = pd.read_csv('data/test_split.csv')
        data_iterator = test_df.iterrows()
    else:
        conf = {'bootstrap.servers': 'localhost:9092', 'group.id': f'gemini-ui-{time.time()}', 'auto.offset.reset': 'latest'}
        consumer = Consumer(conf)
        consumer.subscribe(['sensor-data-stream'])
        data_iterator = None

    while simulation_active:
        # Data Acquisition
        if data_source == "Local Simulation":
            try:
                _, row = next(data_iterator)
                metrics = row.to_dict()
                run_name = metrics['Run_Name']
                eq_id = "EQ-01" # Mock source for EQ-01
            except StopIteration:
                data_iterator = pd.read_csv('data/test_split.csv').iterrows()
                continue
        else:
            msg = consumer.poll(1.0)
            if msg is None: continue
            data = json.loads(msg.value().decode('utf-8'))
            metrics = data['metrics']
            run_name = data['run_name']
            eq_id = data.get('equipment_id', "EQ-01")

        # AI Inference
        result = engine.predict(metrics)
        
        # Update Trend Data
        if eq_id not in st.session_state.mse_trend: st.session_state.mse_trend[eq_id] = []
        st.session_state.mse_trend[eq_id].append(result['mse'])
        if len(st.session_state.mse_trend[eq_id]) > 30: st.session_state.mse_trend[eq_id].pop(0)

        # Update Equipment Card
        if eq_id in eq_placeholders:
            ph = eq_placeholders[eq_id]
            status_class = "status-normal" if result['status'] == "Normal" else "status-fault"
            ph['status'].markdown(f"### {eq_id} | <span class='{status_class}'>{result['status']}</span>", unsafe_allow_html=True)
            
            m1, m2, m3 = ph['metrics'].columns(3)
            m1.metric("MSE", f"{result['mse']:.3f}")
            m2.metric("Conf", f"{result['confidence']:.1%}")
            m3.metric("Run", run_name)
            
            with ph['chart']:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=st.session_state.mse_trend[eq_id], mode='lines', line=dict(color='#4285F4', width=2), fill='tozeroy'))
                fig.add_hline(y=engine.threshold, line_dash="dash", line_color="#EA4335")
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0), height=120, showlegend=False, yaxis=dict(showticklabels=False), xaxis=dict(showticklabels=False))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Trigger AI Analysis
        if result['status'] != "Normal":
            # Pass both the display status and the internal predicted label for SHAP
            analysis = perform_ai_analysis(result['status'], result['predicted_label'], metrics, run_name, api_key, slack_active)
            
            if eq_id in eq_placeholders:
                if ph['button'].button(f"🔍 View {eq_id} Analysis", key=f"btn_{run_name}"):
                    show_analysis_dialog(analysis)

            st.session_state.history.insert(0, {"Time": time.strftime("%H:%M:%S"), "EQ": eq_id, "Status": result['status'], "MSE": f"{result['mse']:.4f}"})
            if len(st.session_state.history) > 20: st.session_state.history.pop()
            with log_expander:
                st.table(pd.DataFrame(st.session_state.history))

        if data_source == "Local Simulation": time.sleep(0.4)
    
    if data_source == "Kafka Stream": consumer.close()

else:
    st.markdown("""
    <div style='text-align: center; padding: 100px;'>
        <h3 style='color: #888;'>System Standby</h3>
        <p>Please toggle 'Start Monitoring' in the sidebar to begin real-time analysis.</p>
    </div>
    """, unsafe_allow_html=True)
