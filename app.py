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

# --- Premium Gemini Design ---
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0B0E14;
        color: #E3E3E3;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4285F4, #9B72F3, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    /* Card Component */
    .gemini-card {
        background-color: #1A1C23;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #2D2F39;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    /* Status Badge */
    .status-badge {
        padding: 6px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-normal { background: rgba(52, 168, 83, 0.1); color: #34A853; border: 1px solid #34A853; }
    .status-fault { background: rgba(234, 67, 53, 0.1); color: #EA4335; border: 1px solid #EA4335; }
    .status-unknown { background: rgba(251, 188, 5, 0.1); color: #FBBC05; border: 1px solid #FBBC05; }
    
    /* Metrics Customization */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
    
    /* Sidebar glassmorphism */
    .stSidebar {
        background-color: #111318 !important;
        border-right: 1px solid #2D2F39;
    }
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
if 'mse_trend' not in st.session_state: st.session_state.mse_trend = []
if 'last_detection' not in st.session_state: st.session_state.last_detection = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}
if 'executor' not in st.session_state: st.session_state.executor = ThreadPoolExecutor(max_workers=3)

def run_deep_analysis(result, run_name, metrics, api_key, slack_active):
    """Background task for heavy AI analysis"""
    analysis_key = f"{run_name}_{result['status']}"
    print(f"🚀 Starting Deep Analysis for {analysis_key}...")
    
    # 1. SHAP Analysis
    try:
        pred_idx = list(engine.le.classes_).index(result['predicted_label'])
        sensors, contribs = explainer.explain(result['scaled_features'], pred_idx)
    except Exception as e:
        print(f"❌ SHAP Analysis Failed: {e}")
        contribs = {}

    # 2. LLM SHAP Agent
    explanation = "AI Analysis disabled (No API Key)"
    if api_key:
        try:
            agent = SHAPAgent()
            explanation = agent.explain_fault(result['status'], contribs)
        except Exception as e:
            print(f"❌ SHAP Agent Failed: {e}")
            explanation = f"Error during AI analysis: {str(e)}"
    
    # 3. GraphRAG Agent
    recommendation = "No recommendation available"
    try:
        rag_agent = GraphRAGAgent()
        recommendation = rag_agent.get_recommendation(result['status'])
        rag_agent.close()
    except Exception as e:
        print(f"❌ GraphRAG Failed: {e}")
        recommendation = f"Knowledge retrieval error: {str(e)}"
        
    # 4. Slack Notification
    if slack_active:
        print(f"📨 Attempting to send Slack alert for {result['status']}...")
        try:
            notifier = SlackNotifier()
            notifier.send_alert(result['status'], run_name, result['mse'], result['confidence'], explanation)
        except Exception as e:
            print(f"❌ Slack Alert Failed: {e}")
        
    # Store result back to session state
    st.session_state.analysis_results[analysis_key] = {
        "explanation": explanation,
        "recommendation": recommendation,
        "contribs": contribs
    }
    print(f"✅ Deep Analysis Completed for {analysis_key}")

# --- Main Interface ---
st.markdown("<h1 class='main-header'>Semiconductor Anomaly Guardian</h1>", unsafe_allow_html=True)

# 1. Top Metrics Section
m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 1])

# 2. Main Chart Area
chart_container = st.empty()

# 3. AI Insights Area (Conditionals)
insight_container = st.empty()

# 4. Logs Area
log_expander = st.expander("📝 System Event Logs", expanded=False)

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
            except StopIteration:
                data_iterator = pd.read_csv('data/test_split.csv').iterrows()
                continue
        else:
            msg = consumer.poll(1.0)
            if msg is None: continue
            data = json.loads(msg.value().decode('utf-8'))
            metrics = data['metrics']
            run_name = data['run_name']

        # AI Inference
        result = engine.predict(metrics)
        st.session_state.mse_trend.append(result['mse'])
        if len(st.session_state.mse_trend) > 60: st.session_state.mse_trend.pop(0)

        # Update Top Metrics
        status_style = "status-normal" if result['status'] == "Normal" else ("status-unknown" if "UNKNOWN" in result['status'] else "status-fault")
        
        with m_col1:
            st.markdown(f"**Current Status**\n<div class='status-badge {status_style}'>{result['status']}</div>", unsafe_allow_html=True)
        m_col2.metric("MSE Score", f"{result['mse']:.4f}", delta=f"{result['mse']-engine.threshold:.3f}", delta_color="inverse")
        m_col3.metric("LGBM Confidence", f"{result['confidence']*100:.1f}%")
        m_col4.metric("Active Run", run_name)

        # Update Main Chart
        with chart_container:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.mse_trend, mode='lines', line=dict(color='#4285F4', width=3), fill='tozeroy', fillcolor='rgba(66, 133, 244, 0.1)', name='MSE Score'))
            fig.add_hline(y=engine.threshold, line_dash="dash", line_color="#EA4335", annotation_text="Threshold", annotation_font_color="#EA4335")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), height=350, yaxis=dict(gridcolor='#2D2F39'), xaxis=dict(showticklabels=False))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Update Insights (Only if Anomaly)
        if result['status'] != "Normal":
            analysis_key = f"{run_name}_{result['status']}"
            
            # Start analysis if not already done or in progress
            if analysis_key not in st.session_state.analysis_results:
                st.session_state.executor.submit(run_deep_analysis, result, run_name, metrics, api_key, slack_active)
                st.session_state.analysis_results[analysis_key] = "PENDING"

            with insight_container.container():
                st.markdown("<div class='gemini-card'>", unsafe_allow_html=True)
                i_col1, i_col2 = st.columns([1, 1])
                
                # Check if analysis is complete
                analysis = st.session_state.analysis_results.get(analysis_key)
                
                if analysis == "PENDING":
                    with i_col1:
                        st.markdown("### 🤖 Root Cause Analysis (SHAP)")
                        st.info("🔍 AI is performing deep analysis... (Data stream continues)")
                    with i_col2:
                        st.markdown("### 🛠 Maintenance Guide (GraphRAG)")
                        st.info("⏳ Retrieving knowledge graph insights...")
                elif isinstance(analysis, dict):
                    # SHAP Analysis Display
                    with i_col1:
                        st.markdown("### 🤖 Root Cause Analysis (SHAP)")
                        contribs = analysis['contribs']
                        shap_df = pd.DataFrame({'Sensor': list(contribs.keys()), 'Impact': list(contribs.values())})
                        fig_shap = px.bar(shap_df, x='Impact', y='Sensor', orientation='h', color='Impact', color_continuous_scale='RdBu_r')
                        fig_shap.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=250, coloraxis_showscale=False)
                        st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
                        st.markdown(f"<div style='font-size: 0.95rem; color: #CCC;'>{analysis['explanation']}</div>", unsafe_allow_html=True)

                    # GraphRAG Recommendation Display
                    with i_col2:
                        st.markdown("### 🛠 Maintenance Guide (GraphRAG)")
                        st.markdown(f"<div style='font-size: 0.95rem; color: #CCC;'>{analysis['recommendation']}</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Update Log
            st.session_state.history.insert(0, {"Time": time.strftime("%H:%M:%S"), "Run": run_name, "Status": result['status'], "MSE": f"{result['mse']:.4f}"})
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
