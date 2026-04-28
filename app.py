import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from inference import InferenceEngine
from shap_analysis import SHAPExplainer
from agents.shap_agent import SHAPAgent
from agents.rag_agent import GraphRAGAgent
from notifications.slack import SlackNotifier
from confluent_kafka import Consumer
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
if "OPEN_AI_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]

# --- Page Configuration ---
st.set_page_config(
    page_title="Semiconductor Anomaly Detection Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .status-card {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    .status-normal { background-color: #1b4332; color: #74c69d; border: 1px solid #2d6a4f; }
    .status-fault { background-color: #431b1b; color: #ff8787; border: 1px solid #6a2d2d; }
    .status-unknown { background-color: #43361b; color: #ffd8a8; border: 1px solid #6a5a2d; }
</style>
""", unsafe_allow_html=True)

# --- Initialize Engines ---
@st.cache_resource
def load_engines():
    engine = InferenceEngine()
    explainer = SHAPExplainer(engine.lgb_model, engine.features)
    # Note: We'll initialize SHAPAgent per-run or with key
    return engine, explainer

engine, explainer = load_engines()

# --- Sidebar ---
st.sidebar.title("🛠 Settings")
existing_key = os.environ.get("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input("OpenAI API Key", value=existing_key, type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

simulation_active = st.sidebar.toggle("Start Monitoring", value=False)
data_source = st.sidebar.radio("Data Source", ["Local Simulation", "Kafka Stream"])
slack_active = st.sidebar.toggle("Enable Slack Alerts", value=False)
threshold_override = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, float(engine.threshold), 0.01)
engine.threshold = threshold_override

st.sidebar.markdown("---")
if data_source == "Kafka Stream":
    st.sidebar.warning("Ensure Kafka broker (localhost:9092) is running.")
st.sidebar.info("This dashboard monitors semiconductor etching processes in real-time using a Two-Stage AI Agent (Autoencoder + LightGBM).")

# --- Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'mse_trend' not in st.session_state:
    st.session_state.mse_trend = []
if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None

# --- Main Dashboard ---
st.title("🔬 Semiconductor Process Monitoring")

col1, col2, col3 = st.columns(3)

# Placeholders for dynamic content
status_placeholder = st.empty()
metric_col1, metric_col2, metric_col3 = st.columns(3)
chart_placeholder = st.empty()
log_placeholder = st.empty()

# --- Simulation Logic ---
if simulation_active:
    if data_source == "Local Simulation":
        test_df = pd.read_csv('data/test_split.csv')
        data_iterator = test_df.iterrows()
    else:
        # Kafka Consumer Setup
        conf = {
            'bootstrap.servers': 'localhost:9092',
            'group.id': f'streamlit-group-{time.time()}',
            'auto.offset.reset': 'latest'
        }
        consumer = Consumer(conf)
        consumer.subscribe(['sensor-data-stream'])
        data_iterator = None # We'll poll manually

    while simulation_active:
        if data_source == "Local Simulation":
            try:
                _, row = next(data_iterator)
                sample = row.to_dict()
                run_name = sample['Run_Name']
                metrics = sample
            except StopIteration:
                st.info("End of local data. Restarting...")
                data_iterator = pd.read_csv('data/test_split.csv').iterrows()
                continue
        else:
            # Poll from Kafka
            msg = consumer.poll(1.0)
            if msg is None:
                status_placeholder.info("Waiting for Kafka messages...")
                continue
            if msg.error():
                st.error(f"Kafka Error: {msg.error()}")
                break
            
            data = json.loads(msg.value().decode('utf-8'))
            run_name = data['run_name']
            metrics = data['metrics']
            sample = metrics # For compatibility
            
        result = engine.predict(metrics)
        
        # Update History
        st.session_state.mse_trend.append(result['mse'])
        if len(st.session_state.mse_trend) > 50:
            st.session_state.mse_trend.pop(0)
            
        st.session_state.last_detection = {
            'Run_Name': run_name,
            'Status': result['status'],
            'MSE': result['mse'],
            'Confidence': result['confidence'],
            'Result': result
        }
        
        if result['status'] != 'Normal':
            st.session_state.history.insert(0, st.session_state.last_detection)
            if len(st.session_state.history) > 10:
                st.session_state.history.pop()

        # Render Metrics
        with status_placeholder:
            status_class = "status-normal" if result['status'] == 'Normal' else ("status-unknown" if "UNKNOWN" in result['status'] else "status-fault")
            st.markdown(f'<div class="status-card {status_class}">Current Status: {result["status"]}</div>', unsafe_allow_html=True)
            
        metric_col1.metric("Anomaly Score (MSE)", f"{result['mse']:.4f}", delta=f"{result['mse'] - engine.threshold:.4f}", delta_color="inverse")
        metric_col2.metric("LGBM Confidence", f"{result['confidence']*100:.1f}%")
        metric_col3.metric("Processed Run", sample['Run_Name'])
        
        # Render Chart
        with chart_placeholder:
            fig = px.line(y=st.session_state.mse_trend, title="Real-time Anomaly Score (MSE) Trend")
            fig.add_hline(y=engine.threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        # Render Logs and SHAP in two columns
        log_col, shap_col = st.columns([1, 1])
        
        with log_col:
            st.subheader("📋 Detection History")
            if st.session_state.history:
                history_df = pd.DataFrame(st.session_state.history)[['Run_Name', 'Status', 'MSE', 'Confidence']]
                st.dataframe(history_df, use_container_width=True)
            else:
                st.write("No anomalies detected yet.")
                
        with shap_col:
            st.subheader("🎯 Root Cause Analysis")
            if result['status'] != 'Normal':
                pred_idx = list(engine.le.classes_).index(result['predicted_label'])
                sensors, contribs = explainer.explain(result['scaled_features'], pred_idx)
                
                # Plot SHAP contributions
                shap_df = pd.DataFrame({'Sensor': list(contribs.keys()), 'Impact': list(contribs.values())})
                fig_shap = px.bar(shap_df, x='Impact', y='Sensor', orientation='h', title=f"Feature Impact for {result['status']}")
                fig_shap.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig_shap, use_container_width=True)
                
                # --- AI Explanation ---
                st.subheader("🤖 AI Engineer Analysis")
                if api_key:
                    with st.spinner("Analyzing fault patterns..."):
                        shap_agent = SHAPAgent()
                        explanation = shap_agent.explain_fault(result['status'], contribs)
                        st.write(explanation)
                    
                    st.subheader("🛠 Recommended SOP (GraphRAG)")
                    with st.spinner("Retrieving countermeasures from Knowledge Graph..."):
                        rag_agent = GraphRAGAgent()
                        try:
                            recommendation = rag_agent.get_recommendation(result['status'])
                            st.write(recommendation)
                        finally:
                            rag_agent.close()
                    
                    # --- Slack Notification ---
                    if slack_active:
                        notifier = SlackNotifier()
                        notifier.send_alert(
                            result['status'], 
                            sample['Run_Name'], 
                            result['mse'], 
                            result['confidence'],
                            explanation if api_key else "AI explanation disabled."
                        )
                else:
                    st.warning("Please enter your OpenAI API Key in the sidebar to enable AI analysis.")
            else:
                st.info("System operating normally. No root cause analysis required.")

        if data_source == "Local Simulation":
            time.sleep(0.5)
    
    if data_source == "Kafka Stream":
        consumer.close()
else:
    st.warning("Monitoring is paused. Start monitoring to see real-time data.")
    if st.session_state.last_detection:
        st.write("Last processed run summary:")
        st.json(st.session_state.last_detection)
