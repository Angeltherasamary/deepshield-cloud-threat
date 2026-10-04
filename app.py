import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="CloudShield SOC: Autonomous Threat Inspector", layout="wide")

@st.cache_resource
def load_app_data():
    if not os.path.exists("cloudshield_model.keras"):
        return None
    try:
        model = load_model("cloudshield_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_app_data()

st.title("CloudShield SOC: Autonomous Threat Inspector")

if model is None:
    st.error("Model not found. Please ensure train.py has been executed successfully before running the application.")
    st.stop()

classes = ['Benign', 'DDoS Flood', 'PortScan', 'BruteForce']
scenarios = [
    "Normal API Traffic",
    "High-Volume DDoS Surge",
    "Reconnaissance PortScan",
    "Privilege Escalation BruteForce"
]
scenario_to_class = {scenarios[i]: i for i in range(4)}

def generate_live_sample(target_class):
    n_features = 20
    timesteps = 10
    X = np.random.normal(loc=0.0, scale=0.85, size=(1, timesteps, n_features))
    X[0] += np.random.normal(0, 0.25, size=(timesteps, n_features))
    
    if target_class == 1:
        ramp = np.linspace(0.8, 2.2, timesteps)[:, None] * np.random.uniform(0.7, 1.2)
        X[0, :, :5] += ramp + np.random.normal(0.4, 0.3, size=(timesteps, 5))
    elif target_class == 2:
        scan_steps = np.random.choice(timesteps, size=np.random.randint(4, 8), replace=False)
        X[0, scan_steps, 5:10] -= np.random.uniform(1.0, 1.8, size=(len(scan_steps), 5))
    elif target_class == 3:
        spike_mask = np.random.binomial(1, 0.45, size=(timesteps, 5))
        X[0, :, 15:] += spike_mask * np.random.normal(1.4, 0.5, size=(timesteps, 5))
    return X[0]

col1, col2 = st.columns([1, 2])

with col1:
    st.header("SOC Controls")
    selected_scenario = st.selectbox("Select Traffic Scenario:", scenarios)
    inspect_btn = st.button("Scan New Sample Batch", use_container_width=True)

with col2:
    st.header("Threat Analysis Dashboard")
    
    target_class = scenario_to_class[selected_scenario]
    sample = generate_live_sample(target_class)
    
    sample_batched = np.expand_dims(sample, axis=0)
    
    # Predict with temperature scaling for realistic confidence spreads
    probs_raw = model.predict(sample_batched)[0]
    temperature = 2.2
    logits = np.log(np.clip(probs_raw, 1e-7, 1.0))
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # numeric stability
    preds = exp_logits / np.sum(exp_logits)
    
    pred_class = np.argmax(preds)
    confidence = preds[pred_class]
    
    st.subheader("System Status")
    if pred_class == 0:
        st.success(f"✅ BENIGN TRAFFIC DETECTED (Confidence: {confidence*100:.2f}%)")
    else:
        st.error(f"🚨 ACTIVE THREAT: {classes[pred_class].upper()} DETECTED (Confidence: {confidence*100:.2f}%)")
        
    prob_df = pd.DataFrame({
        'Threat Class': classes,
        'Probability (%)': preds * 100
    })
    
    fig_bar = px.bar(prob_df, x='Probability (%)', y='Threat Class', orientation='h',
                     title='Live Threat Probabilities',
                     color='Probability (%)', color_continuous_scale='Reds' if pred_class != 0 else 'Greens')
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_range=[0, 100])
    st.plotly_chart(fig_bar, use_container_width=True)
    
    time_steps = np.arange(1, 11)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=time_steps, y=sample[:, 0], mode='lines+markers', name='Packet Volume (f0)'))
    fig_line.add_trace(go.Scatter(x=time_steps, y=sample[:, 6], mode='lines+markers', name='Port Frequency (f6)'))
    fig_line.add_trace(go.Scatter(x=time_steps, y=sample[:, 18], mode='lines+markers', name='Auth Retries (f18)'))
    fig_line.update_layout(title="Multi-Channel Temporal Waveform", xaxis_title="Sequence Step", yaxis_title="Normalized Sensor Value")
    st.plotly_chart(fig_line, use_container_width=True)
    
    with st.expander("View Raw Normalized Sensor Data"):
        raw_df = pd.DataFrame(sample, columns=[f"Feature_{i}" for i in range(20)])
        raw_df.index.name = "Time Step"
        st.dataframe(raw_df.style.background_gradient(cmap='viridis'))
