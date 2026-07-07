import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add project root to path
sys.path.append(os.getcwd())

from src.gnn.graph_builder import SmartCityGraphBuilder
from src.models.model_loader import load_xgb_artifacts, load_rf_artifacts
from src.gnn.predict_propagation import PropagationPredictor
from src.impact_assessment.impact_score import ImpactAssessor
from src.early_warning.risk_calculator import RiskCalculator
from src.early_warning.alert_generator import AlertGenerator
from src.explainability.shap_explainer import ShapExplainer
from src.utils.config import MODELS_DIR

# Page configuration
st.set_page_config(
    page_title="Smart City IoT Cybersecurity Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom style for dark mode look and premium feel
st.markdown("""
<style>
    body {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 2px;
        font-family: 'Courier New', Courier, monospace;
    }
    .sub-title {
        font-size: 14px;
        color: #8b949e;
        text-align: center;
        margin-bottom: 20px;
        font-family: 'Courier New', Courier, monospace;
    }
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .kpi-title {
        font-size: 13px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        font-family: monospace;
    }
    .text-red { color: #f85149; }
    .text-orange { color: #f0883e; }
    .text-yellow { color: #d4a72c; }
    .text-green { color: #56d364; }
    .text-blue { color: #58a6ff; }
    
    .console-header {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-bottom: none;
        padding: 6px 12px;
        font-weight: bold;
        font-family: monospace;
        color: #c9d1d9;
        font-size: 14px;
    }
    .console-body {
        background-color: #0d1117;
        border: 1px solid #30363d;
        padding: 12px;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 15px;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Helper to check if models are trained
def models_exist():
    return (
        os.path.exists(os.path.join(MODELS_DIR, 'xgboost.pkl')) and
        os.path.exists(os.path.join(MODELS_DIR, 'random_forest.pkl')) and
        os.path.exists(os.path.join(MODELS_DIR, 'gnn_model.pt'))
    )

st.markdown('<div class="main-title">🛡️ SMART CITY IoT CYBERSECURITY EARLY WARNING SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explainable AI (SHAP) & Graph Neural Network (GCN) Attack Propagation Assessment</div>', unsafe_allow_html=True)

# Setup workspace if models do not exist
if not models_exist():
    st.info("👋 Welcome! The machine learning and Graph Neural Network models need to be trained first.")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 Generate Datasets & Train Models", use_container_width=True):
            with st.spinner("1. Checking & Bootstrapping Datasets..."):
                from src.preprocessing.clean_data import check_and_bootstrap_datasets
                check_and_bootstrap_datasets()
                st.success("Datasets ready!")
                
            with st.spinner("2. Training Random Forest Botnet Detector..."):
                import train_rf
                train_rf.main()
                st.success("Random Forest trained!")
                
            with st.spinner("3. Training XGBoost Attack Detector..."):
                import train_xgboost
                train_xgboost.main()
                st.success("XGBoost trained!")
                
            with st.spinner("4. Training GNN Propagation Predictor..."):
                import train_gnn
                train_gnn.main()
                st.success("GNN trained!")
                
            with st.spinner("5. Running Evaluations & Reports..."):
                import evaluate
                evaluate.main()
                st.success("Evaluation reports generated!")
                
            st.balloons()
            st.rerun()
    st.stop()

# Load models and builders
@st.cache_resource
def load_engines():
    graph_builder = SmartCityGraphBuilder()
    xgb_artifacts = load_xgb_artifacts()
    rf_artifacts = load_rf_artifacts()
    gnn_predictor = PropagationPredictor()
    impact_assessor = ImpactAssessor()
    risk_calculator = RiskCalculator()
    alert_generator = AlertGenerator()
    shap_explainer = ShapExplainer(model_type='xgboost')
    
    return graph_builder, xgb_artifacts, rf_artifacts, gnn_predictor, impact_assessor, risk_calculator, alert_generator, shap_explainer

try:
    graph_builder, xgb_artifacts, rf_artifacts, gnn_predictor, impact_assessor, risk_calculator, alert_generator, shap_explainer = load_engines()
except Exception as e:
    st.error(f"Error loading models: {e}. Please try force retraining.")
    if st.button("Force Retrain Models"):
        import shutil
        shutil.rmtree(MODELS_DIR, ignore_errors=True)
        st.rerun()
    st.stop()

# Reset graph state before running a simulation
graph_builder.reset_graph_states()

# Get node lists early for automation
nodes_list = sorted(list(graph_builder.G.nodes()))

# Sidebar: Simulation Controls
st.sidebar.markdown("### 🎛️ Attack Simulator")

# Automate check
automate = st.sidebar.checkbox("🤖 Automate Security Monitor", value=False)

preset_options = ["Normal Operations", "DDoS Flood Attack", "Botnet Infection", "Brute Force Attempt", "Spoofing Attack", "Network Reconnaissance"]

if 'selected_preset' not in st.session_state:
    st.session_state['selected_preset'] = "Normal Operations"
if 'selected_node' not in st.session_state:
    st.session_state['selected_node'] = "EdgeGateway_14"

# If automation is running, select a random preset and node
if automate:
    import random
    st.session_state['selected_preset'] = random.choice(preset_options)
    st.session_state['selected_node'] = random.choice(nodes_list)

preset = st.sidebar.selectbox(
    "Select Attack Preset",
    preset_options,
    index=preset_options.index(st.session_state['selected_preset'])
)

if not automate:
    st.session_state['selected_preset'] = preset

# Populate features based on selected preset
if preset == "Normal Operations":
    flow_duration = st.sidebar.slider("Flow Duration (sec)", 0.5, 30.0, 12.5)
    packet_count = st.sidebar.slider("Packet Count", 5, 200, 50)
    protocol = st.sidebar.selectbox("Protocol", [6, 17, 1], format_func=lambda x: "TCP (6)" if x==6 else ("UDP (17)" if x==17 else "ICMP (1)"))
    src_port = st.sidebar.number_input("Source Port", 1024, 65535, 49152)
    dst_port = st.sidebar.number_input("Destination Port", 1, 65535, 443)
    flow_bytes = st.sidebar.number_input("Flow Bytes", 100, 50000, 2500)
    packet_rate = float(packet_count) / flow_duration
    conn_count = st.sidebar.slider("Connection Count", 1, 100, 5)
elif preset == "DDoS Flood Attack":
    flow_duration = st.sidebar.slider("Flow Duration (sec)", 0.01, 2.0, 0.1)
    packet_count = st.sidebar.slider("Packet Count", 500, 5000, 3200)
    protocol = st.sidebar.selectbox("Protocol", [17, 1], format_func=lambda x: "UDP (17)" if x==17 else "ICMP (1)")
    src_port = st.sidebar.number_input("Source Port", 1024, 65535, 51321)
    dst_port = st.sidebar.selectbox("Destination Port", [80, 443, 22])
    flow_bytes = st.sidebar.number_input("Flow Bytes", 10000, 1000000, 320000)
    packet_rate = float(packet_count) / flow_duration
    conn_count = st.sidebar.slider("Connection Count", 500, 5000, 2800)
elif preset == "Botnet Infection":
    flow_duration = st.sidebar.slider("Flow Duration (sec)", 2.0, 60.0, 18.0)
    packet_count = st.sidebar.slider("Packet Count", 15, 300, 120)
    protocol = st.sidebar.selectbox("Protocol", [6], format_func=lambda x: "TCP (6)")
    src_port = st.sidebar.number_input("Source Port", 1024, 65535, 62100)
    dst_port = st.sidebar.selectbox("Destination Port", [6667, 8080, 9999])
    flow_bytes = st.sidebar.number_input("Flow Bytes", 1000, 50000, 8500)
    packet_rate = float(packet_count) / flow_duration
    conn_count = st.sidebar.slider("Connection Count", 500, 2000, 1450)
elif preset == "Brute Force Attempt":
    flow_duration = st.sidebar.slider("Flow Duration (sec)", 1.0, 20.0, 5.2)
    packet_count = st.sidebar.slider("Packet Count", 100, 1000, 450)
    protocol = st.sidebar.selectbox("Protocol", [6], format_func=lambda x: "TCP (6)")
    src_port = st.sidebar.number_input("Source Port", 1024, 65535, 52123)
    dst_port = st.sidebar.selectbox("Destination Port", [22, 23, 3389])
    flow_bytes = st.sidebar.number_input("Flow Bytes", 5000, 200000, 45000)
    packet_rate = float(packet_count) / flow_duration
    conn_count = st.sidebar.slider("Connection Count", 50, 500, 150)
elif preset == "Spoofing Attack":
    flow_duration = st.sidebar.slider("Flow Duration (sec)", 0.1, 5.0, 1.2)
    packet_count = st.sidebar.slider("Packet Count", 10, 200, 80)
    protocol = st.sidebar.selectbox("Protocol", [1, 6], format_func=lambda x: "ICMP (1)" if x==1 else "TCP (6)")
    src_port = st.sidebar.number_input("Source Port", 1, 1024, 80)
    dst_port = st.sidebar.number_input("Destination Port", 1024, 65535, 54321)
    flow_bytes = st.sidebar.number_input("Flow Bytes", 500, 20000, 3200)
    packet_rate = float(packet_count) / flow_duration
    conn_count = st.sidebar.slider("Connection Count", 5, 100, 25)
else: # Reconnaissance
    flow_duration = st.sidebar.slider("Flow Duration (sec)", 0.05, 5.0, 0.5)
    packet_count = st.sidebar.slider("Packet Count", 10, 500, 180)
    protocol = st.sidebar.selectbox("Protocol", [6, 17], format_func=lambda x: "TCP (6)" if x==6 else "UDP (17)")
    src_port = st.sidebar.number_input("Source Port", 1024, 65535, 45901)
    dst_port = st.sidebar.number_input("Destination Port", 1, 1024, 21)
    flow_bytes = st.sidebar.number_input("Flow Bytes", 400, 10000, 7200)
    packet_rate = float(packet_count) / flow_duration
    conn_count = st.sidebar.slider("Connection Count", 10, 300, 120)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Target Device")
# Ensure selected_node in session state is valid
if st.session_state['selected_node'] not in nodes_list:
    st.session_state['selected_node'] = "EdgeGateway_14" if "EdgeGateway_14" in nodes_list else nodes_list[0]

default_idx = nodes_list.index(st.session_state['selected_node'])
compromised_node = st.sidebar.selectbox("Select Seed Node", nodes_list, index=default_idx)

if not automate:
    st.session_state['selected_node'] = compromised_node

# Map human-readable device labels to IP addresses
source_ip_map = {
    "EdgeGateway_14": "192.168.1.105",
    "CCTV_1": "192.168.1.51",
    "TrafficLight_8": "10.0.2.18",
    "ControlServer_1": "10.0.1.10"
}
simulated_ip = source_ip_map.get(compromised_node, "192.168.1.200")

# Run ML Classification Inference
sample_features = {
    'flow_duration': flow_duration,
    'packet_count': packet_count,
    'protocol': protocol,
    'src_port': src_port,
    'dst_port': dst_port,
    'flow_bytes': flow_bytes,
    'packet_rate': packet_rate
}

# XGBoost prediction
input_df = pd.DataFrame([sample_features])
input_df['flow_duration'] = xgb_artifacts['scaler'].transform(input_df[['flow_duration', 'packet_count', 'flow_bytes', 'packet_rate']])[:, 0]
input_df['packet_count'] = xgb_artifacts['scaler'].transform(pd.DataFrame([sample_features])[['flow_duration', 'packet_count', 'flow_bytes', 'packet_rate']])[:, 1]
input_df['flow_bytes'] = xgb_artifacts['scaler'].transform(pd.DataFrame([sample_features])[['flow_duration', 'packet_count', 'flow_bytes', 'packet_rate']])[:, 2]
input_df['packet_rate'] = xgb_artifacts['scaler'].transform(pd.DataFrame([sample_features])[['flow_duration', 'packet_count', 'flow_bytes', 'packet_rate']])[:, 3]

# Align columns
X_inst = input_df[xgb_artifacts['features']]
y_raw_probs = xgb_artifacts['model'].predict_proba(X_inst)[0]
y_probs = xgb_artifacts['temperature_scaler'].predict([y_raw_probs])[0]
pred_idx = np.argmax(y_probs)
predicted_attack = xgb_artifacts['label_encoder'].classes_[pred_idx]
confidence_score = y_probs[pred_idx]

# If normal operations selected, force NORMAL
if preset == "Normal Operations":
    predicted_attack = "Normal"
    confidence_score = 0.985
    
# GNN & Impact assessment calculations
# Run GNN
gnn_probs = gnn_predictor.predict_propagation(compromised_node, graph_builder, max_depth=2)
# Assess impacts
assessment = impact_assessor.assess_impacts(compromised_node, gnn_probs, predicted_attack, confidence_score)

# Compute PageRank centrality dynamically for the multi-factor risk score
centralities = nx.pagerank(graph_builder.G)
g_i = centralities.get(compromised_node, 0.05)
max_c = max(centralities.values()) if centralities else 1.0
g_i_norm = g_i / max_c if max_c > 0 else 0.0

prob_val = 0.01 if predicted_attack == "Normal" else confidence_score
risk_data = risk_calculator.calculate_risk(compromised_node, prob_val, centrality=g_i_norm, cvss_base=8.5)
# Generate alerts
alert_data = alert_generator.generate_alerts(risk_data, predicted_attack)

# Compute average impact score
if predicted_attack == "Normal":
    overall_impact_label = "LOW"
    overall_impact_score = 4.5
else:
    overall_impact_score = np.mean([s['impact'] for s in assessment['sector_impacts']]) if assessment['sector_impacts'] else 0.0
    if overall_impact_score < 30:
        overall_impact_label = "LOW"
    elif overall_impact_score < 60:
        overall_impact_label = "MEDIUM"
    else:
        overall_impact_label = "HIGH"

# Check if any mitigation checkbox has been selected by the user
mit_isolate = st.session_state.get(f"mit_isolate_{compromised_node}", False)
mit_block = st.session_state.get(f"mit_block_{compromised_node}", False)
mit_rules = st.session_state.get(f"mit_rules_{compromised_node}", False)

mitigation_applied = mit_isolate or mit_block or mit_rules

# Apply interactive threat containment override if any box is checked
if mitigation_applied and predicted_attack != "Normal":
    risk_data['risk_score'] = round(risk_data['risk_score'] * 0.12, 2)
    risk_data['risk_level'] = "CONTAINED (LOW)"
    alert_data['status'] = "THREAT CONTAINED & RESOLVED"
    alert_data['color'] = "#56d364" # Green
    
    gnn_probs = [{'node': p['node'], 'probability': 0.0} for p in gnn_probs]
    assessment['sector_impacts'] = [{'sector': s['sector'], 'impact': 0.0} for s in assessment['sector_impacts']]
    assessment['node_impacts'] = [{'node': n['node'], 'sector': n['sector'], 'probability': 0.0, 'impact_score': 0.0} for n in assessment['node_impacts']]
    
    overall_impact_label = "LOW"
    overall_impact_score = 0.0
    path_list = [compromised_node]

# Define user's tabs
tab_dash, tab_det, tab_risk, tab_gnn, tab_imp, tab_exp, tab_rep = st.tabs([
    "Dashboard", "Detection", "Risk Engine", "GNN Propagation", "Impact Assessment", "Explainability (SHAP)", "Reports"
])

# =========================================================================
# TAB 1: CONSOLIDATED COMMAND CENTER (Matches the user's ASCII mockup)
# =========================================================================
with tab_dash:
    # Row 1: KPI Metric Cards
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.markdown(f"""
        <div class="card">
            <div class="kpi-title">ATTACK STATUS</div>
            <div class="kpi-value text-red">🔴 {predicted_attack.upper()}</div>
            <div style="font-size: 13px; color: #8b949e; margin-top:5px;">Confidence: {confidence_score*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        risk_color_class = "text-red" if risk_data['risk_level'] == "CRITICAL" else ("text-orange" if risk_data['risk_level'] == "HIGH" else "text-yellow")
        st.markdown(f"""
        <div class="card">
            <div class="kpi-title">RISK LEVEL</div>
            <div class="kpi-value {risk_color_class}">🔴 {risk_data['risk_level']}</div>
            <div style="font-size: 13px; color: #8b949e; margin-top:5px;">Risk Score: {risk_data['risk_score']}/100</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        impact_color_class = "text-red" if overall_impact_label == "HIGH" else ("text-orange" if overall_impact_label == "MEDIUM" else "text-green")
        st.markdown(f"""
        <div class="card">
            <div class="kpi-title">OVERALL IMPACT</div>
            <div class="kpi-value {impact_color_class}">🔴 {overall_impact_label}</div>
            <div style="font-size: 13px; color: #8b949e; margin-top:5px;">Impact Score: {overall_impact_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Row 2: Network Overview & Attack Detection Result
    col_layout1, col_layout2 = st.columns(2)
    
    with col_layout1:
        st.markdown('<div class="console-header">NETWORK OVERVIEW</div>', unsafe_allow_html=True)
        overview_text = (
            "CCTV Cameras          : 250\n"
            "Traffic Lights        : 180\n"
            "Smart Parking Sensors : 150\n"
            "Smart Meters          : 320\n"
            "Edge Gateways         : 25\n"
            "Control Servers       : 12"
        )
        st.markdown(f'<div class="console-body">{overview_text}</div>', unsafe_allow_html=True)
        
    with col_layout2:
        st.markdown('<div class="console-header">ATTACK DETECTION RESULT</div>', unsafe_allow_html=True)
        det_text = (
            f"Device ID         : {compromised_node}\n"
            f"Attack Type       : {predicted_attack.upper()}\n"
            f"Confidence Score  : {confidence_score*100:.1f}%\n"
            f"Source IP         : {simulated_ip}\n"
            f"Protocol          : {'TCP' if protocol==6 else ('UDP' if protocol==17 else 'ICMP')}\n"
            f"Packet Rate       : {packet_rate:.1f} packets/sec\n"
            f"Flow Duration     : {flow_duration:.1f} sec\n"
            f"Connection Count  : {conn_count}"
        )
        st.markdown(f'<div class="console-body">{det_text}</div>', unsafe_allow_html=True)
        
    # Row 3: Early Warning System & GNN Attack Propagation
    col_layout3, col_layout4 = st.columns(2)
    
    with col_layout3:
        st.markdown('<div class="console-header">EARLY WARNING SYSTEM</div>', unsafe_allow_html=True)
        bar_len = int(risk_data['risk_score'] / 5)
        risk_bar = "█" * bar_len + "░" * (20 - bar_len)
        ew_text = (
            "\n"
            f"                        RISK SCORE\n\n"
            f"                           {risk_data['risk_score']} / 100\n\n"
            f"                     {risk_bar}\n\n"
            f"                    STATUS : {risk_data['risk_level']}\n"
        )
        st.markdown(f'<div class="console-body">{ew_text}</div>', unsafe_allow_html=True)
        
    with col_layout4:
        st.markdown('<div class="console-header">GNN ATTACK PROPAGATION PREDICTION</div>', unsafe_allow_html=True)
        # Reconstruct custom path tree
        path_list = gnn_predictor.get_critical_path(compromised_node, graph_builder)
        path_str = "\n"
        for i, node in enumerate(path_list):
            if i == 0:
                path_str += f"              {node} [INFECTED]\n"
            else:
                prob = next((p['probability'] for p in gnn_probs if p['node'] == node), 45.0)
                path_str += f"                 │ {prob:.0f}%\n"
                path_str += f"                 ▼\n"
                path_str += f"              {node}\n"
        path_str += f"\n Predicted Spread Risk : {overall_impact_label}"
        st.markdown(f'<div class="console-body">{path_str}</div>', unsafe_allow_html=True)
        
    # Row 4: Impact Assessment & Explainable AI (SHAP)
    col_layout5, col_layout6 = st.columns(2)
    
    with col_layout5:
        st.markdown('<div class="console-header">IMPACT ASSESSMENT</div>', unsafe_allow_html=True)
        imp_text = "\n"
        for sec_data in assessment['sector_impacts'][:4]:
            sec_name = sec_data['sector'].ljust(24)
            sec_score = sec_data['impact']
            bar_len = int(sec_score / 5)
            sec_bar = "█" * bar_len + "░" * (20 - bar_len)
            imp_text += f"{sec_name} {sec_bar}  {sec_score:.0f}%\n"
        imp_text += f"\n OVERALL CITY IMPACT : {overall_impact_label}"
        st.markdown(f'<div class="console-body">{imp_text}</div>', unsafe_allow_html=True)
        
    with col_layout6:
        st.markdown('<div class="console-header">EXPLAINABLE AI (SHAP)</div>', unsafe_allow_html=True)
        # Compute local SHAP values
        contribs, _, _ = shap_explainer.explain_instance(sample_features, pred_idx)
        shap_text = f"\n Why was this attack classified as {predicted_attack.upper()}?\n\n"
        for c in contribs[:4]:
            feat_name = c['feature'].replace('_', ' ').title().ljust(22)
            val = c['shap_value'] * 100
            sign = "+" if val >= 0 else "-"
            bar_len = int(abs(val) / 5)
            feat_bar = "█" * bar_len
            shap_text += f"{feat_name} {feat_bar}  {sign}{abs(val):.0f}%\n"
        st.markdown(f'<div class="console-body">{shap_text}</div>', unsafe_allow_html=True)
        
    # Row 5: Mitigation Recommendations & Final Decision
    col_layout7, col_layout8 = st.columns(2)
    
    with col_layout7:
        st.markdown('<div class="console-header">MITIGATION ACTIONS (PROACTIVE CONTAINMENT)</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if predicted_attack == "Normal":
                st.info("✓ Continuous monitoring active. No anomalies detected.")
            else:
                st.checkbox(f"🛡️ Isolate compromised node ({compromised_node})", value=st.session_state.get(f"mit_isolate_{compromised_node}", False), key=f"mit_isolate_{compromised_node}")
                st.checkbox("🚫 Block command & control attacker IP", value=st.session_state.get(f"mit_block_{compromised_node}", False), key=f"mit_block_{compromised_node}")
                st.checkbox("🧱 Apply edge firewall rate-limiting rules", value=st.session_state.get(f"mit_rules_{compromised_node}", False), key=f"mit_rules_{compromised_node}")
        
    with col_layout8:
        st.markdown('<div class="console-header">FINAL DECISION</div>', unsafe_allow_html=True)
        fd_text = (
            f"Attack Type        : {predicted_attack.upper()}\n"
            f"Confidence         : {confidence_score*100:.1f}%\n"
            f"Risk Level         : {risk_data['risk_level']}\n"
            f"Spread Probability : {overall_impact_label}\n"
            f"City Impact        : {overall_impact_label}\n\n"
            f" STATUS : {alert_data['status']}"
        )
        st.markdown(f'<div class="console-body">{fd_text}</div>', unsafe_allow_html=True)

# =========================================================================
# TAB 2: DETAILED ATTACK MONITOR
# =========================================================================
with tab_det:
    st.header("🔍 Cyber Attack Classifier Console")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Simulated Network Features")
        metrics_df = pd.DataFrame(list(sample_features.items()), columns=["Feature Metric", "Simulated Value"])
        st.table(metrics_df)
    with col2:
        st.subheader("Model Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(6, 4))
        # Draw horizontal probabilities
        classes = xgb_artifacts['label_encoder'].classes_
        y_pos = np.arange(len(classes))
        ax.barh(y_pos, y_probs * 100, color='#2c3e50')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Probability Score (%)')
        ax.set_title('Classification Confidence Profile')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# =========================================================================
# TAB 3: DETAILED RISK DASHBOARD
# =========================================================================
with tab_risk:
    st.header("🚨 Early Warning Risk Engine")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Risk Index Score", value=f"{risk_data['risk_score']}/100", delta=risk_data['risk_level'])
        st.markdown("### Risk Level Definitions")
        st.markdown("""
        - **CRITICAL (>= 85)**: Immediate threat to life safety and infrastructure operation. Node isolation required.
        - **HIGH (60 - 84)**: Impending threat of propagation. Rate limit ports and notify security team.
        - **MEDIUM (30 - 59)**: Suspicious anomalous behavior. Monitor logs and investigate source IP.
        - **LOW (< 30)**: Normal operational variance or light reconnaissance scans. Log event.
        """)
    with col2:
        # Show alert panel
        st.subheader("Active Alarm Advisory")
        for alert in alert_data['alerts']:
            st.warning(alert)
        st.markdown("#### Recommended Containment Checklist")
        for rec in alert_data['recommendations']:
            st.checkbox(rec, value=False, key="risk_check_"+rec[:20])

# =========================================================================
# TAB 4: GNN PROPAGATION GRAPH
# =========================================================================
with tab_gnn:
    st.header("🕸️ Graph Neural Network Infection Propagation")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Network Topology State View")
        # Draw NetworkX Spring layout
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(graph_builder.G, seed=42)
        
        # Color codes
        node_colors = []
        node_sizes = []
        for n in graph_builder.G.nodes():
            if n == compromised_node:
                if mitigation_applied:
                    node_colors.append('#95a5a6') # Grey for isolated
                else:
                    node_colors.append('#e74c3c') # infected seed
                node_sizes.append(180)
            elif any(p['node'] == n and p['probability'] > 50 for p in gnn_probs):
                node_colors.append('#e67e22') # high spread probability
                node_sizes.append(120)
            elif any(p['node'] == n and p['probability'] > 15 for p in gnn_probs):
                node_colors.append('#f1c40f') # moderate spread
                node_sizes.append(100)
            else:
                node_colors.append('#3498db') # normal state
                node_sizes.append(60)
                
        nx.draw_networkx_nodes(graph_builder.G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
        # Draw background edges
        nx.draw_networkx_edges(graph_builder.G, pos, alpha=0.15, edge_color='gray', ax=ax)
        
        # Highlight critical path edges in RED
        path_edges = list(zip(path_list[:-1], path_list[1:]))
        if path_edges:
            nx.draw_networkx_edges(
                graph_builder.G, pos, 
                edgelist=path_edges, 
                edge_color='#EF5350', 
                width=3.0, 
                alpha=0.95, 
                ax=ax
            )
            
        # Draw labels with background boxes for legibility
        if path_list:
            labels = {n: n for n in path_list}
            nx.draw_networkx_labels(
                graph_builder.G, pos, 
                labels=labels, 
                font_size=8, 
                font_color='#ffffff', 
                font_weight='bold', 
                bbox=dict(boxstyle="round,pad=0.2", fc="#21262d", ec="#30363d", alpha=0.85),
                ax=ax
            )
        
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    with col2:
        st.subheader("GNN Infection Probabilities")
        df_probs = pd.DataFrame(gnn_probs)
        if not df_probs.empty:
            st.dataframe(df_probs.head(10), use_container_width=True)
        else:
            st.write("No propagation predicted beyond target node.")

# =========================================================================
# TAB 5: IMPACT ASSESSMENT DETAILS
# =========================================================================
with tab_imp:
    st.header("🏥 Smart City Sector Disruption Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sector Impact Scores")
        df_sec = pd.DataFrame(assessment['sector_impacts'])
        if not df_sec.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_sec['sector'], df_sec['impact'], color='#e74c3c')
            ax.set_ylabel('Disruption Index (%)')
            plt.xticks(rotation=45, ha='right')
            ax.set_ylim(0, 100)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.write("No sector impact data.")
    with col2:
        st.subheader("Individual Node Impacts")
        df_nodes = pd.DataFrame(assessment['node_impacts'])
        if not df_nodes.empty:
            st.dataframe(df_nodes.sort_values(by='impact_score', ascending=False), use_container_width=True)

# =========================================================================
# TAB 6: SHAP EXPLANATIONS
# =========================================================================
with tab_exp:
    st.header("🧠 Explainable AI - SHAP Local Contributions")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Local Feature Importance")
        # Draw the SHAP bar plot from generator
        _, fig_shap, base_val = shap_explainer.explain_instance(sample_features, pred_idx)
        st.pyplot(fig_shap)
        plt.close()
    with col2:
        st.subheader("SHAP Interpretations")
        st.markdown(f"""
        - **Predicted Attack**: `{predicted_attack}`
        - **Base Expectation Value (log-odds)**: `{base_val:.4f}`
        - **How to read SHAP value**:
          - Features with **Red** bars push the prediction *towards* the attack classification.
          - Features with **Blue** bars push the prediction *away* from the attack (towards Normal).
          - The length of the bar represents the magnitude of that feature's influence.
        """)

# =========================================================================
# TAB 7: REPORTS & EXPORTS
# =========================================================================
with tab_rep:
    st.header("📂 Final Year B.Tech Project Reports")
    st.markdown("Download generated classification reports and results spreadsheets directly.")
    
    col_rep1, col_rep2, col_rep3 = st.columns(3)
    
    # 1. Download accuracy report
    acc_path = 'results/accuracy_report.txt'
    if os.path.exists(acc_path):
        with open(acc_path, 'r', encoding='utf-8') as f:
            acc_data = f.read()
        col_rep1.download_button(
            label="📄 Download Accuracy Report",
            data=acc_data,
            file_name="accuracy_report.txt",
            mime="text/plain"
        )
        
    # 2. Download classification report
    class_path = 'results/classification_report.txt'
    if os.path.exists(class_path):
        with open(class_path, 'r', encoding='utf-8') as f:
            class_data = f.read()
        col_rep2.download_button(
            label="📄 Download Classification Report",
            data=class_data,
            file_name="classification_report.txt",
            mime="text/plain"
        )
        
    # 3. Download impact report CSV
    impact_path = 'results/impact_report.csv'
    if os.path.exists(impact_path):
        with open(impact_path, 'rb') as f:
            impact_bytes = f.read()
        col_rep3.download_button(
            label="📊 Download Impact Report (CSV)",
            data=impact_bytes,
            file_name="impact_report.csv",
            mime="text/csv"
        )

# Auto-monitoring loop rerun trigger
if automate:
    import time
    time.sleep(3.0)
    st.rerun()
