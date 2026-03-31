import streamlit as st
import joblib
import numpy as np
import json
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --card: #111827;
    --border: #1e2d45;
    --accent: #00d4ff;
    --danger: #ff4d6d;
    --success: #00e5a0;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero h1 {
    font-family: 'Courier New', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    margin-bottom: 0.4rem;
    text-shadow: 0 0 30px rgba(0,212,255,0.3);
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Cards */
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Result boxes */
.result-fraud {
    background: linear-gradient(135deg, #1a0a10, #2d0d1a);
    border: 1px solid var(--danger);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(255,77,109,0.15);
}
.result-legit {
    background: linear-gradient(135deg, #0a1a14, #0d2d20);
    border: 1px solid var(--success);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,229,160,0.15);
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-fraud .result-title { color: var(--danger); }
.result-legit .result-title { color: var(--success); }

.result-sub {
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 1rem;
}
.prob-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
}
.result-fraud .prob-badge {
    background: rgba(255,77,109,0.15);
    color: var(--danger);
    border: 1px solid rgba(255,77,109,0.3);
}
.result-legit .prob-badge {
    background: rgba(0,229,160,0.1);
    color: var(--success);
    border: 1px solid rgba(0,229,160,0.25);
}

/* Input overrides */
div[data-baseweb="input"] input,
div[data-baseweb="select"] {
    background: #1a2235 !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0070f3, #00b4d8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
    margin-top: 1rem;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,112,243,0.4) !important;
}

/* Slider label */
label { color: var(--muted) !important; font-size: 0.82rem !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }

/* Metric pill */
.metric-row {
    display: flex;
    gap: 0.8rem;
    justify-content: center;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.metric-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.78rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
}
.metric-pill span { color: var(--text); font-weight: 600; }

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Assets ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'fraud_model_lr.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'features.json')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')

@st.cache_resource
def load_assets():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        features = json.load(f)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return model, scaler, features, config

try:
    model, scaler, top_features, config = load_assets()
    THRESHOLD = float(config.get('threshold', 0.3))
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error("Failed to load model assets")
    st.exception(e)
    st.info("Make sure fraud_model_lr.pkl, scaler.pkl, features.json, and config.json are in the same folder as app.py.")
    st.warning(f"Current working dir: {os.getcwd()}")
    st.warning(f"Asset paths: {MODEL_PATH}, {SCALER_PATH}, {FEATURES_PATH}, {CONFIG_PATH}")
    st.stop()


# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ FraudShield</h1>
    <p>Credit Card Fraud Detection System &nbsp;·&nbsp; ML InnovateX Hackathon</p>
</div>
""", unsafe_allow_html=True)


# ─── Model Info Bar ────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-pill">Model <span>Logistic Regression</span></div>
    <div class="metric-pill">AUC <span>0.9736</span></div>
    <div class="metric-pill">Threshold <span>0.30</span></div>
    <div class="metric-pill">Features <span>Top 15 (PCA)</span></div>
</div>
<br>
""", unsafe_allow_html=True)


# ─── Input Form ───────────────────────────────────────────────────────────────
st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Transaction Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input(
        "Transaction Amount (₹ / $)",
        min_value=0.0, max_value=20000.0,
        value=120.0, step=0.5,
        help="Actual transaction amount"
    )
with col2:
    time_val = st.number_input(
        "Time (seconds from first txn)",
        min_value=0, max_value=172800,
        value=1500,
        help="Time elapsed since first transaction in dataset"
    )

st.markdown("</div>", unsafe_allow_html=True)


# ─── V Feature Inputs (only features the model actually uses) ─────────────────
v_features = [f for f in top_features if f.startswith('V')]

st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">PCA-Transformed Features (V1–V28)</div>', unsafe_allow_html=True)
st.caption("These are anonymized PCA components from the original transaction data. Default = 0.0 for typical transactions.")

v_inputs = {}
cols = st.columns(3)
for i, feat in enumerate(v_features):
    with cols[i % 3]:
        v_inputs[feat] = st.number_input(
            feat,
            min_value=-30.0, max_value=35.0,
            value=0.0, step=0.01,
            format="%.3f",
            key=feat
        )

st.markdown("</div>", unsafe_allow_html=True)


# ─── Quick Test Presets ────────────────────────────────────────────────────────
st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Quick Test Presets</div>', unsafe_allow_html=True)
st.caption("Load known transaction profiles to test the model.")

pc1, pc2, pc3 = st.columns(3)

with pc1:
    typical = st.button("📦 Typical Purchase")
with pc2:
    suspicious = st.button("⚠️ Suspicious Pattern")
with pc3:
    clear_btn = st.button("🔄 Reset All")

st.markdown("</div>", unsafe_allow_html=True)


# ─── Predict Button ────────────────────────────────────────────────────────────
predict_btn = st.button("🔍  ANALYZE TRANSACTION")


# ─── Prediction Logic ──────────────────────────────────────────────────────────
def build_input(amount_val, time_input, v_vals):
    """Build input row using same scaling as training."""
    # Approximate scaling (replace with actual scaler stats if saved separately)
    scaled_amount = (amount_val - 65.0) / 213.7
    scaled_time   = (time_input - 1638.0) / 1016.6

    row = {}
    for feat in top_features:
        if feat == 'scaled_amount':
            row[feat] = scaled_amount
        elif feat == 'scaled_time':
            row[feat] = scaled_time
        elif feat in v_vals:
            row[feat] = v_vals[feat]
        else:
            row[feat] = 0.0

    return np.array([[row[f] for f in top_features]])


def run_prediction(amount_val, time_input, v_vals):
    input_row = build_input(amount_val, time_input, v_vals)
    proba = model.predict_proba(input_row)[0][1]
    pred  = int(proba >= THRESHOLD)
    return pred, proba


# Handle preset buttons
if typical:
    st.info("Typical purchase profile loaded — all V features at 0.0, amount ₹120, time 1500s. Click Analyze.")

if suspicious:
    st.warning("Suspicious profile: high-value transaction with anomalous V4, V11, V14 values loaded.")
    # These are rough fraud-leaning values based on dataset EDA
    v_inputs['V4']  = 4.5  if 'V4'  in v_inputs else 0.0
    v_inputs['V11'] = -3.2 if 'V11' in v_inputs else 0.0
    v_inputs['V14'] = -8.5 if 'V14' in v_inputs else 0.0

if clear_btn:
    st.rerun()


# Run prediction
if predict_btn:
    with st.spinner("Analyzing transaction..."):
        pred, proba = run_prediction(amount, time_val, v_inputs)

    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"""
        <div class="result-fraud">
            <div class="result-title">⚠️ FRAUDULENT</div>
            <div class="result-sub">This transaction shows fraud indicators above threshold ({THRESHOLD})</div>
            <div class="prob-badge">Fraud Probability: {proba:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.error("🚨 Recommended Action: Block transaction and notify cardholder immediately.")

    else:
        st.markdown(f"""
        <div class="result-legit">
            <div class="result-title">✅ LEGITIMATE</div>
            <div class="result-sub">Transaction appears normal. No fraud indicators detected.</div>
            <div class="prob-badge">Fraud Probability: {proba:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("✔️ Transaction cleared. Safe to proceed.")

    # Probability gauge
    st.markdown("**Fraud Risk Score**")
    st.progress(float(proba))
    st.caption(f"Raw probability: `{proba:.6f}` — Decision threshold: `{THRESHOLD}`")


# ─── Model Info Expander ───────────────────────────────────────────────────────
with st.expander("ℹ️ About this Model"):
    st.markdown("""
    **Dataset:** Credit Card Fraud Detection (Kaggle — ULB)  
    **Records:** 3,972 transactions | 2 fraud cases  
    **Best Model:** Logistic Regression (`class_weight='balanced'`, `C=0.1`)  
    **ANN ROC-AUC:** 1.000 (test) | **LR ROC-AUC:** 0.9736  
    **Imbalance Handling:** RandomOverSampler + class weights  
    **Feature Selection:** Top 15 features via Random Forest importance  
    **Threshold:** 0.30 (lowered from 0.5 to improve fraud recall)

    > ⚠️ This model was trained on a very small fraud sample (2 cases).
    > Results demonstrate methodology. Production use requires more fraud data.
    """)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ML InnovateX Hackathon &nbsp;·&nbsp; Credit Card Fraud Detection &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)