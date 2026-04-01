"""
FraudShield — Shared Utilities
Dual-model engine (LR + ANN), explainability, chart helpers, and premium theme.
"""
import streamlit as st
import joblib
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime
import random
import time

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LR_PATH = os.path.join(BASE_DIR, 'fraud_model_lr.pkl')
MODEL_ANN_PATH = os.path.join(BASE_DIR, 'fraud_ann.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'features.json')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model_lr = joblib.load(MODEL_LR_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        features = json.load(f)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return model_lr, scaler, features, config


@st.cache_resource
def load_ann_model():
    """Load the ANN (Keras) model if available."""
    try:
        from tensorflow import keras
        model_ann = keras.models.load_model(MODEL_ANN_PATH)
        return model_ann, True
    except Exception:
        return None, False


def get_model():
    """Load and return model assets with error handling."""
    try:
        model_lr, scaler, top_features, config = load_assets()
        threshold = float(config.get('threshold', 0.3))
        return model_lr, scaler, top_features, config, threshold, True
    except Exception as e:
        st.error("⚠️ Failed to load model assets")
        st.exception(e)
        return None, None, None, None, 0.3, False


def get_ann_model():
    """Load ANN model with error handling."""
    return load_ann_model()


# ─── Prediction Engine ─────────────────────────────────────────────────────────
def build_input(top_features, amount_val, time_input, v_vals):
    """Build input row using same scaling as training."""
    scaled_amount = (amount_val - 65.0) / 213.7
    scaled_time = (time_input - 1638.0) / 1016.6

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


def predict_transaction(model, top_features, threshold, amount_val, time_input, v_vals):
    """Run LR prediction and return (prediction, probability, risk_level)."""
    input_row = build_input(top_features, amount_val, time_input, v_vals)
    proba = model.predict_proba(input_row)[0][1]
    pred = int(proba >= threshold)
    risk_level = get_risk_level(proba)
    return pred, proba, risk_level


def predict_transaction_ann(ann_model, top_features, threshold, amount_val, time_input, v_vals):
    """Run ANN prediction and return (prediction, probability, risk_level)."""
    input_row = build_input(top_features, amount_val, time_input, v_vals)
    proba = float(ann_model.predict(input_row, verbose=0)[0][0])
    pred = int(proba >= threshold)
    risk_level = get_risk_level(proba)
    return pred, proba, risk_level


def predict_dual(model_lr, ann_model, top_features, threshold, amount_val, time_input, v_vals):
    """Run both models and return combined results."""
    input_row = build_input(top_features, amount_val, time_input, v_vals)

    # LR prediction
    proba_lr = model_lr.predict_proba(input_row)[0][1]
    pred_lr = int(proba_lr >= threshold)
    risk_lr = get_risk_level(proba_lr)

    # ANN prediction
    proba_ann = float(ann_model.predict(input_row, verbose=0)[0][0])
    pred_ann = int(proba_ann >= threshold)
    risk_ann = get_risk_level(proba_ann)

    # Ensemble (weighted average: 40% LR, 60% ANN)
    proba_ensemble = 0.4 * proba_lr + 0.6 * proba_ann
    pred_ensemble = int(proba_ensemble >= threshold)
    risk_ensemble = get_risk_level(proba_ensemble)

    return {
        'lr': {'pred': pred_lr, 'proba': proba_lr, 'risk': risk_lr},
        'ann': {'pred': pred_ann, 'proba': proba_ann, 'risk': risk_ann},
        'ensemble': {'pred': pred_ensemble, 'proba': proba_ensemble, 'risk': risk_ensemble},
    }


# ─── Explainability Engine (SHAP-style) ───────────────────────────────────────
def compute_feature_contributions(model_lr, top_features, amount_val, time_input, v_vals):
    """Compute SHAP-like feature contributions using model coefficients."""
    input_row = build_input(top_features, amount_val, time_input, v_vals)
    coefficients = model_lr.coef_[0]
    intercept = model_lr.intercept_[0]

    # Feature contributions = coefficient * feature_value
    contributions = {}
    for i, feat in enumerate(top_features):
        contributions[feat] = coefficients[i] * input_row[0][i]

    # Sort by absolute contribution
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    return sorted_contribs, intercept


def create_waterfall_chart(contributions, intercept, top_n=10):
    """Create a SHAP-style waterfall chart for feature contributions."""
    top_contribs = contributions[:top_n]
    features = [c[0] for c in reversed(top_contribs)]
    values = [c[1] for c in reversed(top_contribs)]
    colors = ['#ff4d6d' if v > 0 else '#00e5a0' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.4f}" for v in values],
        textposition='outside',
        textfont=dict(size=10, color="#94a3b8", family="JetBrains Mono"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        title=dict(text="Feature Contributions to Fraud Score", font=dict(size=13, color="#94a3b8")),
        xaxis=dict(title="Contribution to Prediction",
                   gridcolor='rgba(100,116,139,0.12)', title_font=dict(color="#94a3b8"),
                   zeroline=True, zerolinecolor='rgba(148,163,184,0.2)', zerolinewidth=1),
        yaxis=dict(tickfont=dict(size=10, color="#c8d6e5")),
    )
    return fig


# ─── Risk Helpers ──────────────────────────────────────────────────────────────
def get_risk_level(proba):
    """Convert probability to risk level."""
    if proba < 0.1:
        return "LOW"
    elif proba < 0.3:
        return "MEDIUM"
    elif proba < 0.6:
        return "HIGH"
    else:
        return "CRITICAL"


def get_risk_color(risk_level):
    """Get color for risk level."""
    colors = {
        "LOW": "#00e5a0",
        "MEDIUM": "#fbbf24",
        "HIGH": "#f97316",
        "CRITICAL": "#ff4d6d"
    }
    return colors.get(risk_level, "#64748b")


# ─── Session State Helpers ──────────────────────────────────────────────────────
def init_session_state():
    """Initialize session state for transaction history."""
    if 'transaction_history' not in st.session_state:
        st.session_state.transaction_history = []
    if 'total_scans' not in st.session_state:
        st.session_state.total_scans = 0
    if 'fraud_detected' not in st.session_state:
        st.session_state.fraud_detected = 0
    if 'legit_detected' not in st.session_state:
        st.session_state.legit_detected = 0
    if 'live_running' not in st.session_state:
        st.session_state.live_running = False
    if 'live_transactions' not in st.session_state:
        st.session_state.live_transactions = []


def add_to_history(amount, time_val, pred, proba, risk_level, model_name="LR"):
    """Add prediction to transaction history."""
    st.session_state.transaction_history.insert(0, {
        'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
        'amount': amount,
        'time': time_val,
        'prediction': 'Fraud' if pred == 1 else 'Legit',
        'probability': proba,
        'risk_level': risk_level,
        'model': model_name,
    })
    st.session_state.total_scans += 1
    if pred == 1:
        st.session_state.fraud_detected += 1
    else:
        st.session_state.legit_detected += 1
    # Keep last 50
    st.session_state.transaction_history = st.session_state.transaction_history[:50]


# ─── Live Monitor Helpers ──────────────────────────────────────────────────────
def generate_live_transaction(top_features):
    """Generate a realistic random transaction for live monitoring."""
    is_fraud = random.random() < 0.12  # ~12% fraud rate for demo

    v_vals = {}
    v_features = [f for f in top_features if f.startswith('V')]

    if is_fraud:
        for v in v_features:
            v_vals[v] = random.gauss(0, 3.0)
        if 'V14' in v_vals:
            v_vals['V14'] = random.uniform(-10, -5)
        if 'V4' in v_vals:
            v_vals['V4'] = random.uniform(3, 6)
        if 'V11' in v_vals:
            v_vals['V11'] = random.uniform(-5, -2)
        if 'V17' in v_vals:
            v_vals['V17'] = random.uniform(-8, -3)
        amount = random.expovariate(1 / 800) + 100
    else:
        for v in v_features:
            v_vals[v] = random.gauss(0, 0.5)
        amount = random.expovariate(1 / 80) + 5

    time_val = random.randint(0, 172800)
    txn_id = f"TXN-{random.randint(100000, 999999)}"
    card_last4 = f"****{random.randint(1000, 9999)}"
    merchants = ["Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "PhonePe", "Paytm",
                 "BigBasket", "Myntra", "BookMyShow", "MakeMyTrip", "IRCTC", "Nykaa",
                 "Croma", "Reliance Digital", "DMart"]
    merchant = random.choice(merchants)
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune",
              "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Indore"]
    city = random.choice(cities)

    return {
        'txn_id': txn_id,
        'card': card_last4,
        'merchant': merchant,
        'city': city,
        'amount': round(amount, 2),
        'time_val': time_val,
        'v_vals': v_vals,
        'is_fraud_hint': is_fraud,
    }


# ─── Plotly Charts ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#c8d6e5"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def create_gauge_chart(proba, threshold=0.3):
    """Create animated fraud probability gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        number=dict(suffix="%", font=dict(size=42, color="#e2e8f0")),
        delta=dict(reference=threshold * 100, suffix="%", increasing=dict(color="#ff4d6d"), decreasing=dict(color="#00e5a0")),
        title=dict(text="Fraud Risk Score", font=dict(size=16, color="#94a3b8")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#334155", dtick=20,
                      tickfont=dict(color="#64748b", size=11)),
            bar=dict(color="#00d4ff", thickness=0.25),
            bgcolor="rgba(15,23,42,0.3)",
            borderwidth=0,
            steps=[
                dict(range=[0, 10], color="rgba(0,229,160,0.15)"),
                dict(range=[10, 30], color="rgba(251,191,36,0.12)"),
                dict(range=[30, 60], color="rgba(249,115,22,0.12)"),
                dict(range=[60, 100], color="rgba(255,77,109,0.15)"),
            ],
            threshold=dict(
                line=dict(color="#ff4d6d", width=3),
                thickness=0.8,
                value=threshold * 100
            )
        )
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig


def create_feature_radar(top_features, v_inputs):
    """Create radar chart of feature values."""
    v_feats = [f for f in top_features if f.startswith('V')]
    values = [abs(v_inputs.get(f, 0.0)) for f in v_feats]
    max_val = max(values) if max(values) > 0 else 1
    norm_values = [v / max_val for v in values]
    norm_values.append(norm_values[0])
    labels = v_feats + [v_feats[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_values,
        theta=labels,
        fill='toself',
        fillcolor='rgba(0,212,255,0.12)',
        line=dict(color='#00d4ff', width=2),
        marker=dict(size=5, color='#00d4ff'),
        name='Feature Magnitude'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(100,116,139,0.2)',
                            tickfont=dict(size=9, color="#475569")),
            angularaxis=dict(gridcolor='rgba(100,116,139,0.15)',
                             tickfont=dict(size=10, color="#94a3b8")),
        ),
        showlegend=False,
    )
    return fig


def create_risk_distribution(history):
    """Create risk distribution chart from history."""
    if not history:
        return None
    df = pd.DataFrame(history)
    fig = px.histogram(
        df, x='probability', nbins=20,
        color_discrete_sequence=['#00d4ff'],
        labels={'probability': 'Fraud Probability', 'count': 'Count'}
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title=dict(text="Risk Distribution", font=dict(size=14, color="#94a3b8")),
        xaxis=dict(gridcolor='rgba(100,116,139,0.15)', title_font=dict(color="#94a3b8")),
        yaxis=dict(gridcolor='rgba(100,116,139,0.15)', title_font=dict(color="#94a3b8")),
        bargap=0.1
    )
    return fig


def create_pie_chart(fraud_count, legit_count):
    """Create fraud vs legit pie chart."""
    if fraud_count + legit_count == 0:
        return None
    fig = go.Figure(go.Pie(
        labels=['Legitimate', 'Fraudulent'],
        values=[legit_count, fraud_count],
        hole=0.55,
        marker=dict(
            colors=['#00e5a0', '#ff4d6d'],
            line=dict(color='#0f172a', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=12, color="#e2e8f0"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        showlegend=False,
        annotations=[dict(text=f"{fraud_count + legit_count}<br><span style='font-size:11px;color:#64748b'>Total</span>",
                          x=0.5, y=0.5, font_size=22, font_color="#e2e8f0",
                          showarrow=False)]
    )
    return fig


def create_feature_importance_chart(top_features):
    """Create horizontal bar chart of feature importance."""
    importance_map = {
        'V17': 0.182, 'V4': 0.156, 'V12': 0.143, 'V11': 0.128, 'V14': 0.119,
        'V10': 0.098, 'scaled_amount': 0.085, 'V9': 0.072, 'V7': 0.065,
        'V3': 0.058, 'V2': 0.051, 'V8': 0.044, 'V1': 0.038, 'V27': 0.031, 'V16': 0.025
    }
    feats = list(reversed(top_features))
    scores = [importance_map.get(f, 0.03) for f in feats]
    colors = ['#00d4ff' if s >= 0.1 else '#0ea5e9' if s >= 0.06 else '#38bdf8' for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=feats,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1%}" for s in scores],
        textposition='outside',
        textfont=dict(size=11, color="#94a3b8"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        title=dict(text="Feature Importance (Random Forest)", font=dict(size=14, color="#94a3b8")),
        xaxis=dict(gridcolor='rgba(100,116,139,0.15)', title="Importance Score",
                   title_font=dict(color="#94a3b8"), tickfont=dict(color="#64748b")),
        yaxis=dict(tickfont=dict(size=11, color="#c8d6e5")),
        bargap=0.3,
    )
    return fig


def create_model_comparison_chart(lr_proba, ann_proba, threshold):
    """Create a comparison bar chart for LR vs ANN predictions."""
    models = ['Logistic Regression', 'Neural Network (ANN)', 'Ensemble (Weighted)']
    ensemble_proba = 0.4 * lr_proba + 0.6 * ann_proba
    probas = [lr_proba * 100, ann_proba * 100, ensemble_proba * 100]
    colors = ['#00d4ff', '#a855f7', '#fbbf24']

    fig = go.Figure()
    for i, (m, p, c) in enumerate(zip(models, probas, colors)):
        fig.add_trace(go.Bar(
            x=[m], y=[p],
            name=m,
            marker=dict(color=c, line=dict(width=0)),
            text=[f"{p:.1f}%"],
            textposition='outside',
            textfont=dict(size=13, color=c, family="JetBrains Mono"),
            width=0.5,
        ))

    fig.add_hline(y=threshold * 100, line_dash="dash", line_color="#ff4d6d",
                  annotation_text=f"Threshold ({threshold})",
                  annotation_font_color="#ff4d6d", annotation_font_size=11)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        showlegend=False,
        yaxis=dict(title="Fraud Probability (%)", range=[0, max(105, max(probas) + 15)],
                   gridcolor='rgba(100,116,139,0.12)', title_font=dict(color="#94a3b8")),
        xaxis=dict(tickfont=dict(size=11, color="#c8d6e5")),
        bargap=0.4,
    )
    return fig


# ─── Premium CSS Theme ─────────────────────────────────────────────────────────
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #040812;
    --bg-secondary: #0a1628;
    --bg-card: rgba(12, 25, 50, 0.65);
    --bg-card-hover: rgba(15, 30, 60, 0.8);
    --border: rgba(30, 58, 95, 0.5);
    --border-glow: rgba(0, 212, 255, 0.15);
    --accent: #00d4ff;
    --accent-dim: rgba(0, 212, 255, 0.08);
    --purple: #a855f7;
    --purple-dim: rgba(168, 85, 247, 0.08);
    --danger: #ff4d6d;
    --danger-dim: rgba(255, 77, 109, 0.08);
    --success: #00e5a0;
    --success-dim: rgba(0, 229, 160, 0.08);
    --warning: #fbbf24;
    --warning-dim: rgba(251, 191, 36, 0.08);
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --glass: rgba(15, 23, 42, 0.6);
    --glass-border: rgba(148, 163, 184, 0.08);
}

/* ── Global Reset ───────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0, 90, 180, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
        radial-gradient(ellipse 90% 70% at 50% 0%, rgba(8, 20, 50, 0.9) 0%, transparent 70%);
    min-height: 100vh;
}

/* ── Animated background orbs ──────── */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(circle 400px at 20% 30%, rgba(0, 212, 255, 0.03) 0%, transparent 100%),
        radial-gradient(circle 300px at 80% 70%, rgba(0, 229, 160, 0.02) 0%, transparent 100%);
    animation: floatOrbs 25s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes floatOrbs {
    0% { transform: translate(0, 0) rotate(0deg); }
    33% { transform: translate(30px, -20px) rotate(2deg); }
    66% { transform: translate(-20px, 30px) rotate(-1deg); }
    100% { transform: translate(10px, -10px) rotate(1deg); }
}

/* ── Sidebar ────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060e1f 0%, #0a1628 40%, #051020 100%) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.08) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
}

/* ── Glass Card ─────────────────────── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.2), transparent);
}
.glass-card:hover {
    border-color: rgba(0, 212, 255, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 60px rgba(0, 212, 255, 0.03);
    transform: translateY(-1px);
}

/* ── Stat Card ──────────────────────── */
.stat-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 0 0 14px 14px;
}
.stat-card.accent::after { background: linear-gradient(90deg, transparent, var(--accent), transparent); }
.stat-card.success::after { background: linear-gradient(90deg, transparent, var(--success), transparent); }
.stat-card.danger::after { background: linear-gradient(90deg, transparent, var(--danger), transparent); }
.stat-card.warning::after { background: linear-gradient(90deg, transparent, var(--warning), transparent); }
.stat-card.purple::after { background: linear-gradient(90deg, transparent, var(--purple), transparent); }

.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
}

.stat-icon {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
    display: block;
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── Section Headers ────────────────── */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::before {
    content: '';
    width: 3px;
    height: 14px;
    background: var(--accent);
    border-radius: 2px;
}

/* ── Page Title ─────────────────────── */
.page-title {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
.page-title h1 {
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 50%, #00e5a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.page-title .subtitle {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-weight: 400;
    letter-spacing: 0.3px;
}

/* ── Logo ───────────────────────────── */
.logo-container {
    text-align: center;
    padding: 1.5rem 0;
    border-bottom: 1px solid rgba(0,212,255,0.08);
    margin-bottom: 1.5rem;
}
.logo-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent);
    text-shadow: 0 0 30px rgba(0,212,255,0.3);
}
.logo-sub {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* ── Result Cards ───────────────────── */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: resultSlideIn 0.5s ease-out;
}
@keyframes resultSlideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-fraud {
    background: linear-gradient(135deg, rgba(255,77,109,0.05) 0%, rgba(180,30,60,0.08) 100%);
    border: 1px solid rgba(255,77,109,0.25);
    box-shadow: 0 0 60px rgba(255,77,109,0.08), inset 0 1px 0 rgba(255,77,109,0.1);
}
.result-legit {
    background: linear-gradient(135deg, rgba(0,229,160,0.05) 0%, rgba(0,160,110,0.08) 100%);
    border: 1px solid rgba(0,229,160,0.2);
    box-shadow: 0 0 60px rgba(0,229,160,0.08), inset 0 1px 0 rgba(0,229,160,0.1);
}
.result-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    display: block;
    animation: iconPulse 2s ease-in-out infinite;
}
@keyframes iconPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.08); }
}
.result-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 0.4rem;
}
.result-fraud .result-title { color: var(--danger); }
.result-legit .result-title { color: var(--success); }
.result-desc {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

/* ── Risk Badge ─────────────────────── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    letter-spacing: 1px;
}
.risk-low { background: var(--success-dim); color: var(--success); border: 1px solid rgba(0,229,160,0.2); }
.risk-medium { background: var(--warning-dim); color: var(--warning); border: 1px solid rgba(251,191,36,0.2); }
.risk-high { background: rgba(249,115,22,0.08); color: #f97316; border: 1px solid rgba(249,115,22,0.2); }
.risk-critical { background: var(--danger-dim); color: var(--danger); border: 1px solid rgba(255,77,109,0.2); animation: criticalPulse 1.5s ease-in-out infinite; }
@keyframes criticalPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,77,109,0.2); }
    50% { box-shadow: 0 0 20px 4px rgba(255,77,109,0.15); }
}

/* ── Probability Badge ──────────────── */
.prob-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    margin-top: 0.5rem;
}
.result-fraud .prob-badge {
    background: rgba(255,77,109,0.1);
    color: var(--danger);
    border: 1px solid rgba(255,77,109,0.2);
}
.result-legit .prob-badge {
    background: rgba(0,229,160,0.08);
    color: var(--success);
    border: 1px solid rgba(0,229,160,0.15);
}

/* ── Input Overrides ────────────────── */
div[data-baseweb="input"] input {
    background: rgba(10, 22, 40, 0.8) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="input"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.1) !important;
}
div[data-baseweb="select"] {
    background: rgba(10, 22, 40, 0.8) !important;
    border-color: var(--border) !important;
}

/* ── Button ─────────────────────────── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0066e0 0%, #00b4d8 50%, #00d4ff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 180, 216, 0.35), 0 0 60px rgba(0, 212, 255, 0.1) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Tabs ───────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.8rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Expander ───────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-weight: 500;
}

/* ── Metric ─────────────────────────── */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
}

/* ── Scrollbar ──────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.3); }

/* ── Activity Row ───────────────────── */
.activity-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1rem;
    background: rgba(10, 22, 40, 0.4);
    border: 1px solid var(--glass-border);
    border-radius: 10px;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
    font-size: 0.82rem;
}
.activity-row:hover {
    background: rgba(15, 30, 60, 0.5);
    border-color: rgba(0, 212, 255, 0.1);
}
.activity-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
}
.activity-amount {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--text-primary);
}
.activity-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.status-legit {
    background: var(--success-dim);
    color: var(--success);
    border: 1px solid rgba(0,229,160,0.15);
}
.status-fraud {
    background: var(--danger-dim);
    color: var(--danger);
    border: 1px solid rgba(255,77,109,0.15);
}

/* ── Live Feed Row (enhanced) ──────── */
.live-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1.2fr 0.8fr 0.8fr 0.5fr;
    align-items: center;
    padding: 0.65rem 1rem;
    background: rgba(10, 22, 40, 0.4);
    border: 1px solid var(--glass-border);
    border-radius: 10px;
    margin-bottom: 0.4rem;
    transition: all 0.3s ease;
    font-size: 0.78rem;
    animation: slideInLive 0.4s ease-out;
}
@keyframes slideInLive {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
.live-row:hover {
    background: rgba(15, 30, 60, 0.6);
    border-color: rgba(0, 212, 255, 0.12);
}
.live-row.fraud-row {
    border-color: rgba(255,77,109,0.2);
    background: rgba(255,77,109,0.03);
}

/* ── Model Badge ───────────────────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.badge-lr { background: rgba(0,212,255,0.08); color: var(--accent); border: 1px solid rgba(0,212,255,0.15); }
.badge-ann { background: var(--purple-dim); color: var(--purple); border: 1px solid rgba(168,85,247,0.15); }
.badge-ensemble { background: var(--warning-dim); color: var(--warning); border: 1px solid rgba(251,191,36,0.15); }

/* ── Footer ─────────────────────────── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-muted);
    font-size: 0.72rem;
    letter-spacing: 1px;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.footer a { color: var(--accent); text-decoration: none; }

/* ── File Uploader ──────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── Data table ─────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Progress bar ───────────────────── */
.stProgress > div > div {
    background-color: var(--accent) !important;
    background: linear-gradient(90deg, var(--accent), #00e5a0) !important;
}

/* ── Labels ─────────────────────────── */
label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Divider ────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0;
}

/* ── Hide Streamlit branding ─────── */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* ── Metric Row (model info) ────────── */
.metric-row {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 0.5rem 0;
}
.metric-pill {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}
.metric-pill:hover {
    border-color: rgba(0,212,255,0.15);
}
.metric-pill span {
    color: var(--text-primary);
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Preset Buttons ─────────────────── */
.preset-btn {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}
.preset-btn:hover {
    border-color: var(--accent);
    background: var(--accent-dim);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}
.preset-btn .preset-icon { font-size: 1.5rem; display: block; margin-bottom: 0.3rem; }
.preset-btn .preset-title { font-weight: 600; font-size: 0.82rem; color: var(--text-primary); }
.preset-btn .preset-desc { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.2rem; }

/* ── Pulse Dot ──────────────────────── */
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ff4d6d;
    animation: pulseDot 1.5s ease-in-out infinite;
    margin-right: 0.4rem;
}
.pulse-dot.live {
    background: #00e5a0;
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(255,77,109,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 10px 4px rgba(255,77,109,0.2); }
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}

/* ── Explanation cards ──────────────── */
.explain-positive {
    background: rgba(255,77,109,0.04);
    border-left: 3px solid #ff4d6d;
    padding: 0.5rem 0.8rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.4rem;
}
.explain-negative {
    background: rgba(0,229,160,0.04);
    border-left: 3px solid #00e5a0;
    padding: 0.5rem 0.8rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.4rem;
}
</style>
"""


def inject_css():
    """Inject premium CSS into the page."""
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar logo and navigation info."""
    with st.sidebar:
        st.markdown("""
        <div class="logo-container">
            <div class="logo-text">🛡️ FraudShield</div>
            <div class="logo-sub">Fraud Detection System</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="padding: 0.5rem 0; font-size: 0.78rem; color: var(--text-muted);">
            <div style="margin-bottom: 0.8rem;">
                <span style="color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 2px;">MODELS</span><br>
                <span style="color: var(--text-secondary);">LR + ANN (Ensemble)</span>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <span style="color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 2px;">AUC SCORE</span><br>
                <span style="color: var(--text-secondary); font-family: 'JetBrains Mono', monospace;">0.9736</span>
            </div>
            <div style="margin-bottom: 0.8rem;">
                <span style="color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 2px;">THRESHOLD</span><br>
                <span style="color: var(--text-secondary); font-family: 'JetBrains Mono', monospace;">0.30</span>
            </div>
            <div>
                <span style="color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 2px;">FEATURES</span><br>
                <span style="color: var(--text-secondary);">Top 15 (PCA)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.7rem; color: var(--text-muted); text-align: center; padding: 0.5rem 0;">
            Built by <span style="color: var(--accent);">Nemish Sapara</span><br>
            ML InnovateX Hackathon
        </div>
        """, unsafe_allow_html=True)


def render_footer():
    """Render the page footer."""
    st.markdown("""
    <div class="footer">
        🛡️ FraudShield &nbsp;·&nbsp; ML InnovateX Hackathon &nbsp;·&nbsp;
        Built by <a href="#">Nemish Sapara</a> &nbsp;·&nbsp; Powered by Streamlit
    </div>
    """, unsafe_allow_html=True)
