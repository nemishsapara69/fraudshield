import streamlit as st
from utils import (
    inject_css, render_sidebar, render_footer,
    get_model, get_ann_model, init_session_state,
    predict_transaction, predict_transaction_ann, predict_dual,
    add_to_history,
    create_gauge_chart, create_feature_radar,
    create_model_comparison_chart,
    compute_feature_contributions, create_waterfall_chart,
    get_risk_color
)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Analyze Transaction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()
render_sidebar()
init_session_state()
model, scaler, top_features, config, threshold, model_loaded = get_model()
ann_model, ann_loaded = get_ann_model()

if not model_loaded:
    st.stop()


# ─── Preset Data ────────────────────────────────────────────────────────────────
PRESETS = {
    'typical': {
        'name': '📦 Typical Purchase',
        'amount': 120.0,
        'time': 1500,
        'v_vals': {},
        'msg': '📦 **Typical purchase** — All features at baseline (0.0), amount ₹120, time 1500s.'
    },
    'suspicious': {
        'name': '⚠️ Suspicious Pattern',
        'amount': 350.0,
        'time': 4200,
        'v_vals': {'V4': 4.5, 'V11': -3.2, 'V14': -8.5, 'V17': -5.0},
        'msg': '⚠️ **Suspicious pattern** loaded — Anomalous V4, V11, V14, V17 values.'
    },
    'high_value': {
        'name': '💎 High Value Txn',
        'amount': 15000.0,
        'time': 800,
        'v_vals': {'V4': 2.1, 'V14': -4.2},
        'msg': '💎 **High-value transaction** — ₹15,000 with slightly anomalous features.'
    },
    'fraud_extreme': {
        'name': '🚨 Extreme Fraud',
        'amount': 2800.0,
        'time': 200,
        'v_vals': {'V4': 5.8, 'V11': -4.5, 'V14': -9.8, 'V17': -7.2, 'V12': -6.0, 'V10': -5.0},
        'msg': '🚨 **Extreme fraud pattern** — Multiple critical anomalies detected.'
    }
}

# Initialize preset state
if 'preset' not in st.session_state:
    st.session_state.preset = None


# ─── Page Title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">
    <h1>🔍 Transaction Analysis</h1>
    <div class="subtitle">Enter transaction details to detect potential fraud using dual AI models</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Input Section ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

# Determine defaults from preset
active_preset = st.session_state.preset
preset_data = PRESETS.get(active_preset, None)

default_amount = preset_data['amount'] if preset_data else 120.0
default_time = preset_data['time'] if preset_data else 1500

with left_col:
    # Transaction Details Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input(
            "💰 Transaction Amount (₹ / $)",
            min_value=0.0, max_value=50000.0,
            value=default_amount, step=0.5,
            help="The monetary value of the transaction"
        )
    with c2:
        time_val = st.number_input(
            "⏱️ Time (seconds from first txn)",
            min_value=0, max_value=172800,
            value=default_time,
            help="Time elapsed since first transaction in the dataset"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # PCA Features Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">PCA-Transformed Features</div>', unsafe_allow_html=True)
    st.caption("Anonymized PCA components from original transaction data. Default = 0.0 for typical transactions.")

    v_features = [f for f in top_features if f.startswith('V')]
    v_inputs = {}
    cols = st.columns(3)
    for i, feat in enumerate(v_features):
        default_v = 0.0
        if preset_data and feat in preset_data.get('v_vals', {}):
            default_v = preset_data['v_vals'][feat]
        with cols[i % 3]:
            v_inputs[feat] = st.number_input(
                feat,
                min_value=-30.0, max_value=35.0,
                value=default_v, step=0.01,
                format="%.3f",
                key=f"analyze_{feat}"
            )

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # Quick Test Presets
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Quick Test Presets</div>', unsafe_allow_html=True)
    st.caption("Load predefined transaction profiles to test the model.")

    p1, p2 = st.columns(2)
    with p1:
        if st.button("📦 Typical Purchase", use_container_width=True):
            st.session_state.preset = 'typical'
            st.rerun()
    with p2:
        if st.button("⚠️ Suspicious Pattern", use_container_width=True):
            st.session_state.preset = 'suspicious'
            st.rerun()

    p3, p4 = st.columns(2)
    with p3:
        if st.button("💎 High Value Txn", use_container_width=True):
            st.session_state.preset = 'high_value'
            st.rerun()
    with p4:
        if st.button("🚨 Extreme Fraud", use_container_width=True):
            st.session_state.preset = 'fraud_extreme'
            st.rerun()

    p5, _ = st.columns(2)
    with p5:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.preset = None
            st.rerun()

    if preset_data:
        if 'Suspicious' in preset_data['msg'] or 'Extreme' in preset_data['msg']:
            st.warning(preset_data['msg'])
        else:
            st.info(preset_data['msg'])

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Info Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)

    ann_status = "✅ Loaded" if ann_loaded else "❌ Not Available"
    st.markdown(f"""
    <div class="metric-row" style="flex-direction:column; gap:0.6rem;">
        <div class="metric-pill" style="text-align:left;">Primary &nbsp;&nbsp;<span>Logistic Regression</span></div>
        <div class="metric-pill" style="text-align:left;">Secondary &nbsp;&nbsp;<span>ANN (Deep Learning)</span></div>
        <div class="metric-pill" style="text-align:left;">ANN Status &nbsp;&nbsp;<span>{ann_status}</span></div>
        <div class="metric-pill" style="text-align:left;">ROC-AUC &nbsp;&nbsp;<span>0.9736</span></div>
        <div class="metric-pill" style="text-align:left;">Threshold &nbsp;&nbsp;<span>{threshold}</span></div>
        <div class="metric-pill" style="text-align:left;">Ensemble &nbsp;&nbsp;<span>40% LR + 60% ANN</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Analyze Button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  ANALYZE TRANSACTION", use_container_width=True, type="primary")


# ─── Prediction Results ────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("🔄 Analyzing transaction with dual AI models..."):
        # LR prediction
        pred_lr, proba_lr, risk_lr = predict_transaction(model, top_features, threshold, amount, time_val, v_inputs)

        # ANN prediction (if available)
        if ann_loaded:
            pred_ann, proba_ann, risk_ann = predict_transaction_ann(ann_model, top_features, threshold, amount, time_val, v_inputs)
            # Ensemble
            proba_ensemble = 0.4 * proba_lr + 0.6 * proba_ann
            pred_ensemble = int(proba_ensemble >= threshold)
            risk_ensemble = "LOW" if proba_ensemble < 0.1 else "MEDIUM" if proba_ensemble < 0.3 else "HIGH" if proba_ensemble < 0.6 else "CRITICAL"
            # Use ensemble as primary result
            pred = pred_ensemble
            proba = proba_ensemble
            risk_level = risk_ensemble
            add_to_history(amount, time_val, pred, proba, risk_level, "Ensemble")
        else:
            pred = pred_lr
            proba = proba_lr
            risk_level = risk_lr
            proba_ann = None
            add_to_history(amount, time_val, pred, proba, risk_level, "LR")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result Card ──
    res_left, res_right = st.columns([3, 2])

    with res_left:
        risk_color = get_risk_color(risk_level)
        risk_class = f"risk-{risk_level.lower()}"

        if pred == 1:
            st.markdown(f"""
            <div class="result-card result-fraud">
                <span class="result-icon">🚨</span>
                <div class="result-title">FRAUDULENT TRANSACTION</div>
                <div class="result-desc">This transaction exhibits fraud indicators above the decision threshold ({threshold})</div>
                <div class="prob-badge">Fraud Probability: {proba:.1%}</div>
                <div style="margin-top: 1rem;">
                    <span class="risk-badge {risk_class}">⚡ Risk Level: {risk_level}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.error("🚨 **Recommended Action:** Block transaction immediately and notify cardholder. Flag for manual review.")
        else:
            st.markdown(f"""
            <div class="result-card result-legit">
                <span class="result-icon">✅</span>
                <div class="result-title">LEGITIMATE TRANSACTION</div>
                <div class="result-desc">Transaction appears normal. No significant fraud indicators detected.</div>
                <div class="prob-badge">Fraud Probability: {proba:.1%}</div>
                <div style="margin-top: 1rem;">
                    <span class="risk-badge {risk_class}">✓ Risk Level: {risk_level}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("✅ **Transaction cleared.** Safe to proceed with authorization.")

    with res_right:
        # Gauge Chart
        st.markdown('<div class="glass-card" style="padding: 1rem;">', unsafe_allow_html=True)
        gauge = create_gauge_chart(proba, threshold)
        st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Comparison (if ANN available) ──
    if ann_loaded and proba_ann is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        comp_left, comp_right = st.columns([1, 1])

        with comp_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Dual Model Comparison</div>', unsafe_allow_html=True)
            fig_comp = create_model_comparison_chart(proba_lr, proba_ann, threshold)
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with comp_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Model Agreement</div>', unsafe_allow_html=True)

            agree = "✅ AGREE" if pred_lr == pred_ann else "⚠️ DISAGREE"
            agree_color = "#00e5a0" if pred_lr == pred_ann else "#fbbf24"

            st.markdown(f"""
            <div style="text-align:center; padding: 1rem 0;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:{agree_color}; margin-bottom:1rem;">{agree}</div>

                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; margin-top:1rem;">
                    <div>
                        <span class="model-badge badge-lr">LR</span>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#00d4ff; margin-top:0.5rem;">{proba_lr:.1%}</div>
                        <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.2rem;">{"🚨 Fraud" if pred_lr == 1 else "✅ Legit"}</div>
                    </div>
                    <div>
                        <span class="model-badge badge-ann">ANN</span>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#a855f7; margin-top:0.5rem;">{proba_ann:.1%}</div>
                        <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.2rem;">{"🚨 Fraud" if pred_ann == 1 else "✅ Legit"}</div>
                    </div>
                    <div>
                        <span class="model-badge badge-ensemble">Ensemble</span>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#fbbf24; margin-top:0.5rem;">{proba_ensemble:.1%}</div>
                        <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.2rem;">{"🚨 Fraud" if pred_ensemble == 1 else "✅ Legit"}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ── Explainability (SHAP-style) ──
    st.markdown("<br>", unsafe_allow_html=True)
    explain_left, explain_right = st.columns([3, 2])

    with explain_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🧠 AI Explainability — Why This Decision?</div>', unsafe_allow_html=True)
        st.caption("Feature contributions based on model coefficients — shows which features pushed the prediction toward fraud or legitimate.")

        contributions, intercept = compute_feature_contributions(model, top_features, amount, time_val, v_inputs)
        waterfall = create_waterfall_chart(contributions)
        st.plotly_chart(waterfall, use_container_width=True, config={'displayModeBar': False})

        st.markdown('</div>', unsafe_allow_html=True)

    with explain_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Top Contributing Factors</div>', unsafe_allow_html=True)

        for feat, val in contributions[:6]:
            if abs(val) < 0.0001:
                continue
            css_class = "explain-positive" if val > 0 else "explain-negative"
            direction = "↑ Increases fraud risk" if val > 0 else "↓ Decreases fraud risk"
            color = "#ff4d6d" if val > 0 else "#00e5a0"
            st.markdown(f"""
            <div class="{css_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; font-weight:600; color:var(--text-primary);">{feat}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:700; color:{color};">{val:+.4f}</span>
                </div>
                <div style="font-size:0.68rem; color:var(--text-muted); margin-top:0.2rem;">{direction}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature Radar ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Analysis Radar</div>', unsafe_allow_html=True)
    st.caption("Visualizes the magnitude of each PCA feature — higher values indicate more deviation from normal.")

    radar = create_feature_radar(top_features, v_inputs)
    st.plotly_chart(radar, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Detailed Breakdown ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Detailed Breakdown</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1.5px; text-transform:uppercase;">LR Probability</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:#00d4ff; margin-top:0.3rem;">{proba_lr:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        ann_display = f"{proba_ann:.6f}" if proba_ann is not None else "N/A"
        ann_color = "#a855f7" if proba_ann is not None else "#475569"
        st.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1.5px; text-transform:uppercase;">ANN Probability</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:{ann_color}; margin-top:0.3rem;">{ann_display}</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1.5px; text-transform:uppercase;">Decision Threshold</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:var(--warning); margin-top:0.3rem;">{threshold}</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        margin = proba - threshold
        margin_color = "#ff4d6d" if margin > 0 else "#00e5a0"
        st.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1.5px; text-transform:uppercase;">Margin from Threshold</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:{margin_color}; margin-top:0.3rem;">{margin:+.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────────
render_footer()
