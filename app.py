import streamlit as st
from utils import (
    inject_css, render_sidebar, render_footer,
    get_model, get_ann_model, init_session_state,
    create_pie_chart, create_risk_distribution,
    get_risk_color
)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Inject Theme ───────────────────────────────────────────────────────────────
inject_css()
render_sidebar()
init_session_state()
model, scaler, top_features, config, threshold, model_loaded = get_model()
ann_model, ann_loaded = get_ann_model()

if not model_loaded:
    st.stop()


# ─── Page Title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">
    <h1>🛡️ FraudShield Dashboard</h1>
    <div class="subtitle">Real-time fraud detection monitoring & analytics</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Stats Cards ────────────────────────────────────────────────────────────────
total = st.session_state.total_scans
fraud = st.session_state.fraud_detected
legit = st.session_state.legit_detected
rate = (fraud / total * 100) if total > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="stat-card accent">
        <span class="stat-icon">📊</span>
        <div class="stat-value">{total}</div>
        <div class="stat-label">Total Scans</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card success">
        <span class="stat-icon">✅</span>
        <div class="stat-value">{legit}</div>
        <div class="stat-label">Legitimate</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card danger">
        <span class="stat-icon">🚨</span>
        <div class="stat-value">{fraud}</div>
        <div class="stat-label">Fraud Detected</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="stat-card warning">
        <span class="stat-icon">📈</span>
        <div class="stat-value">{rate:.1f}%</div>
        <div class="stat-label">Fraud Rate</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    model_status = "✅ Dual" if ann_loaded else "⚡ LR Only"
    st.markdown(f"""
    <div class="stat-card purple">
        <span class="stat-icon">🤖</span>
        <div class="stat-value" style="font-size:1.2rem;">{model_status}</div>
        <div class="stat-label">Model Status</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Charts Row ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Detection Distribution</div>', unsafe_allow_html=True)

    if total > 0:
        fig = create_pie_chart(fraud, legit)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color: var(--text-muted);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-size: 0.85rem;">No transactions analyzed yet</div>
            <div style="font-size: 0.75rem; margin-top: 0.3rem; color: var(--text-muted);">
                Navigate to <b>Analyze</b> to start scanning
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)

    history = st.session_state.transaction_history
    if history:
        fig = create_risk_distribution(history)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color: var(--text-muted);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📈</div>
            <div style="font-size: 0.85rem;">Risk distribution will appear here</div>
            <div style="font-size: 0.75rem; margin-top: 0.3rem; color: var(--text-muted);">
                Analyze transactions to populate data
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Model Performance Cards ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown("""
    <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#00d4ff;">0.9736</div>
        <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; margin-top:0.2rem;">ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#00e5a0;">96.2%</div>
        <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; margin-top:0.2rem;">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#fbbf24;">0.30</div>
        <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; margin-top:0.2rem;">Threshold</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#f97316;">15</div>
        <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; margin-top:0.2rem;">Features</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown("""
    <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#a855f7;">ANN</div>
        <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; margin-top:0.2rem;">Deep Learning</div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    st.markdown("""
    <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#e2e8f0;">LR</div>
        <div style="font-size:0.7rem; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; margin-top:0.2rem;">Statistical</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─── Recent Activity ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Recent Activity Feed</div>', unsafe_allow_html=True)

if history:
    for txn in history[:8]:
        status_class = "status-fraud" if txn['prediction'] == 'Fraud' else "status-legit"
        status_icon = "🚨" if txn['prediction'] == 'Fraud' else "✅"
        risk_color = get_risk_color(txn['risk_level'])
        model_name = txn.get('model', 'LR')
        st.markdown(f"""
        <div class="activity-row">
            <span class="activity-time">{txn['timestamp']}</span>
            <span class="activity-amount">₹{txn['amount']:,.2f}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:{risk_color}; font-weight:600;">{txn['risk_level']}</span>
            <span class="activity-status {status_class}">{status_icon} {txn['prediction']}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0; color: var(--text-muted);">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🕐</div>
        <div style="font-size: 0.85rem;">No recent activity</div>
        <div style="font-size: 0.75rem; margin-top: 0.3rem;">Transaction history will appear here after analysis</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─── Quick Navigation ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)

q1, q2, q3, q4 = st.columns(4)

with q1:
    st.markdown("""
    <div class="preset-btn">
        <span class="preset-icon">🔍</span>
        <div class="preset-title">Analyze Transaction</div>
        <div class="preset-desc">Single transaction fraud check</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Analyze →", key="nav_analyze"):
        st.switch_page("pages/1_🔍_Analyze.py")

with q2:
    st.markdown("""
    <div class="preset-btn">
        <span class="preset-icon">📊</span>
        <div class="preset-title">Batch Processing</div>
        <div class="preset-desc">Upload CSV for bulk analysis</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Batch →", key="nav_batch"):
        st.switch_page("pages/2_📊_Batch.py")

with q3:
    st.markdown("""
    <div class="preset-btn">
        <span class="preset-icon">📈</span>
        <div class="preset-title">Analytics & Insights</div>
        <div class="preset-desc">Feature importance & model stats</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Analytics →", key="nav_analytics"):
        st.switch_page("pages/3_📈_Analytics.py")

with q4:
    st.markdown("""
    <div class="preset-btn">
        <span class="preset-icon">🔴</span>
        <div class="preset-title">Live Monitor</div>
        <div class="preset-desc">Real-time transaction stream</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Live →", key="nav_live"):
        st.switch_page("pages/5_🔴_Live.py")


# ─── Footer ─────────────────────────────────────────────────────────────────────
render_footer()