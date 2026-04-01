import streamlit as st
import time
import random
from utils import (
    inject_css, render_sidebar, render_footer,
    get_model, get_ann_model, init_session_state,
    predict_dual, predict_transaction,
    generate_live_transaction,
    get_risk_color, get_risk_level,
    PLOTLY_LAYOUT
)
import plotly.graph_objects as go
import pandas as pd
import datetime

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Live Monitor",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()
render_sidebar()
init_session_state()
model_lr, scaler, top_features, config, threshold, model_loaded = get_model()
ann_model, ann_loaded = get_ann_model()

if not model_loaded:
    st.stop()


# ─── Page Title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">
    <h1>🔴 Live Transaction Monitor</h1>
    <div class="subtitle">Real-time transaction stream simulation with dual AI fraud detection</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Controls ──────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Stream Controls</div>', unsafe_allow_html=True)

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.5, 1, 1, 1.5])

with ctrl1:
    n_transactions = st.slider("Number of transactions", min_value=5, max_value=50, value=15, step=5)

with ctrl2:
    speed = st.select_slider("Speed", options=["Slow", "Normal", "Fast"], value="Normal")

with ctrl3:
    fraud_rate = st.slider("Simulated fraud %", min_value=5, max_value=40, value=12)

with ctrl4:
    st.markdown("<br>", unsafe_allow_html=True)
    start_btn = st.button("▶️  START LIVE STREAM", use_container_width=True, type="primary")

st.markdown('</div>', unsafe_allow_html=True)

speed_map = {"Slow": 1.5, "Normal": 0.7, "Fast": 0.25}
delay = speed_map[speed]


# ─── Live Stream ────────────────────────────────────────────────────────────────
if start_btn:
    st.markdown("<br>", unsafe_allow_html=True)

    # Live stats placeholders
    stats_container = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)

    # Feed header
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <span class="pulse-dot live"></span> Live Transaction Feed
    </div>
    """, unsafe_allow_html=True)

    # Table header
    st.markdown("""
    <div class="live-row" style="background: rgba(0,212,255,0.04); font-weight:600; font-size:0.72rem; color:var(--text-muted); letter-spacing:1px;">
        <span>TXN ID</span>
        <span>MERCHANT</span>
        <span>AMOUNT</span>
        <span>RISK</span>
        <span>PROBABILITY</span>
        <span>STATUS</span>
    </div>
    """, unsafe_allow_html=True)

    feed_container = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart placeholders
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        timeline_container = st.empty()
    with chart_col2:
        risk_pie_container = st.empty()

    progress = st.progress(0, text="Starting live stream...")

    # Run simulation
    live_results = []
    live_fraud = 0
    live_legit = 0
    feed_html_rows = []

    for i in range(n_transactions):
        txn = generate_live_transaction(top_features)

        # Adjust fraud rate
        if random.random() > (fraud_rate / 100):
            # Force more legit
            for v in txn['v_vals']:
                txn['v_vals'][v] = random.gauss(0, 0.5)

        # Predict
        pred, proba, risk_level = predict_transaction(
            model_lr, top_features, threshold,
            txn['amount'], txn['time_val'], txn['v_vals']
        )

        is_fraud = pred == 1
        if is_fraud:
            live_fraud += 1
        else:
            live_legit += 1

        risk_color = get_risk_color(risk_level)
        status_text = "🚨 FRAUD" if is_fraud else "✅ LEGIT"
        status_color = "#ff4d6d" if is_fraud else "#00e5a0"
        row_class = "live-row fraud-row" if is_fraud else "live-row"

        live_results.append({
            'txn_id': txn['txn_id'],
            'merchant': txn['merchant'],
            'city': txn['city'],
            'amount': txn['amount'],
            'probability': proba,
            'risk_level': risk_level,
            'prediction': 'Fraud' if is_fraud else 'Legit',
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
        })

        # Build feed HTML
        feed_html_rows.insert(0, f"""
        <div class="{row_class}">
            <span style="font-family:'JetBrains Mono',monospace; color:var(--text-muted); font-size:0.72rem;">{txn['txn_id']}</span>
            <span style="color:var(--text-secondary); font-size:0.78rem;">{txn['merchant']}<br><span style="font-size:0.65rem; color:var(--text-muted);">{txn['city']}</span></span>
            <span style="font-family:'JetBrains Mono',monospace; font-weight:600; color:var(--text-primary);">₹{txn['amount']:,.2f}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:{risk_color}; font-weight:600;">{risk_level}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:{status_color};">{proba:.1%}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:600; color:{status_color};">{status_text}</span>
        </div>
        """)

        feed_container.markdown("\n".join(feed_html_rows[:20]), unsafe_allow_html=True)

        # Update stats
        total_so_far = i + 1
        fraud_rate_live = (live_fraud / total_so_far * 100)
        stats_container.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:0.8rem;">
            <div class="stat-card accent">
                <span class="stat-icon">📡</span>
                <div class="stat-value">{total_so_far}</div>
                <div class="stat-label">Processed</div>
            </div>
            <div class="stat-card success">
                <span class="stat-icon">✅</span>
                <div class="stat-value">{live_legit}</div>
                <div class="stat-label">Legitimate</div>
            </div>
            <div class="stat-card danger">
                <span class="stat-icon">🚨</span>
                <div class="stat-value">{live_fraud}</div>
                <div class="stat-label">Flagged</div>
            </div>
            <div class="stat-card warning">
                <span class="stat-icon">📈</span>
                <div class="stat-value">{fraud_rate_live:.1f}%</div>
                <div class="stat-label">Fraud Rate</div>
            </div>
            <div class="stat-card purple">
                <span class="stat-icon">🔴</span>
                <div class="stat-value" style="font-size:1rem; color:#00e5a0;">● LIVE</div>
                <div class="stat-label">Status</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Update timeline chart
        df_live = pd.DataFrame(live_results)
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=list(range(len(df_live))),
            y=df_live['probability'].tolist(),
            mode='lines+markers',
            line=dict(color='#00d4ff', width=2),
            marker=dict(
                size=8,
                color=['#ff4d6d' if p >= threshold else '#00e5a0' for p in df_live['probability']],
                line=dict(width=1, color='rgba(0,0,0,0.3)')
            ),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.04)',
        ))
        fig_timeline.add_hline(y=threshold, line_dash="dash", line_color="rgba(255,77,109,0.5)",
                               annotation_text="Threshold", annotation_font_color="#ff4d6d")
        fig_timeline.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            title=dict(text="Fraud Probability Timeline", font=dict(size=13, color="#94a3b8")),
            xaxis=dict(title="Transaction #", gridcolor='rgba(100,116,139,0.12)'),
            yaxis=dict(title="Probability", gridcolor='rgba(100,116,139,0.12)', range=[0, 1]),
            showlegend=False,
        )
        with chart_col1:
            timeline_container.markdown('<div class="glass-card">', unsafe_allow_html=True)
            timeline_container.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})

        # Update risk pie
        risk_counts = df_live['risk_level'].value_counts()
        risk_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        risk_colors_map = {'LOW': '#00e5a0', 'MEDIUM': '#fbbf24', 'HIGH': '#f97316', 'CRITICAL': '#ff4d6d'}
        present_risks = [r for r in risk_order if r in risk_counts.index]

        if present_risks:
            fig_risk = go.Figure(go.Pie(
                labels=present_risks,
                values=[risk_counts[r] for r in present_risks],
                hole=0.5,
                marker=dict(
                    colors=[risk_colors_map[r] for r in present_risks],
                    line=dict(color='#0f172a', width=2)
                ),
                textinfo='label+percent',
                textfont=dict(size=10, color='#e2e8f0'),
            ))
            fig_risk.update_layout(
                **PLOTLY_LAYOUT,
                height=300,
                title=dict(text="Risk Distribution", font=dict(size=13, color="#94a3b8")),
                showlegend=False,
            )
            with chart_col2:
                risk_pie_container.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})

        progress.progress((i + 1) / n_transactions, text=f"Processing transaction {i + 1} of {n_transactions}...")
        time.sleep(delay)

    progress.empty()

    # Final status
    stats_container.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:0.8rem;">
        <div class="stat-card accent">
            <span class="stat-icon">📡</span>
            <div class="stat-value">{n_transactions}</div>
            <div class="stat-label">Processed</div>
        </div>
        <div class="stat-card success">
            <span class="stat-icon">✅</span>
            <div class="stat-value">{live_legit}</div>
            <div class="stat-label">Legitimate</div>
        </div>
        <div class="stat-card danger">
            <span class="stat-icon">🚨</span>
            <div class="stat-value">{live_fraud}</div>
            <div class="stat-label">Flagged</div>
        </div>
        <div class="stat-card warning">
            <span class="stat-icon">📈</span>
            <div class="stat-value">{(live_fraud / n_transactions * 100):.1f}%</div>
            <div class="stat-label">Fraud Rate</div>
        </div>
        <div class="stat-card purple">
            <span class="stat-icon">✓</span>
            <div class="stat-value" style="font-size:1rem; color:var(--text-secondary);">COMPLETE</div>
            <div class="stat-label">Status</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"✅ **Stream complete.** Processed {n_transactions} transactions — {live_fraud} flagged as fraud ({live_fraud/n_transactions*100:.1f}%)")

    # Export
    if live_results:
        export_df = pd.DataFrame(live_results)
        csv_out = export_df.to_csv(index=False)
        st.download_button(
            "📥  DOWNLOAD LIVE STREAM RESULTS",
            csv_out,
            "fraudshield_live_results.csv",
            "text/csv",
            use_container_width=True
        )

else:
    # Default view when stream hasn't started
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 3rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔴</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; font-weight:700; color:var(--text-primary); margin-bottom:0.8rem;">
            Live Transaction Monitor
        </div>
        <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6; max-width:500px; margin:0 auto;">
            Simulate a real-time transaction stream to see FraudShield's dual AI models 
            analyze transactions as they arrive. Configure the stream parameters above and press 
            <span style="color:var(--accent); font-weight:600;">START LIVE STREAM</span> to begin.
        </div>
        <div style="margin-top:1.5rem; display:flex; gap:0.6rem; justify-content:center; flex-wrap:wrap;">
            <span class="model-badge badge-lr">LR Model</span>
            <span class="model-badge badge-ann">ANN Model</span>
            <span class="model-badge badge-ensemble">Ensemble</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────────
render_footer()
