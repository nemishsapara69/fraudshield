import streamlit as st
import pandas as pd
import numpy as np
import io
from utils import (
    inject_css, render_sidebar, render_footer,
    get_model, get_ann_model, init_session_state,
    build_input, get_risk_level, get_risk_color,
    predict_dual,
    PLOTLY_LAYOUT
)
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Batch Analysis",
    page_icon="📊",
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


# ─── Page Title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">
    <h1>📊 Batch Analysis</h1>
    <div class="subtitle">Upload a CSV file to analyze multiple transactions at once with Ensembled AI</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Upload Section ────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Upload Transaction Data</div>', unsafe_allow_html=True)

st.markdown("""
<div style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 1rem; line-height: 1.6;">
    Upload a CSV file containing transaction data. The file should include columns for the PCA-transformed features 
    (V1-V28), Amount, and optionally Time. The dual-model engine will analyze each transaction and classify it as legitimate or fraudulent.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV with transaction data. Required columns: V features + Amount"
)

st.markdown('</div>', unsafe_allow_html=True)


# ─── Sample Data Generator ─────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Generate Sample Data</div>', unsafe_allow_html=True)
st.caption("Don't have a CSV? Generate sample transaction data for testing.")

s1, s2, s3 = st.columns(3)
with s1:
    n_samples = st.number_input("Number of transactions", min_value=5, max_value=500, value=20, step=5)
with s2:
    fraud_pct = st.slider("Approximate fraud %", min_value=0, max_value=50, value=15)
with s3:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🎲 Generate Sample CSV", use_container_width=True)

if generate_btn:
    np.random.seed(42)
    n_fraud = max(1, int(n_samples * fraud_pct / 100))
    n_legit = n_samples - n_fraud

    v_features_all = [f for f in top_features if f.startswith('V')]

    # Generate legit transactions (near zero V-values)
    legit_data = {}
    for v in v_features_all:
        legit_data[v] = np.random.normal(0, 0.5, n_legit)
    legit_data['Amount'] = np.random.exponential(80, n_legit)
    legit_data['Time'] = np.random.randint(0, 172800, n_legit)

    # Generate fraud transactions (anomalous V-values)
    fraud_data = {}
    for v in v_features_all:
        fraud_data[v] = np.random.normal(0, 3.0, n_fraud)
    # Push key fraud indicators
    if 'V14' in fraud_data:
        fraud_data['V14'] = np.random.uniform(-10, -5, n_fraud)
    if 'V4' in fraud_data:
        fraud_data['V4'] = np.random.uniform(3, 6, n_fraud)
    if 'V11' in fraud_data:
        fraud_data['V11'] = np.random.uniform(-5, -2, n_fraud)
    fraud_data['Amount'] = np.random.exponential(500, n_fraud)
    fraud_data['Time'] = np.random.randint(0, 172800, n_fraud)

    df_legit = pd.DataFrame(legit_data)
    df_fraud = pd.DataFrame(fraud_data)
    sample_df = pd.concat([df_legit, df_fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # Round for cleanliness
    for col in sample_df.columns:
        if col.startswith('V'):
            sample_df[col] = sample_df[col].round(4)
    sample_df['Amount'] = sample_df['Amount'].round(2)

    st.session_state['sample_csv'] = sample_df
    st.success(f"✅ Generated {n_samples} sample transactions ({n_fraud} fraud, {n_legit} legit)")

    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "📥 Download Sample CSV",
        csv_buffer.getvalue(),
        "sample_transactions.csv",
        "text/csv",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)


# ─── Process Uploaded/Generated File ───────────────────────────────────────────
data_to_process = None

if uploaded_file is not None:
    data_to_process = pd.read_csv(uploaded_file)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(data_to_process.head(10), use_container_width=True)
    st.caption(f"Loaded **{len(data_to_process)}** transactions with **{len(data_to_process.columns)}** columns")
    st.markdown('</div>', unsafe_allow_html=True)

elif 'sample_csv' in st.session_state:
    data_to_process = st.session_state['sample_csv']


if data_to_process is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀  RUN ENSEMBLED BATCH ANALYSIS", use_container_width=True, type="primary")

    if analyze_btn:
        v_features = [f for f in top_features if f.startswith('V')]
        results = []

        progress = st.progress(0, text="Analyzing transactions with dual AI...")

        for idx, row in data_to_process.iterrows():
            # Build v_inputs from row
            v_vals = {}
            for v in v_features:
                v_vals[v] = float(row.get(v, 0.0))

            amt = float(row.get('Amount', row.get('amount', 0.0)))
            t = int(row.get('Time', row.get('time', 0)))

            # Use dual model predictions if ANN is loaded, else fallback to LR
            if ann_loaded:
                res = predict_dual(model, ann_model, top_features, threshold, amt, t, v_vals)
                proba = res['ensemble']['proba']
                pred = res['ensemble']['pred']
                risk_level = res['ensemble']['risk']
                lr_prob = res['lr']['proba']
                ann_prob = res['ann']['proba']
            else:
                input_row = build_input(top_features, amt, t, v_vals)
                proba = model.predict_proba(input_row)[0][1]
                pred = int(proba >= threshold)
                risk_level = get_risk_level(proba)
                lr_prob = proba
                ann_prob = None

            results.append({
                'Transaction #': idx + 1,
                'Amount': f"₹{amt:,.2f}",
                'Fraud Prob (Ensemble)': round(proba, 4) if ann_loaded else round(proba, 4),
                'Prediction': '🚨 Fraud' if pred == 1 else '✅ Legit',
                'Risk Level': risk_level,
                'LR Prob': round(lr_prob, 4),
                'ANN Prob': round(ann_prob, 4) if ann_prob is not None else 'N/A',
                'Raw Probability': proba,
                'Pred': pred,
            })

            progress.progress((idx + 1) / len(data_to_process),
                              text=f"Analyzing transaction {idx + 1} of {len(data_to_process)}...")

        progress.empty()
        results_df = pd.DataFrame(results)

        # ── Summary Stats ──
        total = len(results)
        fraud_count = results_df['Pred'].sum()
        legit_count = total - fraud_count
        avg_prob = results_df['Raw Probability'].mean()
        max_prob = results_df['Raw Probability'].max()

        st.markdown("<br>", unsafe_allow_html=True)

        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.markdown(f"""
            <div class="stat-card accent">
                <span class="stat-icon">📊</span>
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Processed</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="stat-card success">
                <span class="stat-icon">✅</span>
                <div class="stat-value">{legit_count}</div>
                <div class="stat-label">Legitimate</div>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div class="stat-card danger">
                <span class="stat-icon">🚨</span>
                <div class="stat-value">{fraud_count}</div>
                <div class="stat-label">Fraudulent</div>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="stat-card warning">
                <span class="stat-icon">📈</span>
                <div class="stat-value">{avg_prob:.2%}</div>
                <div class="stat-label">Avg Risk Score</div>
            </div>
            """, unsafe_allow_html=True)
        with s5:
            max_color = "#ff4d6d" if max_prob >= threshold else "#00e5a0"
            st.markdown(f"""
            <div class="stat-card {'danger' if max_prob >= threshold else 'success'}">
                <span class="stat-icon">⚡</span>
                <div class="stat-value" style="color:{max_color}">{max_prob:.2%}</div>
                <div class="stat-label">Max Risk Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Distribution Chart ──
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)

            fig = px.histogram(
                results_df, x='Raw Probability', nbins=25,
                color_discrete_sequence=['#00d4ff'],
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="#ff4d6d",
                          annotation_text=f"Threshold ({threshold})", annotation_font_color="#ff4d6d")
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                xaxis=dict(title="Fraud Probability", gridcolor='rgba(100,116,139,0.15)'),
                yaxis=dict(title="Count", gridcolor='rgba(100,116,139,0.15)'),
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Risk Level Breakdown</div>', unsafe_allow_html=True)

            risk_counts = results_df['Risk Level'].value_counts()
            risk_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            risk_colors = ['#00e5a0', '#fbbf24', '#f97316', '#ff4d6d']
            
            # Match counts to ordered categories
            counts = [risk_counts.get(r, 0) for r in risk_order]

            fig2 = go.Figure(go.Bar(
                x=counts,
                y=risk_order,
                orientation='h',
                marker=dict(color=risk_colors),
                text=counts,
                textposition='outside',
                textfont=dict(color='#94a3b8', size=12),
            ))
            fig2.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                xaxis=dict(title="Count", gridcolor='rgba(100,116,139,0.15)'),
                yaxis=dict(tickfont=dict(size=12, color='#c8d6e5')),
            )
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Results Table ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)

        display_df = results_df[['Transaction #', 'Amount', 'Prediction', 'Risk Level', 'Fraud Prob (Ensemble)', 'LR Prob', 'ANN Prob']].copy()
        st.dataframe(display_df, use_container_width=True, height=400)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Export ──
        st.markdown("<br>", unsafe_allow_html=True)
        export_df = results_df[['Transaction #', 'Amount', 'Raw Probability', 'Prediction', 'Risk Level', 'LR Prob', 'ANN Prob']].copy()
        csv_out = export_df.to_csv(index=False)

        st.download_button(
            "📥  DOWNLOAD RESULTS AS CSV",
            csv_out,
            "fraudshield_batch_results.csv",
            "text/csv",
            use_container_width=True
        )


# ─── Footer ─────────────────────────────────────────────────────────────────────
render_footer()
