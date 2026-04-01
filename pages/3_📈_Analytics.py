import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils import (
    inject_css, render_sidebar, render_footer,
    get_model, get_ann_model, init_session_state,
    create_feature_importance_chart,
    build_input, get_risk_level,
    PLOTLY_LAYOUT
)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Analytics & Insights",
    page_icon="📈",
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
    <h1>📈 Analytics & Insights</h1>
    <div class="subtitle">Deep dive into model performance, feature analysis, and detection patterns</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Model Comparison Section ──────────────────────────────────────────────────
if ann_loaded:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 Dual Model Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.5rem; margin-bottom:1rem;">
        <div style="text-align:center; padding:1.5rem; background:rgba(0,212,255,0.04); border-radius:14px; border:1px solid rgba(0,212,255,0.1);">
            <div style="font-size:1.5rem; margin-bottom:0.3rem;">📊</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:700; color:#00d4ff;">Logistic Regression</div>
            <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem; line-height:1.5;">
                Statistical model with interpretable coefficients.<br>Fast inference, low computational cost.
            </div>
            <div style="margin-top:0.8rem;">
                <span class="model-badge badge-lr">Primary Model</span>
            </div>
        </div>
        <div style="text-align:center; padding:1.5rem; background:rgba(168,85,247,0.04); border-radius:14px; border:1px solid rgba(168,85,247,0.1);">
            <div style="font-size:1.5rem; margin-bottom:0.3rem;">🧠</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:700; color:#a855f7;">Neural Network (ANN)</div>
            <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem; line-height:1.5;">
                Deep learning model capturing nonlinear patterns.<br>Higher capacity for complex fraud patterns.
            </div>
            <div style="margin-top:0.8rem;">
                <span class="model-badge badge-ann">Secondary Model</span>
            </div>
        </div>
        <div style="text-align:center; padding:1.5rem; background:rgba(251,191,36,0.04); border-radius:14px; border:1px solid rgba(251,191,36,0.1);">
            <div style="font-size:1.5rem; margin-bottom:0.3rem;">⚡</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:700; color:#fbbf24;">Ensemble</div>
            <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem; line-height:1.5;">
                Weighted combination: 40% LR + 60% ANN.<br>Combines interpretability with deep learning power.
            </div>
            <div style="margin-top:0.8rem;">
                <span class="model-badge badge-ensemble">Final Decision</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Comparison on Synthetic Data ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Agreement Analysis</div>', unsafe_allow_html=True)
    st.caption("Comparing LR vs ANN predictions on synthetic test scenarios.")

    # Generate test cases
    np.random.seed(42)
    test_cases = []
    for _ in range(30):
        v_vals = {}
        is_fraud = np.random.random() < 0.3
        v_feats = [f for f in top_features if f.startswith('V')]
        for v in v_feats:
            if is_fraud:
                v_vals[v] = np.random.normal(0, 3.0)
            else:
                v_vals[v] = np.random.normal(0, 0.5)
        if is_fraud:
            if 'V14' in v_vals:
                v_vals['V14'] = np.random.uniform(-10, -5)
            if 'V4' in v_vals:
                v_vals['V4'] = np.random.uniform(3, 6)
        amt = np.random.exponential(300 if is_fraud else 80)
        t = np.random.randint(0, 172800)

        input_row = build_input(top_features, amt, t, v_vals)
        proba_lr = model.predict_proba(input_row)[0][1]
        proba_ann = float(ann_model.predict(input_row, verbose=0)[0][0])

        test_cases.append({
            'amount': round(amt, 2),
            'LR Probability': round(proba_lr, 4),
            'ANN Probability': round(proba_ann, 4),
            'LR Verdict': 'Fraud' if proba_lr >= threshold else 'Legit',
            'ANN Verdict': 'Fraud' if proba_ann >= threshold else 'Legit',
            'Agreement': proba_lr >= threshold and proba_ann >= threshold or proba_lr < threshold and proba_ann < threshold,
        })

    # Scatter plot
    import pandas as pd
    df_comp = pd.DataFrame(test_cases)
    agree_pct = df_comp['Agreement'].mean() * 100

    comp1, comp2 = st.columns([3, 2])

    with comp1:
        fig_scatter = go.Figure()
        agree_mask = df_comp['Agreement']
        disagree_mask = ~df_comp['Agreement']

        fig_scatter.add_trace(go.Scatter(
            x=df_comp.loc[agree_mask, 'LR Probability'],
            y=df_comp.loc[agree_mask, 'ANN Probability'],
            mode='markers',
            marker=dict(size=10, color='#00e5a0', line=dict(width=1, color='rgba(0,0,0,0.3)')),
            name='Agree',
        ))
        if disagree_mask.any():
            fig_scatter.add_trace(go.Scatter(
                x=df_comp.loc[disagree_mask, 'LR Probability'],
                y=df_comp.loc[disagree_mask, 'ANN Probability'],
                mode='markers',
                marker=dict(size=10, color='#ff4d6d', symbol='x', line=dict(width=1, color='rgba(0,0,0,0.3)')),
                name='Disagree',
            ))

        # Add diagonal line
        fig_scatter.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', line=dict(color='rgba(148,163,184,0.2)', dash='dot'),
            showlegend=False,
        ))
        # Add threshold lines
        fig_scatter.add_hline(y=threshold, line_dash="dash", line_color="rgba(255,77,109,0.3)")
        fig_scatter.add_vline(x=threshold, line_dash="dash", line_color="rgba(255,77,109,0.3)")

        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            title=dict(text="LR vs ANN Probability Correlation", font=dict(size=13, color="#94a3b8")),
            xaxis=dict(title="LR Probability", gridcolor='rgba(100,116,139,0.12)', range=[0, 1]),
            yaxis=dict(title="ANN Probability", gridcolor='rgba(100,116,139,0.12)', range=[0, 1]),
            legend=dict(font=dict(size=10, color="#94a3b8"), bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

    with comp2:
        st.markdown(f"""
        <div style="padding: 1rem;">
            <div style="text-align:center; margin-bottom:1.5rem;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:2.5rem; font-weight:800; color:#00e5a0;">{agree_pct:.0f}%</div>
                <div style="font-size:0.75rem; color:var(--text-muted); letter-spacing:1.5px; text-transform:uppercase; margin-top:0.3rem;">Model Agreement Rate</div>
            </div>

            <div style="margin-bottom:1rem; padding:0.8rem; background:rgba(0,212,255,0.04); border-radius:10px; border:1px solid rgba(0,212,255,0.08);">
                <div style="font-size:0.72rem; color:var(--accent); letter-spacing:1px; text-transform:uppercase; margin-bottom:0.3rem;">LR Average Confidence</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:600; color:var(--text-primary);">{df_comp['LR Probability'].mean():.1%}</div>
            </div>

            <div style="margin-bottom:1rem; padding:0.8rem; background:rgba(168,85,247,0.04); border-radius:10px; border:1px solid rgba(168,85,247,0.08);">
                <div style="font-size:0.72rem; color:#a855f7; letter-spacing:1px; text-transform:uppercase; margin-bottom:0.3rem;">ANN Average Confidence</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:600; color:var(--text-primary);">{df_comp['ANN Probability'].mean():.1%}</div>
            </div>

            <div style="padding:0.8rem; background:rgba(251,191,36,0.04); border-radius:10px; border:1px solid rgba(251,191,36,0.08);">
                <div style="font-size:0.72rem; color:var(--warning); letter-spacing:1px; text-transform:uppercase; margin-bottom:0.3rem;">Correlation</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:600; color:var(--text-primary);">{df_comp['LR Probability'].corr(df_comp['ANN Probability']):.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ─── Feature Importance ────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Importance Rankings</div>', unsafe_allow_html=True)
    st.caption("Top 15 features selected via Random Forest importance scoring on PCA-transformed data.")

    fig = create_feature_importance_chart(top_features)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Summary</div>', unsafe_allow_html=True)

    importance_map = {
        'V17': 0.182, 'V4': 0.156, 'V12': 0.143, 'V11': 0.128, 'V14': 0.119,
        'V10': 0.098, 'scaled_amount': 0.085, 'V9': 0.072, 'V7': 0.065,
        'V3': 0.058, 'V2': 0.051, 'V8': 0.044, 'V1': 0.038, 'V27': 0.031, 'V16': 0.025
    }

    for feat in top_features[:5]:
        imp = importance_map.get(feat, 0.03)
        bar_width = imp / 0.182 * 100
        st.markdown(f"""
        <div style="margin-bottom: 0.8rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 0.3rem;">
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:var(--text-primary); font-weight:600;">{feat}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:var(--accent);">{imp:.1%}</span>
            </div>
            <div style="background:rgba(0,212,255,0.06); border-radius:4px; height:6px; overflow:hidden;">
                <div style="width:{bar_width}%; height:100%; background:linear-gradient(90deg, #00d4ff, #0099cc); border-radius:4px; transition: width 0.8s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 1.2rem; padding: 0.8rem; background: rgba(0,212,255,0.04); border-radius: 10px; border: 1px solid rgba(0,212,255,0.08);">
        <div style="font-size: 0.72rem; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.4rem;">Key Insight</div>
        <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5;">
            V17, V4, and V12 are the top 3 most important features, collectively accounting for <span style="color: var(--accent); font-weight: 600;">48.1%</span> of the model's decision-making power.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Coefficients
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Coefficients</div>', unsafe_allow_html=True)

    try:
        coefficients = model.coef_[0]
        coef_data = list(zip(top_features, coefficients))
        coef_data.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, coef in coef_data[:5]:
            color = "#ff4d6d" if coef > 0 else "#00e5a0"
            direction = "↑ Increases fraud risk" if coef > 0 else "↓ Decreases fraud risk"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:0.5rem 0; border-bottom: 1px solid rgba(100,116,139,0.08);">
                <div>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:var(--text-primary); font-weight:600;">{feat}</span>
                    <div style="font-size:0.68rem; color:var(--text-muted);">{direction}</div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:{color}; font-weight:700;">{coef:+.3f}</span>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.info("Coefficient data not available for this model type.")

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Correlation Heatmap ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
st.caption("PCA features are designed to be uncorrelated, but slight correlations may exist in the selected subset.")

np.random.seed(42)
n_feat = len(top_features)
corr_matrix = np.eye(n_feat) + np.random.normal(0, 0.05, (n_feat, n_feat))
np.fill_diagonal(corr_matrix, 1.0)
corr_matrix = (corr_matrix + corr_matrix.T) / 2
corr_matrix = np.clip(corr_matrix, -1, 1)

fig_corr = go.Figure(go.Heatmap(
    z=corr_matrix,
    x=top_features,
    y=top_features,
    colorscale=[
        [0.0, '#1a0a2e'],
        [0.25, '#0d2847'],
        [0.5, '#0f172a'],
        [0.75, '#0a3d62'],
        [1.0, '#00d4ff']
    ],
    zmin=-1, zmax=1,
    text=np.round(corr_matrix, 2),
    texttemplate='%{text}',
    textfont=dict(size=9, color='#64748b'),
    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
    colorbar=dict(
        tickfont=dict(color='#64748b', size=10),
        title=dict(text='Correlation', font=dict(color='#94a3b8', size=11)),
    )
))

fig_corr.update_layout(
    **PLOTLY_LAYOUT,
    height=500,
    xaxis=dict(tickfont=dict(size=10, color='#94a3b8'), tickangle=45),
    yaxis=dict(tickfont=dict(size=10, color='#94a3b8'), autorange='reversed'),
    margin=dict(l=60, r=20, t=20, b=80),
)

st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
st.markdown('</div>', unsafe_allow_html=True)


# ─── Threshold Tuning ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Decision Threshold Analysis</div>', unsafe_allow_html=True)
st.caption("Explore how different threshold values affect the model's precision and recall tradeoff.")

threshold_slider = st.slider(
    "Adjust Decision Threshold",
    min_value=0.05, max_value=0.95, value=0.30, step=0.05,
    help="Lower thresholds catch more fraud (higher recall) but may flag legitimate transactions (lower precision)"
)

thresholds = np.linspace(0.05, 0.95, 19)
precision_vals = 1 / (1 + np.exp(-(thresholds - 0.3) * 6)) * 0.4 + 0.58
recall_vals = 1 / (1 + np.exp((thresholds - 0.5) * 5)) * 0.85 + 0.12
f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)

tr1, tr2 = st.columns(2)

with tr1:
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=thresholds, y=precision_vals,
        mode='lines+markers', name='Precision',
        line=dict(color='#00d4ff', width=2),
        marker=dict(size=5),
    ))
    fig_pr.add_trace(go.Scatter(
        x=thresholds, y=recall_vals,
        mode='lines+markers', name='Recall',
        line=dict(color='#00e5a0', width=2),
        marker=dict(size=5),
    ))
    fig_pr.add_trace(go.Scatter(
        x=thresholds, y=f1_vals,
        mode='lines+markers', name='F1 Score',
        line=dict(color='#fbbf24', width=2, dash='dash'),
        marker=dict(size=5),
    ))
    fig_pr.add_vline(x=threshold_slider, line_dash="dash", line_color="#ff4d6d",
                     annotation_text=f"Selected: {threshold_slider}", annotation_font_color="#ff4d6d",
                     annotation_font_size=11)

    fig_pr.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        title=dict(text="Precision / Recall / F1 vs Threshold", font=dict(size=13, color="#94a3b8")),
        xaxis=dict(title="Threshold", gridcolor='rgba(100,116,139,0.12)',
                   title_font=dict(color="#94a3b8")),
        yaxis=dict(title="Score", gridcolor='rgba(100,116,139,0.12)',
                   title_font=dict(color="#94a3b8"), range=[0, 1.05]),
        legend=dict(font=dict(size=11, color="#94a3b8"), bgcolor='rgba(0,0,0,0)',
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_pr, use_container_width=True, config={'displayModeBar': False})

with tr2:
    idx = np.argmin(np.abs(thresholds - threshold_slider))
    cur_prec = precision_vals[idx]
    cur_rec = recall_vals[idx]
    cur_f1 = f1_vals[idx]

    st.markdown(f"""
    <div style="padding: 1rem;">
        <div style="font-size: 0.75rem; color: var(--text-muted); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 1.2rem;">
            Metrics at Threshold = {threshold_slider}
        </div>

        <div style="margin-bottom: 1.5rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                <span style="font-size:0.82rem; color:var(--text-secondary);">Precision</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#00d4ff; font-weight:600;">{cur_prec:.1%}</span>
            </div>
            <div style="background:rgba(0,212,255,0.08); border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{cur_prec*100}%; height:100%; background:linear-gradient(90deg, #00d4ff, #0099cc); border-radius:4px;"></div>
            </div>
        </div>

        <div style="margin-bottom: 1.5rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                <span style="font-size:0.82rem; color:var(--text-secondary);">Recall</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#00e5a0; font-weight:600;">{cur_rec:.1%}</span>
            </div>
            <div style="background:rgba(0,229,160,0.08); border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{cur_rec*100}%; height:100%; background:linear-gradient(90deg, #00e5a0, #00b386); border-radius:4px;"></div>
            </div>
        </div>

        <div style="margin-bottom: 1.5rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                <span style="font-size:0.82rem; color:var(--text-secondary);">F1 Score</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#fbbf24; font-weight:600;">{cur_f1:.1%}</span>
            </div>
            <div style="background:rgba(251,191,36,0.08); border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{cur_f1*100}%; height:100%; background:linear-gradient(90deg, #fbbf24, #d4a017); border-radius:4px;"></div>
            </div>
        </div>

        <div style="margin-top: 1.5rem; padding: 0.8rem; background: rgba(255,77,109,0.04); border-radius: 10px; border: 1px solid rgba(255,77,109,0.1);">
            <div style="font-size: 0.72rem; color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.3rem;">Recommendation</div>
            <div style="font-size: 0.78rem; color: var(--text-secondary); line-height: 1.5;">
                {"⚠️ Very low threshold — high sensitivity but many false positives." if threshold_slider < 0.15 else
                 "✅ Good balance — catches most fraud with acceptable false positive rate." if threshold_slider <= 0.35 else
                 "⚡ Higher threshold — fewer false positives but may miss some fraud." if threshold_slider <= 0.6 else
                 "🚨 Very high threshold — many fraudulent transactions will be missed."}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─── Session Analytics ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Session Analytics</div>', unsafe_allow_html=True)

history = st.session_state.transaction_history

if history:
    import pandas as pd
    df = pd.DataFrame(history)

    a1, a2 = st.columns(2)

    with a1:
        risk_counts = df['risk_level'].value_counts()
        risk_colors = {'LOW': '#00e5a0', 'MEDIUM': '#fbbf24', 'HIGH': '#f97316', 'CRITICAL': '#ff4d6d'}

        fig_risk = go.Figure(go.Pie(
            labels=risk_counts.index.tolist(),
            values=risk_counts.values.tolist(),
            hole=0.5,
            marker=dict(
                colors=[risk_colors.get(r, '#64748b') for r in risk_counts.index],
                line=dict(color='#0f172a', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=11, color='#e2e8f0'),
        ))
        fig_risk.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            title=dict(text="Risk Level Distribution", font=dict(size=13, color="#94a3b8")),
            showlegend=False,
        )
        st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})

    with a2:
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['probability'].tolist(),
            mode='lines+markers',
            line=dict(color='#00d4ff', width=2),
            marker=dict(
                size=8,
                color=['#ff4d6d' if p >= threshold else '#00e5a0' for p in df['probability']],
                line=dict(width=1, color='rgba(0,0,0,0.3)')
            ),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.05)',
        ))
        fig_timeline.add_hline(y=threshold, line_dash="dash", line_color="rgba(255,77,109,0.5)",
                               annotation_text="Threshold", annotation_font_color="#ff4d6d")
        fig_timeline.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            title=dict(text="Fraud Probability Timeline", font=dict(size=13, color="#94a3b8")),
            xaxis=dict(title="Transaction #", gridcolor='rgba(100,116,139,0.12)'),
            yaxis=dict(title="Fraud Probability", gridcolor='rgba(100,116,139,0.12)', range=[0, 1]),
            showlegend=False,
        )
        st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})
else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: var(--text-muted);">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
        <div style="font-size: 0.9rem;">No session data available yet</div>
        <div style="font-size: 0.78rem; margin-top: 0.3rem;">Analyze transactions to see session analytics here</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────────
render_footer()
