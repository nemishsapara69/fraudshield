import streamlit as st
from utils import inject_css, render_sidebar, render_footer, init_session_state

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — About",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()
render_sidebar()
init_session_state()


# ─── Page Title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">
    <h1>ℹ️ About FraudShield</h1>
    <div class="subtitle">Project overview, methodology, and technical specifications</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Hero Section ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card" style="text-align: center; padding: 2.5rem;">
    <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🛡️</div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 800;
                background: linear-gradient(135deg, #00d4ff 0%, #00e5a0 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
                margin-bottom: 0.5rem;">
        FraudShield
    </div>
    <div style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1.5rem; line-height: 1.6;">
        An intelligent credit card fraud detection system powered by Machine Learning.<br>
        Built for the <span style="color: var(--accent); font-weight: 600;">ML InnovateX Hackathon</span>.
    </div>
    <div class="metric-row">
        <div class="metric-pill">Version <span>2.0 (Dual AI)</span></div>
        <div class="metric-pill">Model 1 <span>Logistic Regression</span></div>
        <div class="metric-pill">Model 2 <span>Artificial Neural Network</span></div>
        <div class="metric-pill">AUC <span>0.9736</span></div>
        <div class="metric-pill">Framework <span>Streamlit</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Project Overview & Developer ───────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.8;">
        <p><strong style="color: var(--text-primary);">FraudShield</strong> is a comprehensive credit card fraud detection system 
        that leverages a dual-model machine learning architecture to identify potentially fraudulent transactions in real-time. 
        The system analyzes PCA-transformed transaction features using a weighted ensemble of Logistic Regression 
        and an Artificial Neural Network, achieving highly accurate and robust predictions.</p>

        <div style="margin-top: 1.2rem;">
            <div style="color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; 
                        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.6rem;">Key Capabilities</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;">
                <div style="padding: 0.6rem; background: rgba(0,212,255,0.04); border-radius: 8px; border: 1px solid rgba(0,212,255,0.06);">
                    🔍 Single transaction analysis (Dual Model)
                </div>
                <div style="padding: 0.6rem; background: rgba(0,229,160,0.04); border-radius: 8px; border: 1px solid rgba(0,229,160,0.06);">
                    📊 Batch CSV processing
                </div>
                <div style="padding: 0.6rem; background: rgba(251,191,36,0.04); border-radius: 8px; border: 1px solid rgba(251,191,36,0.06);">
                    📈 Interactive analytics & Model Comparison
                </div>
                <div style="padding: 0.6rem; background: rgba(168,85,247,0.04); border-radius: 8px; border: 1px solid rgba(168,85,247,0.06);">
                    🧠 SHAP-style Explainable AI (XAI)
                </div>
                <div style="padding: 0.6rem; background: rgba(255,77,109,0.04); border-radius: 8px; border: 1px solid rgba(255,77,109,0.06);">
                    🔴 Real-time Live Stream Simulation
                </div>
                <div style="padding: 0.6rem; background: rgba(0,229,160,0.04); border-radius: 8px; border: 1px solid rgba(0,229,160,0.06);">
                    📥 Export & download results
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Developer Card
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div class="section-header" style="justify-content: center;">Developer</div>
        <div style="width: 80px; height: 80px; border-radius: 50%; 
                    background: linear-gradient(135deg, #00d4ff, #00e5a0);
                    display: flex; align-items: center; justify-content: center;
                    margin: 1rem auto; font-size: 2rem; font-weight: 700; color: #0a0e1a;">
            NS
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 700; 
                    color: var(--text-primary); margin-bottom: 0.3rem;">
            Nemish Sapara
        </div>
        <div style="font-size: 0.78rem; color: var(--accent); font-weight: 500; margin-bottom: 1rem;">
            ML InnovateX Hackathon Participant
        </div>
        <div style="display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap;">
            <span class="metric-pill">🎓 AI/ML</span>
            <span class="metric-pill">🐍 Python</span>
            <span class="metric-pill">📊 Data Science</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hackathon Badge
    st.markdown("""
    <div class="glass-card" style="text-align: center; border-color: rgba(0,212,255,0.15);">
        <div style="font-size: 2rem; margin-bottom: 0.3rem;">🏆</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; 
                    color: var(--accent); letter-spacing: 1px;">ML InnovateX</div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem;">Hackathon 2026</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Methodology ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Methodology & Pipeline</div>', unsafe_allow_html=True)

st.markdown("""
<div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.7;">
    <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 0.5rem;">
        <div style="flex: 1; min-width: 200px; padding: 1rem; background: rgba(0,212,255,0.03); 
                    border-radius: 12px; border: 1px solid rgba(0,212,255,0.08);">
            <div style="color: var(--accent); font-size: 1.2rem; margin-bottom: 0.3rem;">01</div>
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.3rem;">Data Collection</div>
            <div style="font-size: 0.78rem; color: var(--text-muted);">
                Credit Card Fraud Detection dataset from Kaggle (ULB). 
                3,972 transactions with 2 fraud cases identified.
            </div>
        </div>
        <div style="flex: 1; min-width: 200px; padding: 1rem; background: rgba(0,229,160,0.03); 
                    border-radius: 12px; border: 1px solid rgba(0,229,160,0.08);">
            <div style="color: var(--success); font-size: 1.2rem; margin-bottom: 0.3rem;">02</div>
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.3rem;">Preprocessing</div>
            <div style="font-size: 0.78rem; color: var(--text-muted);">
                PCA transformation applied to original features. 
                Amount and Time scaled using StandardScaler for normalization.
            </div>
        </div>
        <div style="flex: 1; min-width: 200px; padding: 1rem; background: rgba(251,191,36,0.03); 
                    border-radius: 12px; border: 1px solid rgba(251,191,36,0.08);">
            <div style="color: var(--warning); font-size: 1.2rem; margin-bottom: 0.3rem;">03</div>
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.3rem;">Imbalance Handling</div>
            <div style="font-size: 0.78rem; color: var(--text-muted);">
                RandomOverSampler used to address extreme class imbalance. 
                Combined with class_weight='balanced' in the model.
            </div>
        </div>
        <div style="flex: 1; min-width: 200px; padding: 1rem; background: rgba(249,115,22,0.03); 
                    border-radius: 12px; border: 1px solid rgba(249,115,22,0.08);">
            <div style="color: #f97316; font-size: 1.2rem; margin-bottom: 0.3rem;">04</div>
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.3rem;">Dual Models</div>
            <div style="font-size: 0.78rem; color: var(--text-muted);">
                Trained both a transparent Logistic Regression model and a 
                high-capacity Artificial Neural Network (Keras).
            </div>
        </div>
        <div style="flex: 1; min-width: 200px; padding: 1rem; background: rgba(168,85,247,0.03); 
                    border-radius: 12px; border: 1px solid rgba(168,85,247,0.08);">
            <div style="color: var(--purple); font-size: 1.2rem; margin-bottom: 0.3rem;">05</div>
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.3rem;">Ensemble & XAI</div>
            <div style="font-size: 0.78rem; color: var(--text-muted);">
                Weighted predictions (40% LR + 60% ANN). Extract feature 
                importance dynamically for Explainable AI (XAI) insights.
            </div>
        </div>
        <div style="flex: 1; min-width: 200px; padding: 1rem; background: rgba(0,212,255,0.03); 
                    border-radius: 12px; border: 1px solid rgba(0,212,255,0.08);">
            <div style="color: var(--accent); font-size: 1.2rem; margin-bottom: 0.3rem;">06</div>
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.3rem;">Deployment</div>
            <div style="font-size: 0.78rem; color: var(--text-muted);">
                Streamlit-based multi-page web application featuring real-time live simulations, 
                explainability, and dark premium UI.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─── Technical Specifications ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

t1, t2 = st.columns(2)

with t1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Specifications</div>', unsafe_allow_html=True)

    specs = [
        ("Architecture", "Ensemble (LR + ANN)"),
        ("Model 1", "Logistic Regression (C=0.1)"),
        ("Model 2", "Keras Sequential ANN"),
        ("Ensemble Strategy", "Weighted (0.4 LR + 0.6 ANN)"),
        ("Features Used", "15 (PCA-selected)"),
        ("Imbalance Strategy", "RandomOverSampler"),
        ("ROC-AUC Score", "0.9736"),
        ("Baseline Threshold", "0.30"),
    ]

    for label, value in specs:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; padding:0.55rem 0; 
                    border-bottom: 1px solid rgba(100,116,139,0.06);">
            <span style="font-size:0.82rem; color:var(--text-muted);">{label}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; 
                        color:var(--text-primary); font-weight:500;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with t2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Technology Stack</div>', unsafe_allow_html=True)

    tech_stack = [
        ("🐍", "Python 3.10+", "Core programming language"),
        ("📊", "Streamlit", "Web application framework"),
        ("🤖", "TensorFlow & Keras", "Deep learning framework"),
        ("🧠", "Scikit-learn", "Statistical machine learning"),
        ("📈", "Plotly", "Interactive charting & visualizations"),
        ("🔢", "NumPy & Pandas", "Data structures & manipulation"),
    ]

    for icon, name, desc in tech_stack:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.8rem; padding:0.55rem 0; 
                    border-bottom: 1px solid rgba(100,116,139,0.06);">
            <span style="font-size:1.1rem;">{icon}</span>
            <div>
                <div style="font-size:0.82rem; color:var(--text-primary); font-weight:600;">{name}</div>
                <div style="font-size:0.7rem; color:var(--text-muted);">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Dataset Info ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Dataset Information</div>', unsafe_allow_html=True)

st.markdown("""
<div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.7;">
    <p><strong style="color: var(--text-primary);">Source:</strong> 
    <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud" target="_blank" 
       style="color: var(--accent); text-decoration: none;">
        Credit Card Fraud Detection — Kaggle (Université Libre de Bruxelles)
    </a></p>
    
    <p>The dataset contains transactions made by European credit cardholders 
    in September 2013. It presents transactions that occurred in two days, 
    where we have 492 frauds out of 284,807 transactions (in the full dataset).</p>
    
    <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(251,191,36,0.04); 
                border-radius: 10px; border: 1px solid rgba(251,191,36,0.1);">
        <div style="font-size: 0.72rem; color: var(--warning); letter-spacing: 1px; 
                    text-transform: uppercase; margin-bottom: 0.3rem;">⚠️ Disclaimer</div>
        <div style="font-size: 0.78rem; color: var(--text-muted); line-height: 1.5;">
            This model was trained on a subset of 3,972 transactions with only 2 fraud cases. 
            Results demonstrate the methodology and approach. Production deployment would 
            require a significantly larger and more balanced fraud dataset.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────────
render_footer()
