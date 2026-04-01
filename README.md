# 🛡️ FraudShield

**FraudShield** is an intelligent credit card fraud detection system powered by a dual-model Machine Learning architecture. Built as a comprehensive, real-time dashboard using Streamlit, this project was designed to quickly identify fraudulent transactions while minimizing false positives.

Created for the **ML InnovateX Hackathon**.

## 🚀 Key Features

*   **Dual-Model Ensemble AI:** Leverages both a transparent **Logistic Regression** model and a high-capacity **Artificial Neural Network (ANN)**, ensembling them for robust predictions.
*   **Real-time Analysis:** Analyze single transactions on-the-fly, adjusting over 15 distinct PCA-transformed features.
*   **Live Stream Simulation:** Experience real-time fraud monitoring via an animated live transaction feed dashboard.
*   **Explainable AI (XAI):** Built-in SHAP-style waterfall charts explain *why* the AI flagged a transaction, showing individual feature contributions.
*   **Batch CSV Processing:** Upload bulk sets of transactions and analyze them in an instant, exporting the full results as CSV.
*   **Interactive Analytics:** Deep dive into probability distribution, feature importance (Random Forest derived), model correlation, and Precision/Recall tradeoffs.
*   **Premium Dark UI:** Stunning glassmorphism UI built strictly with Python and custom CSS injections.

## 🛠️ Technology Stack

*   **Python 3.10+**
*   **Streamlit** - Web framework and UI components
*   **TensorFlow / Keras** - ANN architecture and inference
*   **Scikit-Learn** - Logistic Regression, preprocessing, PCA, and metrics
*   **Plotly** - Dynamic, interactive charting and radar graphs
*   **Pandas & NumPy** - Data manipulation and numerical operations

## 🧠 Methodology

1.  **Data:** Utilizing the Credit Card Fraud Detection dataset from Kaggle (ULB). Only highly anonymized PCA features are used to maintain privacy.
2.  **Imbalance Handling:** Handled the extreme class imbalance (only 0.17% fraud) using `RandomOverSampler` and class weights.
3.  **Feature Selection:** Evaluated all 28 PCA features + Amount/Time, narrowing down to the Top 15 most important features based on Random Forest importance.
4.  **Ensemble System:**
    *   **LR (Primary):** Fast, easily explainable statistical baseline.
    *   **ANN (Secondary):** Captures non-linear nuances.
    *   **Ensemble:** 40% LR / 60% ANN weighted average, optimizing for both recall and precision.

## 📦 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nemishsapara69/fraudshield.git
   cd fraudshield
   ```

2. **Install dependencies:**
    *(It is recommended to use a virtual environment)*
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Navigate:** Open your browser to the local URL provided (typically `http://localhost:8501`).

## 👨‍💻 Developer

**Nemish Sapara**  
AI/ML Enthusiast | Data Science | Python  
*ML InnovateX Hackathon Participant*
