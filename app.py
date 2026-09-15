import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Bank Transaction Classification",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .result-banner {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
    .result-number {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💳 Bank Transaction Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classifies banking transactions into categories (Shopping, Bills, Groceries, Entertainment, Salary) using <b>RandomForestClassifier</b>.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "models/bank_transaction_classifier.pkl"
csv_path = BASE_DIR / "data/bank_transactions.csv"
chart_path = BASE_DIR / "outputs/feature_importance.png"

if not model_path.exists():
    st.error("Model file not found! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Live Transaction Classifier", "📈 Feature Importance & Metrics", "📋 Dataset Preview"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Transaction Parameters")
        with st.form("tx_form"):
            amount = st.number_input("Transaction Amount (₹)", min_value=1.0, max_value=500000.0, value=1250.0, step=50.0)
            hour = st.slider("Transaction Hour (0-23)", 0, 23, 18)
            weekend = 1 if st.checkbox("Is Weekend Transaction?", False) else 0
            balance = st.number_input("Account Balance (₹)", min_value=0.0, max_value=2000000.0, value=45000.0, step=1000.0)
            merchant = st.selectbox("Merchant Type Code", [1, 2, 3, 4, 5], format_func=lambda x: f"Merchant Type {x}")
            channel = st.selectbox("Transaction Channel", [1, 2, 3], format_func=lambda x: {1: "Online / UPI (Channel 1)", 2: "POS Terminal (Channel 2)", 3: "ATM / Cash (Channel 3)"}.get(x))
            
            submit_btn = st.form_submit_button("🚀 Classify Transaction", use_container_width=True)
            
    with col_result:
        st.subheader("Classification Outcome")
        if submit_btn:
            tx_data = pd.DataFrame([{
                "amount": amount,
                "transaction_hour": hour,
                "is_weekend": weekend,
                "account_balance": balance,
                "merchant_type": merchant,
                "transaction_channel": channel
            }])
            
            prediction = model.predict(tx_data)[0]
            probability = model.predict_proba(tx_data).max() if hasattr(model, "predict_proba") else 1.0
            
            st.markdown(f"""
            <div class="result-banner">
                <div style="font-size: 0.95rem; opacity: 0.9;">Predicted Category</div>
                <div class="result-number">{prediction}</div>
                <div style="font-size: 1.1rem; font-weight: 600;">Prediction Confidence: {probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Transaction successfully classified!")
            
            with st.expander("🔍 Transaction Input Payload"):
                st.json(tx_data.to_dict(orient="records")[0])
        else:
            st.info("👈 Set the transaction values and click **'Classify Transaction'**.")

with tab2:
    st.subheader("Feature Importance")
    if chart_path.exists():
        st.image(str(chart_path), caption="Random Forest Feature Importance", use_container_width=True)
    else:
        st.info("Feature importance plot will appear after running train_model.py")

with tab3:
    st.subheader("Training Dataset (bank_transactions.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.write(f"Total Records: **{len(df):,}** | Columns: **{len(df.columns)}**")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.warning("Dataset not found.")
