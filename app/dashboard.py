"""
dashboard.py
Simple Streamlit UI simulating a loan officer's screen: enter applicant
details, get a risk score back, with a plain-English explanation.

Run with:
    streamlit run app/dashboard.py
"""

import json
import joblib
import pandas as pd
import streamlit as st

model = joblib.load("models/xgb_model.pkl")
with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

st.set_page_config(page_title="Micro-Loan Risk Scoring Tool")
st.title("Micro-Loan Risk Scoring Tool")
st.write("Enter applicant details to estimate default risk.")

income = st.number_input("Annual Income", min_value=0, value=150000, step=10000)
credit = st.number_input("Loan Amount Requested", min_value=0, value=500000, step=10000)
annuity = st.number_input("Annual Repayment (Annuity)", min_value=0, value=25000, step=1000)
age = st.slider("Age", 18, 75, 35)
years_employed = st.slider("Years Employed", 0, 40, 5)
family_size = st.number_input("Family Size", min_value=1, value=2)
utility_score = st.slider(
    "Utility Payment Consistency (simulated: 0 = always late, 1 = always on time)",
    0.0, 1.0, 0.8
)
recharge_freq = st.slider("Mobile Recharge Frequency per Month (simulated)", 0, 15, 4)

if st.button("Calculate Risk"):
    row = {col: 0 for col in FEATURE_COLUMNS}
    row.update({
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AGE_YEARS": age,
        "YEARS_EMPLOYED": years_employed,
        "CNT_FAM_MEMBERS": family_size,
        "SIM_UTILITY_PAYMENT_SCORE": utility_score,
        "SIM_RECHARGE_FREQ": recharge_freq,
    })
    input_df = pd.DataFrame([row])[FEATURE_COLUMNS]

    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Predicted Default Probability: {prob:.1%}")
    if prob >= 0.5:
        st.error("Risk level: HIGH")
    elif prob >= 0.2:
        st.warning("Risk level: MEDIUM")
    else:
        st.success("Risk level: LOW")
