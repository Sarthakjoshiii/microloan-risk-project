"""
main.py
FastAPI service exposing the trained model as a /predict endpoint.

Run with:
    uvicorn app.main:app --reload

Then test at:
    http://127.0.0.1:8000/docs
"""

import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Micro-Loan Default Risk API")

model = joblib.load("models/xgb_model.pkl")
with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)


class ApplicantInput(BaseModel):
    # Minimal set of raw inputs a caller provides.
    # Anything not listed here is filled with a dataset-wide default
    # inside build_feature_row(), so the demo works even with partial input.
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AGE_YEARS: float
    YEARS_EMPLOYED: float
    CNT_FAM_MEMBERS: float = 1
    SIM_UTILITY_PAYMENT_SCORE: float = 0.8
    SIM_RECHARGE_FREQ: float = 4


def build_feature_row(data: ApplicantInput) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}  # default every dummy column to 0
    for key, value in data.dict().items():
        if key in row:
            row[key] = value
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.post("/predict")
def predict(data: ApplicantInput):
    row = build_feature_row(data)
    prob = model.predict_proba(row)[0][1]
    return {
        "default_probability": round(float(prob), 4),
        "risk_level": "high" if prob >= 0.5 else "medium" if prob >= 0.2 else "low",
    }


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Micro-Loan Risk API is running"}
