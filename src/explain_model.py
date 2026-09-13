"""
explain_model.py
Loads the trained model, generates SHAP explainability plots (global
and for one example applicant), and runs a basic fairness check across
gender. Saves plots as PNG files into the report/ folder.

Run this after train_model.py:
    python src/explain_model.py
"""

import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt

PROCESSED_DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/xgb_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"


def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    X = df[feature_columns]
    model = joblib.load(MODEL_PATH)

    # --- SHAP global explanation ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("report/shap_summary.png")
    plt.close()
    print("Saved global SHAP summary plot to report/shap_summary.png")

    # --- SHAP explanation for a single applicant (the first row) ---
    plt.figure()
    shap.force_plot(
        explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=True, show=False
    )
    plt.tight_layout()
    plt.savefig("report/shap_single_applicant.png")
    plt.close()
    print("Saved single-applicant SHAP plot to report/shap_single_applicant.png")

    # --- Basic fairness check across gender ---
    probs = model.predict_proba(X)[:, 1]
    df_check = X.copy()
    df_check["predicted_default_prob"] = probs

    gender_col = [c for c in X.columns if c.startswith("CODE_GENDER_")]
    if gender_col:
        fairness_summary = df_check.groupby(gender_col[0])["predicted_default_prob"].mean()
        print("\nAverage predicted default probability by gender group:")
        print(fairness_summary)
        fairness_summary.to_csv("report/fairness_check_gender.csv")
        print("Saved fairness check to report/fairness_check_gender.csv")
    else:
        print("No gender column found for fairness check -- skipping.")


if __name__ == "__main__":
    main()
