"""
train_model.py
Trains a baseline Logistic Regression and an XGBoost model on the
processed data, evaluates both with imbalance-aware metrics, and saves
the better-performing model plus the exact feature column order it
expects (needed later by the API/dashboard).

Run this after prepare_data.py:
    python src/train_model.py
"""

import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier

PROCESSED_DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/xgb_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    return X, y


def evaluate(name: str, y_test, probs):
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)  # PR-AUC, more informative than accuracy here
    preds = (probs >= 0.5).astype(int)
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC (average precision): {ap:.4f}")
    print(classification_report(y_test, preds))
    return auc


def main():
    X, y = load_data(PROCESSED_DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Baseline: Logistic Regression with class weighting ---
    baseline = LogisticRegression(class_weight="balanced", max_iter=1000)
    baseline.fit(X_train, y_train)
    baseline_probs = baseline.predict_proba(X_test)[:, 1]
    baseline_auc = evaluate("Logistic Regression (baseline)", y_test, baseline_probs)

    # --- Main model: XGBoost with imbalance handling ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="auc",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = evaluate("XGBoost", y_test, xgb_probs)

    # --- Pick the better model ---
    if xgb_auc >= baseline_auc:
        print("\nXGBoost performed better -- saving XGBoost as the final model.")
        final_model = xgb
    else:
        print("\nLogistic Regression performed better -- saving it as the final model.")
        final_model = baseline

    joblib.dump(final_model, MODEL_PATH)

    # Save the exact column order the model expects, so the API/dashboard
    # can build matching input rows later.
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(list(X.columns), f)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved feature column order to {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
