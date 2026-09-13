# Micro-Loan Default Risk Scoring for Unbanked Users

An end-to-end machine learning project that predicts loan default risk using alternative data signals — built for applicants who lack traditional credit history.

## Problem

Millions of people worldwide are excluded from formal credit because they have no credit bureau history — despite being perfectly capable of repaying a loan. This project explores how alternative behavioral data (utility payments, mobile usage patterns) can be used to assess creditworthiness for this underserved population, using the Home Credit Default Risk dataset as a real-world proxy.

## What this project includes

- **Feature engineering** on applicant financial and demographic data, plus simulated alternative-data signals (utility payment consistency, mobile recharge frequency)
- **Imbalance-aware modeling**: Logistic Regression baseline vs XGBoost, evaluated with ROC-AUC and PR-AUC (not accuracy, since defaults are ~8% of the data)
- **Explainability**: SHAP plots showing global feature importance and individual applicant explanations
- **Fairness check**: a basic audit of predicted risk across demographic groups
- **Deployment**: a FastAPI backend serving live predictions, plus a Streamlit dashboard for interactive use

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit

## Project structure

```
├── src/                # data prep, training, explainability scripts
├── app/                # FastAPI backend + Streamlit dashboard
├── notebooks/          # exploratory data analysis
├── models/             # saved trained model
├── report/             # SHAP plots, fairness check output
└── requirements.txt
```

## How to run

See [SETUP.md](SETUP.md) for full setup and run instructions.

Quick start:
```bash
pip install -r requirements.txt
python src/prepare_data.py
python src/train_model.py
python src/explain_model.py
uvicorn app.main:app --reload      # API
streamlit run app/dashboard.py     # Dashboard
```

## Results

- XGBoost outperformed a Logistic Regression baseline (see terminal output from `train_model.py` for exact ROC-AUC / PR-AUC scores)
- SHAP analysis identifies age and employment length as the strongest predictors of default risk
- A fairness check across gender is included to audit for potential bias in predictions

## Limitations

- The dataset (Home Credit Default Risk, via Kaggle) is a real-world proxy, not literal unbanked-user data
- Alternative-data features (utility payment score, recharge frequency) are simulated for demonstration, since real behavioral data of this kind is proprietary and not publicly available
- Predicted probabilities are useful for relative risk ranking but are not fully calibrated to true default rates

## Dataset

[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk) — download `application_train.csv` and place it in the `data/` folder before running.

## Output
<img width="1010" height="823" alt="image" src="https://github.com/user-attachments/assets/7c186e36-1988-4541-bf91-0766b26d88e1" />
<img width="961" height="397" alt="image" src="https://github.com/user-attachments/assets/668e7b48-d4bb-4c4b-b81c-4f6f7f7aeb16" />

