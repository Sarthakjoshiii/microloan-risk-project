# Micro-Loan Default Risk Scoring — Step-by-Step Tutorial (Beginner-Friendly)

This tutorial builds a complete project: predicting whether someone will default on a small loan, using data that mimics "alternative data" for people without formal credit history. Every step includes code and a plain-English explanation of *why* you're doing it.

---

## Step 0: What you'll build, in one sentence

A model that looks at an applicant's data (income, payment history, etc.) and outputs: "This person has a 12% chance of defaulting, and here's why."

---

## Step 1: Set up your project folder

Open a terminal and run:

```bash
mkdir microloan-risk-project
cd microloan-risk-project
python -m venv venv
```

Activate it:
```bash
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

Install the libraries you'll need:
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn imbalanced-learn jupyter fastapi uvicorn streamlit joblib
```

**Why a virtual environment?** It keeps this project's libraries separate from everything else on your computer, so nothing conflicts. Think of it as a clean room just for this project.

Create these folders inside your project:
```bash
mkdir data notebooks src models app report
```

- `data/` → your CSV files go here
- `notebooks/` → where you explore and experiment
- `src/` → clean, reusable Python scripts
- `models/` → saved trained models
- `app/` → your FastAPI + Streamlit code
- `report/` → your final write-up

---

## Step 2: Get the dataset

1. Go to Kaggle and search **"Home Credit Default Risk"**.
2. Download `application_train.csv` (this is the main file — start with just this one, ignore the other files for now, they add complexity you don't need yet).
3. Put it inside your `data/` folder.

**What is this dataset?** Each row is one loan applicant. Columns include income, family status, employment length, etc. There's a column called `TARGET`: `1` means the person had payment difficulty (defaulted), `0` means they repaid fine. This is what you're trying to predict.

---

## Step 3: Look at your data (EDA)

Open Jupyter:
```bash
jupyter notebook
```

Create a new notebook inside `notebooks/` called `01_eda.ipynb`. Run this:

```python
import pandas as pd

df = pd.read_csv('../data/application_train.csv')

print(df.shape)          # how many rows and columns
print(df['TARGET'].value_counts(normalize=True))  # how balanced is the target?
df.head()
```

**What you're checking:** `TARGET` will show something like 92% zeros, 8% ones. This means defaults are rare — this is called **class imbalance**, and it's the single biggest thing that makes this project non-trivial. Remember this number; you'll deal with it in Step 6.

Now check for missing data:
```python
missing = df.isnull().mean().sort_values(ascending=False)
print(missing.head(20))
```

**Why check missing values?** Some columns might be 60-70% empty — you'll need to decide whether to fill them in or drop them. Don't fix this yet, just know what you're dealing with.

Look at a few relationships:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df[df['AMT_INCOME_TOTAL'] < 1000000])
plt.title('Income vs Default')
plt.show()
```

**Why plot this?** You're building intuition — does income actually seem to relate to default? Do this for 3-4 columns you find interesting (`DAYS_EMPLOYED`, `AMT_CREDIT`, `NAME_EDUCATION_TYPE`). Don't overdo this step — half a day max.

---

## Step 4: Pick a manageable set of features

The real dataset has 120+ columns. As a beginner, don't use all of them — pick 15-20 that make sense. Create `notebooks/02_features.ipynb`:

```python
features = [
    'AMT_INCOME_TOTAL',      # applicant's income
    'AMT_CREDIT',            # loan amount requested
    'AMT_ANNUITY',           # yearly payment amount
    'DAYS_BIRTH',            # age (negative days, we'll fix this)
    'DAYS_EMPLOYED',         # how long employed (negative days)
    'CNT_FAM_MEMBERS',       # family size
    'NAME_EDUCATION_TYPE',   # education level
    'NAME_INCOME_TYPE',      # job type
    'CODE_GENDER',           # gender
    'FLAG_OWN_CAR',          # owns a car? (proxy for asset ownership)
    'FLAG_OWN_REALTY',       # owns property?
]

df_model = df[features + ['TARGET']].copy()
```

**Why start small?** A simple model with 15 good features you understand beats a complex model with 120 features you don't. You can always add more later.

Clean up a couple of odd columns:
```python
# DAYS_BIRTH and DAYS_EMPLOYED are negative (days before today) - convert to positive years
df_model['AGE_YEARS'] = -df_model['DAYS_BIRTH'] / 365
df_model['YEARS_EMPLOYED'] = -df_model['DAYS_EMPLOYED'] / 365
df_model.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
```

**Why?** `DAYS_EMPLOYED = -2000` is confusing to read. `YEARS_EMPLOYED = 5.5` is intuitive. Small readability fixes like this matter for clean feature engineering.

### Add your "alternative data" features (this is your project's original contribution)

Since real unbanked-user data (mobile recharge, utility bills) isn't public, simulate a couple of features so you can talk about this concept in your report — be upfront that these are simulated:

```python
import numpy as np
np.random.seed(42)

# Simulated: consistency of utility bill payments (0 = always late, 1 = always on time)
df_model['SIM_UTILITY_PAYMENT_SCORE'] = np.random.beta(5, 2, size=len(df_model))

# Simulated: mobile recharge frequency per month
df_model['SIM_RECHARGE_FREQ'] = np.random.poisson(4, size=len(df_model))
```

**Why simulate instead of skip?** This is the whole point of your project — showing you understand what "alternative data for the unbanked" means, even if you can't get the real dataset. Just be honest about it in your report; don't pretend it's real.

Handle categorical columns (text → numbers, since models need numbers):
```python
df_model = pd.get_dummies(df_model, columns=[
    'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'CODE_GENDER',
    'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'
], drop_first=True)
```

**What does `get_dummies` do?** Turns a column like `Gender: Male/Female` into two columns of 0s and 1s, because ML models can't read text directly.

Handle missing values simply:
```python
df_model.fillna(df_model.median(numeric_only=True), inplace=True)
```

**Why median, not mean?** Median is less thrown off by extreme outliers (like one applicant with a huge income).

Save your cleaned data:
```python
df_model.to_csv('../data/processed_data.csv', index=False)
```

---

## Step 5: Split your data

New notebook: `03_modeling.ipynb`

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df_model = pd.read_csv('../data/processed_data.csv')

X = df_model.drop('TARGET', axis=1)
y = df_model['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Why `stratify=y`?** This makes sure both your train and test sets have the same ~8% default rate. Without it, you could accidentally end up with a test set that has almost no defaults, making evaluation meaningless.

---

## Step 6: Handle the imbalance problem

Remember: only ~8% of applicants default. If your model just predicted "no default" for everyone, it'd be 92% accurate — and completely useless. This is why **accuracy is the wrong metric** for this project.

Train a baseline model first, using class weighting (a simple, effective fix):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

baseline = LogisticRegression(class_weight='balanced', max_iter=1000)
baseline.fit(X_train, y_train)

preds = baseline.predict(X_test)
probs = baseline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
```

**What does `class_weight='balanced'` do?** It tells the model "treat mistakes on the rare class (defaulters) as more costly than mistakes on the common class." Without this, the model would mostly ignore defaulters since they're rare.

**What is ROC-AUC?** A single number (0.5 to 1.0) measuring how well your model separates defaulters from non-defaulters, regardless of threshold. 0.5 = random guessing, 1.0 = perfect. Aim for 0.65-0.75 with this simple feature set — that's a respectable, honest result.

---

## Step 7: Try a stronger model (XGBoost)

```python
from xgboost import XGBClassifier

# scale_pos_weight roughly = (number of non-defaults) / (number of defaults)
scale = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    scale_pos_weight=scale,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    eval_metric='auc',
    random_state=42
)
xgb.fit(X_train, y_train)

xgb_probs = xgb.predict_proba(X_test)[:, 1]
print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgb_probs))
```

**Why XGBoost over Logistic Regression?** XGBoost captures non-linear patterns (e.g., "risk is high ONLY when income is low AND employment is short") that a simple linear model can't. It usually wins on this kind of tabular data.

**Why `scale_pos_weight`?** Same idea as `class_weight='balanced'` before, just XGBoost's version of the same fix.

Compare both models' ROC-AUC — whichever is higher is your main model going forward.

---

## Step 8: Make it explainable (SHAP)

A risk score alone isn't enough — you need to explain *why* someone was flagged as risky. This matters for real lending (regulations require it) and it'll impress interviewers.

```python
import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Global importance: which features matter most overall?
shap.summary_plot(shap_values, X_test)
```

**What does this plot show?** Each dot is one applicant. Features are ranked top-to-bottom by importance. Red dots on the right mean "high value of this feature pushed the prediction toward default."

Explain one specific applicant's prediction:
```python
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)
```

**Why look at just one applicant?** This is what a loan officer would actually see: "This specific person was flagged risky mainly because of X and Y." That's the explainability piece regulators and businesses care about.

---

## Step 9: Basic fairness check

```python
# Example: check approval rates by gender
X_test_with_preds = X_test.copy()
X_test_with_preds['predicted_default_prob'] = xgb_probs
X_test_with_preds['CODE_GENDER_M'] = X_test['CODE_GENDER_M']  # from your dummy columns

print(X_test_with_preds.groupby('CODE_GENDER_M')['predicted_default_prob'].mean())
```

**Why check this?** If one group's average predicted risk is drastically different without a good reason, your model might have picked up unfair bias from the data. You don't need to "fix" this for a student project — just noticing it and discussing it in your report shows real maturity.

---

## Step 10: Save your model

```python
import joblib
joblib.dump(xgb, '../models/xgb_model.pkl')
```

---

## Step 11: Build a simple API (FastAPI)

Create `app/main.py`:

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('../models/xgb_model.pkl')

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    return {"default_probability": float(prob)}
```

Run it:
```bash
uvicorn app.main:app --reload
```

**What does this do?** Turns your model into a live service. Anyone (or any app) can send applicant data to this API and get a risk score back — this is how models actually get used in real products.

---

## Step 12: Build a simple front-end (Streamlit)

Create `app/dashboard.py`:

```python
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('../models/xgb_model.pkl')

st.title("Micro-Loan Risk Scoring Tool")

income = st.number_input("Applicant Income", value=150000)
credit = st.number_input("Loan Amount Requested", value=500000)
age = st.number_input("Age (years)", value=35)

if st.button("Calculate Risk"):
    # build the same feature format your model was trained on
    input_df = pd.DataFrame([{
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': credit,
        'AGE_YEARS': age,
        # ... fill in remaining features with reasonable defaults
    }])
    prob = model.predict_proba(input_df)[0][1]
    st.write(f"Predicted Default Probability: {prob:.2%}")
```

Run it:
```bash
streamlit run app/dashboard.py
```

**Why bother with this?** A working demo you can click through in an interview is far more memorable than a Jupyter notebook nobody will open.

---

## Step 13: Write your report

Structure it like this:
1. **Problem statement** — why unbanked users get excluded from credit
2. **Data** — what you used, and honesty about simulated features
3. **Methodology** — feature engineering, imbalance handling, model choice
4. **Results** — ROC-AUC, SHAP plots, fairness observations
5. **Limitations** — be upfront (proxy dataset, simulated features, small feature set)
6. **Impact** — who this could help in the real world

---

## Quick troubleshooting tips

- **Model AUC is very low (~0.5)?** Check you didn't accidentally leave `TARGET` inside your features (`X`).
- **`get_dummies` mismatch between train/test?** Always fit your encoding on the full dataset before splitting, or save the column list and reindex your test set to match.
- **XGBoost complains about column types?** Make sure everything is numeric — check with `X_train.dtypes`.

---

You now have: cleaned data → a trained, imbalance-aware model → SHAP explainability → a basic fairness check → a working API + dashboard → a report outline. That's a complete, defensible end-to-end project.
