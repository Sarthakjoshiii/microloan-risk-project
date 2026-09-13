"""
prepare_data.py
Loads the raw Home Credit dataset, selects a manageable set of features,
cleans them, adds simulated 'alternative data' features, and saves a
ready-to-model CSV.

Run this first:
    python src/prepare_data.py
"""

import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/application_train.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"

# Columns we keep from the raw ~120-column dataset.
# Kept deliberately small so the project stays easy to reason about.
SELECTED_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

CATEGORICAL_COLUMNS = [
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df_model = df[SELECTED_FEATURES + ["TARGET"]].copy()

    # DAYS_BIRTH / DAYS_EMPLOYED are stored as negative day counts.
    # Convert to readable positive years.
    df_model["AGE_YEARS"] = -df_model["DAYS_BIRTH"] / 365
    df_model["YEARS_EMPLOYED"] = -df_model["DAYS_EMPLOYED"] / 365
    df_model.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

    # DAYS_EMPLOYED has a known anomaly value (365243) for unemployed/retired
    # applicants, which turns into a huge negative "years employed" after
    # conversion. Cap it at 0 so it doesn't distort the model.
    df_model.loc[df_model["YEARS_EMPLOYED"] < 0, "YEARS_EMPLOYED"] = 0

    return df_model


def add_simulated_alternative_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Adds simulated 'alternative data' features representing signals an
    unbanked applicant might realistically generate (utility payments,
    mobile recharge activity), since the real dataset has no such columns.
    This is clearly a simulation, not real behavioral data -- documented
    here and should be documented again in your project report.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # 0 = always late, 1 = always on time
    df["SIM_UTILITY_PAYMENT_SCORE"] = rng.beta(5, 2, size=n)

    # recharges per month
    df["SIM_RECHARGE_FREQ"] = rng.poisson(4, size=n)

    return df


def encode_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


def main():
    df = load_raw_data(RAW_DATA_PATH)
    df = select_and_clean(df)
    df = add_simulated_alternative_data(df)
    df = encode_and_fill(df)

    print(f"Final processed shape: {df.shape}")
    print(f"Default rate: {df['TARGET'].mean():.2%}")

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
