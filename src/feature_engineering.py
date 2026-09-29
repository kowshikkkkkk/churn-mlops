# ============================================================
# STAGE 4: FEATURE ENGINEERING
# src/feature_engineering.py
#
# Job of this file: add derived features on top of cleaned data.
#
# CRITICAL DESIGN RULE — read this before touching this file:
# This module gets imported and called from TWO places:
#   1. train.py — on the full historical dataset
#   2. app/main.py — on a single live customer at prediction time
#
# Both call the exact same engineer_features() function. This is
# what prevents training-serving skew: there is only ONE place
# these formulas are defined. If a formula ever needs to change,
# it changes here once, and both training and serving pick up
# the change automatically. Do NOT let app/main.py reimplement
# these calculations separately — that's the exact bug that
# broke the previous version of this project.
# ============================================================

import pandas as pd
import numpy as np


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    avg_monthly_spend = Total Charges / Tenure Months

    Handles the tenure=0 edge case (a brand new customer, first
    month, not yet billed a full cycle) by falling back to
    Monthly Charges instead — dividing by zero would otherwise
    produce inf/NaN for every new customer, which is exactly the
    population most at risk of early churn, so we can't afford
    to silently lose or corrupt this feature for them.
    """
    df = df.copy()

    df['avg_monthly_spend'] = np.where(
        df['Tenure Months'] > 0,
        df['Total Charges'] / df['Tenure Months'],
        df['Monthly Charges']
    )

    return df


def add_senior_long_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """
    senior_long_tenure = 1 if the customer is a senior citizen
    AND has been with the company more than 24 months, else 0.

    This is an interaction feature — it captures a customer
    segment (loyal seniors) that neither 'Senior Citizen' nor
    'Tenure Months' alone fully represents. A tree-based model
    (Random Forest, XGBoost) can often learn this interaction on
    its own from the two raw columns, but Logistic Regression
    can't unless it's handed the interaction explicitly, since
    it only learns linear relationships. Including it here means
    all three of our contender models get to use it fairly.

    Assumes 'Senior Citizen' has already been standardized to
    0/1 by preprocessing.py — this function will raise a clear
    error rather than silently produce wrong results if it
    hasn't been.
    """
    df = df.copy()

    unique_vals = set(df['Senior Citizen'].unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Expected 'Senior Citizen' to be 0/1 (already standardized by "
            f"preprocessing.py), found: {unique_vals}. Run preprocessing "
            f"before feature_engineering."
        )

    df['senior_long_tenure'] = (
        (df['Senior Citizen'] == 1) & (df['Tenure Months'] > 24)
    ).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering steps in sequence.

    Input: cleaned data from preprocessing.clean_data() —
    numeric Total Charges, numeric Tenure Months, Senior
    Citizen already 0/1.

    Output: same data plus 'avg_monthly_spend' and
    'senior_long_tenure' columns.
    """
    print("=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)

    df = add_avg_monthly_spend(df)
    df = add_senior_long_tenure(df)

    print(f"✅ Added 'avg_monthly_spend' and 'senior_long_tenure'")
    print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    from data_ingestion import load_raw_data
    from data_validation import run_all_validations
    from preprocessing import clean_data

    RAW_PATH = "data/raw/Telco_customer_churn.xlsx"

    df = load_raw_data(RAW_PATH)
    run_all_validations(df)
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)

    print(f"\nNew columns: {['avg_monthly_spend', 'senior_long_tenure']}")
    print(f"\nSample of new features:\n{df_features[['Tenure Months', 'Total Charges', 'Monthly Charges', 'avg_monthly_spend', 'Senior Citizen', 'senior_long_tenure']].head(5)}")