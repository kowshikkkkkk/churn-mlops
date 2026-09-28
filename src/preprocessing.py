# ============================================================
# STAGE 3: PREPROCESSING
# src/preprocessing.py
#
# Job of this file: turn validated-but-messy raw data into
# clean, correctly-typed data — still with human-readable
# categorical values (e.g. "Yes"/"No"), NOT yet one-hot encoded.
#
# Why one-hot encoding does NOT happen here:
# One-hot encoding needs to be FIT only on the training split,
# then applied (not re-fit) to validation and test. If we encode
# here, before the train/val/test split exists, we'd be letting
# the full dataset (including val/test rows) influence which
# categories the encoder learns about. That's a data leakage
# pattern, and it's also what caused the old project's bug where
# main.py had to hand-reimplement encoding logic by hand instead
# of loading one shared, fitted encoder object.
#
# So: preprocessing.py cleans. Splitting + fitting the encoder
# happens later, in train.py, using ONLY the training data.
# ============================================================

import pandas as pd
import numpy as np


# Columns dropped for LEAKAGE — they encode information that
# would not be available at prediction time for a live customer,
# or that comes from another model's own churn prediction.
LEAKAGE_COLUMNS = [
    'Churn Label',   # same target, just as Yes/No text instead of 0/1
    'Churn Score',   # IBM SPSS Modeler's OWN churn prediction — training on
                      # this would mean learning to copy another model,
                      # not learning from real customer behavior
    'Churn Reason',  # only populated for customers who already churned —
                      # unknowable for a live customer being scored
    'CLTV',           # a derived/predicted metric, not a raw customer fact —
                      # excluded because we can't verify it's leakage-free
]

# Columns dropped for having NO GENERALIZABLE SIGNAL — not a
# fairness or leakage issue, just genuinely uninformative:
# - CustomerID: unique per row, nothing to learn from
# - Count: constant (always 1), a reporting artifact
# - Country: constant ("United States" for every row)
# - State: constant ("California" for every row)
# - City / Zip Code / Lat Long / Latitude / Longitude: high-cardinality
#   location data — risks overfitting to specific ZIP codes that
#   won't generalize, especially with only ~7000 rows total
NO_SIGNAL_COLUMNS = [
    'CustomerID', 'Count', 'Country', 'State',
    'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
]

TARGET_COLUMN = 'Churn Value'

# Columns that are Yes/No in the raw data and should become 0/1.
# NOT included: Multiple Lines, Internet Service, Online Security, etc —
# those have a THIRD value like "No internet service", so they need
# full one-hot encoding later, not a simple Yes/No -> 1/0 swap.
BINARY_YES_NO_COLUMNS = [
    'Senior Citizen', 'Partner', 'Dependents',
    'Phone Service', 'Paperless Billing',
]


def drop_unneeded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop leakage columns and no-signal columns, keeping the
    reasons visibly separate (see comments above) rather than
    dumping everything into one unexplained list.
    """
    df = df.copy()

    cols_to_drop = LEAKAGE_COLUMNS + NO_SIGNAL_COLUMNS
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    df = df.drop(columns=existing_cols_to_drop)

    print(f"✅ Dropped {len(LEAKAGE_COLUMNS)} leakage columns: {LEAKAGE_COLUMNS}")
    print(f"✅ Dropped {len(NO_SIGNAL_COLUMNS)} no-signal columns: {NO_SIGNAL_COLUMNS}")
    print(f"   Remaining: {df.shape[1]} columns")

    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Total Charges' arrives as text and has ~11 blank/space rows
    (confirmed in validation). Coerce to numeric, then impute
    blanks with the median.

    Why median, not mean: Total Charges is right-skewed (a few
    long-tenure customers have very high totals), so the median
    is a more representative "typical" value than the mean,
    which gets pulled upward by those outliers.

    Why impute at all, rather than drop 11 rows: 11 rows is a
    trivial fraction of 7043, but there's no strong reason to
    lose them — these are very likely new customers (Total
    Charges blank because they haven't been billed yet, which
    also means Tenure Months is probably 0 for these rows).
    """
    df = df.copy()

    before_nulls = pd.to_numeric(df['Total Charges'], errors='coerce').isnull().sum()

    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    median_value = df['Total Charges'].median()
    df['Total Charges'] = df['Total Charges'].fillna(median_value)

    print(f"✅ Fixed 'Total Charges': {before_nulls} blank values imputed with median ({median_value:.2f})")

    return df


def standardize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert simple Yes/No columns to 1/0. Keeping this separate
    from full one-hot encoding (which happens later, at train
    time) because these columns genuinely only have two states —
    no need to wait for a fitted encoder to handle something
    this simple, and it makes the data easier to inspect while
    we're still in the cleaning stage.
    """
    df = df.copy()

    for col in BINARY_YES_NO_COLUMNS:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({'Yes', 'No'}):
            raise ValueError(
                f"Expected '{col}' to only contain Yes/No, found: {unique_vals}. "
                f"This column may have a third category — check before converting."
            )
        df[col] = (df[col] == 'Yes').astype(int)

    print(f"✅ Standardized {len(BINARY_YES_NO_COLUMNS)} binary columns to 0/1: {BINARY_YES_NO_COLUMNS}")

    return df


def rename_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 'Churn Value' to 'Churn' for a cleaner, shorter name downstream."""
    df = df.copy()
    df = df.rename(columns={TARGET_COLUMN: 'Churn'})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing sequence in order.

    Order matters here:
    1. Drop columns first — no point fixing types on columns
       we're about to throw away.
    2. Fix Total Charges — needs to happen before anything
       downstream assumes it's numeric.
    3. Standardize binaries — independent of the above, but
       kept last among cleaning steps for readability.
    4. Rename target — cosmetic, doesn't depend on anything else.
    """
    print("=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    df = drop_unneeded_columns(df)
    df = fix_total_charges(df)
    df = standardize_binary_columns(df)
    df = rename_target(df)

    print(f"\n✅ Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Nulls remaining: {df.isnull().sum().sum()}")

    return df


if __name__ == "__main__":
    from data_ingestion import load_raw_data
    from data_validation import run_all_validations

    RAW_PATH = "data/raw/Telco_customer_churn.xlsx"

    df = load_raw_data(RAW_PATH)
    run_all_validations(df)
    df_clean = clean_data(df)

    print(f"\nColumns after preprocessing: {df_clean.columns.tolist()}")
    print(f"\nFirst 3 rows:\n{df_clean.head(3)}")