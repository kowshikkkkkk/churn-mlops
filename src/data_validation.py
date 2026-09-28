# ============================================================
# STAGE 2: DATA VALIDATION
# src/data_validation.py
#
# Job of this file: inspect the RAW data and confirm it looks
# the way we expect, BEFORE preprocessing touches it.
#
# What this file does NOT do:
# - Fix anything it finds wrong
# - Drop columns, fill nulls, change types
#
# Why: validation's only job is to tell the truth about what
# came in. If it started "helpfully" fixing things, we'd lose
# the ability to catch real upstream data problems — the fix
# would silently mask an issue that maybe should have stopped
# the pipeline instead.
# ============================================================

import pandas as pd


# Columns we expect to exist in the raw IBM Telco file.
# If any of these are missing, something upstream changed
# (wrong file version, corrupted export, schema drift) and
# we want to know immediately — not three stages later when
# preprocessing crashes with a confusing KeyError.
EXPECTED_COLUMNS = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude', 'Gender', 'Senior Citizen',
    'Partner', 'Dependents', 'Tenure Months', 'Phone Service',
    'Multiple Lines', 'Internet Service', 'Online Security',
    'Online Backup', 'Device Protection', 'Tech Support',
    'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing',
    'Payment Method', 'Monthly Charges', 'Total Charges', 'Churn Label',
    'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason'
]

EXPECTED_ROW_COUNT = 7043
ROW_COUNT_TOLERANCE = 0.10  # allow +/- 10% before treating it as suspicious

TARGET_COLUMN = 'Churn Value'
NULL_RATE_WARNING_THRESHOLD = 0.05  # flag any column >5% null


class DataValidationError(Exception):
    """Raised when raw data fails a hard validation check."""
    pass


def validate_shape(df: pd.DataFrame) -> None:
    """
    Check 1: Row and column counts are in the expected range.

    A big drop in rows could mean a partial/corrupted export.
    A big jump could mean duplicated data. Either way, we want
    a human to look before the pipeline proceeds.
    """
    lower_bound = EXPECTED_ROW_COUNT * (1 - ROW_COUNT_TOLERANCE)
    upper_bound = EXPECTED_ROW_COUNT * (1 + ROW_COUNT_TOLERANCE)

    if not (lower_bound <= df.shape[0] <= upper_bound):
        raise DataValidationError(
            f"Row count {df.shape[0]} is outside expected range "
            f"[{lower_bound:.0f}, {upper_bound:.0f}] "
            f"(expected ~{EXPECTED_ROW_COUNT}). "
            f"This suggests a corrupted or partial export — investigate "
            f"before proceeding."
        )

    print(f"✅ Shape check passed: {df.shape[0]} rows, {df.shape[1]} columns")


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Check 2: All expected columns are present.

    Catches schema drift immediately — e.g. if a future export
    of this dataset renames or drops a column, we fail here with
    a clear message instead of crashing deep inside preprocessing.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing:
        raise DataValidationError(
            f"Missing expected columns: {sorted(missing)}. "
            f"The raw data schema has changed — update EXPECTED_COLUMNS "
            f"in data_validation.py only after confirming this is intentional."
        )

    if extra:
        # Not fatal — new columns showing up isn't necessarily broken,
        # but we should know about it rather than silently ignore it.
        print(f"⚠️  Unexpected extra columns found (not failing): {sorted(extra)}")

    print(f"✅ Required columns check passed: all {len(EXPECTED_COLUMNS)} expected columns present")


def validate_total_charges(df: pd.DataFrame) -> None:
    """
    Check 3: Report how many rows have a non-numeric 'Total Charges'.

    We know from prior work that some rows have blank/space values
    here instead of a number. Validation's job is only to REPORT
    this — preprocessing.py is where it actually gets fixed (coerced
    to numeric, imputed). Reporting it here means we always know
    how many rows are affected, rather than that fix happening
    silently and invisibly downstream.
    """
    numeric_version = pd.to_numeric(df['Total Charges'], errors='coerce')
    non_numeric_count = numeric_version.isnull().sum()

    if non_numeric_count > 0:
        pct = non_numeric_count / len(df) * 100
        print(
            f"⚠️  'Total Charges' has {non_numeric_count} non-numeric values "
            f"({pct:.2f}% of rows). These will be coerced and imputed in "
            f"preprocessing — flagging here so the count is visible and tracked."
        )
    else:
        print("✅ 'Total Charges' check passed: all values numeric")


def validate_target_column(df: pd.DataFrame) -> None:
    """
    Check 4: Target column only contains 0 and 1, with no nulls.

    If this fails, something is seriously wrong with the labels
    themselves — this should always hard-fail, never just warn,
    since a broken target silently corrupts every model we train
    on it.
    """
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(f"Target column '{TARGET_COLUMN}' not found.")

    null_count = df[TARGET_COLUMN].isnull().sum()
    if null_count > 0:
        raise DataValidationError(
            f"Target column '{TARGET_COLUMN}' has {null_count} null values. "
            f"Every row must have a known outcome — cannot proceed."
        )

    unique_values = set(df[TARGET_COLUMN].unique())
    if unique_values != {0, 1}:
        raise DataValidationError(
            f"Target column '{TARGET_COLUMN}' should only contain {{0, 1}}, "
            f"found: {unique_values}"
        )

    churn_rate = df[TARGET_COLUMN].mean()
    print(f"✅ Target column check passed: no nulls, values are binary, churn rate = {churn_rate:.2%}")


def validate_null_rates(df: pd.DataFrame) -> None:
    """
    Check 5: Flag any column with an unexpectedly high null rate.

    This doesn't fail the pipeline — high nulls in a column we
    already plan to drop (like 'Churn Reason', which is only
    populated for churned customers) is expected and fine. This
    check just makes sure nothing SURPRISING is null-heavy, so
    a real data quality problem doesn't slip through silently.
    """
    null_rates = df.isnull().mean()
    flagged = null_rates[null_rates > NULL_RATE_WARNING_THRESHOLD]

    if len(flagged) > 0:
        print(f"⚠️  Columns with >{NULL_RATE_WARNING_THRESHOLD:.0%} null rate:")
        for col, rate in flagged.items():
            print(f"     {col}: {rate:.2%} null")
    else:
        print(f"✅ Null rate check passed: no column exceeds {NULL_RATE_WARNING_THRESHOLD:.0%} nulls")


def run_all_validations(df: pd.DataFrame) -> None:
    """
    Run every validation check in sequence.

    Order matters: shape and required-columns checks run first,
    since if those fail, checking anything else about the data
    (like the target column) doesn't make sense — the DataFrame
    isn't even the shape we expect yet.
    """
    print("=" * 50)
    print("DATA VALIDATION")
    print("=" * 50)

    validate_shape(df)
    validate_required_columns(df)
    validate_total_charges(df)
    validate_target_column(df)
    validate_null_rates(df)

    print("=" * 50)
    print("✅ All validation checks passed")
    print("=" * 50)


if __name__ == "__main__":
    from data_ingestion import load_raw_data

    RAW_PATH = "data/raw/Telco_customer_churn.xlsx"
    df = load_raw_data(RAW_PATH)
    run_all_validations(df)