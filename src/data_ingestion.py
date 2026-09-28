# ============================================================
# STAGE 1: DATA INGESTION
# src/data_ingestion.py
#
# Job of this file: load the raw file, confirm it loaded
# correctly, and report basic facts about it.
#
# What this file does NOT do:
# - Drop columns
# - Fix data types
# - Handle missing values
# - Any business logic
#
# Why: data_validation.py (next stage) needs to inspect the
# data exactly as it arrived, before anything touches it.
# If ingestion quietly "helped" by cleaning something, we'd
# lose the ability to validate the real raw data.
# ============================================================

import pandas as pd
import os


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw data file and return it completely untouched.

    Supports both .csv and .xlsx, since the IBM Telco dataset
    ships as .xlsx while other churn datasets ship as .csv.
    Reading .xlsx requires the 'openpyxl' package installed.

    Raises a clear error immediately if the file is missing,
    rather than letting a downstream step fail with a confusing
    KeyError or crash several stages later.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw data file not found at: {filepath}\n"
            f"Expected the Telco churn data file to be placed here before "
            f"running the pipeline."
        )

    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(
            f"Unsupported file type: {filepath}. Expected .csv or .xlsx"
        )

    if df.empty:
        raise ValueError(f"Loaded file at {filepath} but it has 0 rows.")

    print(f"✅ Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    RAW_PATH = "data/raw/Telco_customer_churn.xlsx"
    df = load_raw_data(RAW_PATH)
    print(f"\nFirst 3 rows:\n{df.head(3)}")