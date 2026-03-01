# ============================================================
# DATA PREPARATION PIPELINE
# src/data_prep.py
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(filepath):
    """Load raw data and return dataframe."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    return df

# ============================================================
# 2. CLEAN DATA
# ============================================================

def clean_data(df):
    """
    Clean raw data:
    - Drop columns with no predictive value
    - Fix data types
    - Convert target to binary
    """
    df = df.copy()

    # Drop identifier and location columns — no predictive value
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City',
        'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
        'Churn Label', 'Churn Score', 'Churn Reason', 'CLTV'
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    # Fix TotalCharges — may have spaces
    df['Total Charges'] = pd.to_numeric(
        df['Total Charges'], errors='coerce'
    )

    # Fill nulls with median
    df['Total Charges'] = df['Total Charges'].fillna(
    df['Total Charges'].median()
)

    # Target is Churn Value — already binary (0/1)
    # Rename for clarity
    df.rename(columns={'Churn Value': 'Churn'}, inplace=True)

    print(f"\nAfter cleaning : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn rate     : {df['Churn'].mean():.2%}")
    print(f"Missing values : {df.isnull().sum().sum()}")

    return df

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Create new features:
    - tenure_group: bucket tenure into groups
    - avg_monthly_spend: total charges / tenure
    - senior_long_tenure: senior with long tenure flag
    """
    df = df.copy()

    # Tenure groups
    df['tenure_group'] = pd.cut(
        df['Tenure Months'],
        bins   = [0, 12, 24, 48, 72],
        labels = ['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    )

    # Average monthly spend
    df['avg_monthly_spend'] = np.where(
        df['Tenure Months'] > 0,
        df['Total Charges'] / df['Tenure Months'],
        df['Monthly Charges']
    )

    # Senior citizen with long tenure flag
    df['senior_long_tenure'] = (
        (df['Senior Citizen'] == 'Yes') &
        (df['Tenure Months'] > 24)
    ).astype(int)

    print(f"\nFeatures after engineering: {df.shape[1]} columns")
    return df

# ============================================================
# 4. ENCODE CATEGORICAL VARIABLES
# ============================================================

def encode_features(df):
    """
    Encode categorical variables:
    - Binary yes/no columns to 0/1
    - Multi-category columns to one-hot encoding
    """
    df = df.copy()

    # Binary yes/no columns
    binary_cols = [
        'Phone Service', 'Paperless Billing',
        'Partner', 'Dependents'
    ]
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # Senior Citizen — Yes/No to 1/0
    df['Senior Citizen'] = (df['Senior Citizen'] == 'Yes').astype(int)

    # Drop tenure_group — tenure numeric captures this
    df.drop(columns=['tenure_group'], inplace=True)

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"\nOne-hot encoding: {cat_cols}")

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"Final feature count: {df.shape[1]} columns")
    return df

# ============================================================
# 5. SPLIT AND SAVE
# ============================================================

def split_and_save(df, output_dir):
    """Split into train/test and save to processed folder."""

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y
    )

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv',   index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv',   index=False)

    print(f"\nData saved to    : {output_dir}")
    print(f"Train size       : {X_train.shape[0]} rows")
    print(f"Test size        : {X_test.shape[0]} rows")
    print(f"Features         : {X_train.shape[1]} columns")
    print(f"Train churn rate : {y_train.mean():.2%}")
    print(f"Test churn rate  : {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test

# ============================================================
# 6. MAIN — Run the full pipeline
# ============================================================

if __name__ == "__main__":

    RAW_PATH   = 'data/raw/Telco-customer-churn.csv'
    OUTPUT_DIR = 'data/processed'

    print("="*50)
    print("DATA PREPARATION PIPELINE")
    print("="*50)

    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)
    split_and_save(df, OUTPUT_DIR)

    print("\n✅ Data preparation complete.")