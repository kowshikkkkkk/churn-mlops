# ============================================================
# STAGE 5: TRAINING
# src/train.py
#
# Job of this file:
#   1. Run the full pipeline (ingest -> validate -> preprocess
#      -> engineer features) to get clean, feature-complete data
#   2. Split into train/val/test (70/15/15, stratified)
#   3. Fit a preprocessor (one-hot encoding + scaling) on
#      TRAINING DATA ONLY, then apply it to val/test
#   4. Train three contender models: Logistic Regression,
#      Random Forest, XGBoost
#   5. Log everything to MLflow: params, validation metrics,
#      the trained model, AND the fitted preprocessor
#
# Why the preprocessor gets saved as an artifact, not just the
# model: app/main.py needs to transform a live customer's raw
# input EXACTLY the way training data was transformed. Loading
# the same fitted preprocessor object (instead of hand-writing
# encoding logic again in the API, like the old project did)
# is what guarantees training and serving never drift apart.
# ============================================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score)

from data_ingestion import load_raw_data
from data_validation import run_all_validations
from preprocessing import clean_data
from feature_engineering import engineer_features

RAW_PATH = "data/raw/Telco_customer_churn.xlsx"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42


# ============================================================
# 1. BUILD THE FULL DATASET (stages 1-4 chained together)
# ============================================================

def build_dataset() -> pd.DataFrame:
    """Run ingestion through feature engineering, return one clean dataframe."""
    df = load_raw_data(RAW_PATH)
    run_all_validations(df)
    df = clean_data(df)
    df = engineer_features(df)
    return df


# ============================================================
# 2. THREE-WAY SPLIT (70 / 15 / 15, stratified on target)
# ============================================================

def split_data(df: pd.DataFrame):
    """
    Split into train/val/test.

    Two-step split because sklearn's train_test_split only
    splits into two pieces at a time:
      Step 1: carve off 15% as test
      Step 2: from the remaining 85%, carve off enough to leave
              15% of the ORIGINAL total as validation
              (0.15 / 0.85 ≈ 0.1765 of the remaining 85%)

    stratify=y on both splits keeps the ~26.5% churn rate
    consistent across all three sets — without this, a random
    split could accidentally give val/test sets with a
    meaningfully different churn rate, making model comparison
    across train/val misleading.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"✅ Split: train={X_train.shape[0]} ({y_train.mean():.2%} churn), "
          f"val={X_val.shape[0]} ({y_val.mean():.2%} churn), "
          f"test={X_test.shape[0]} ({y_test.mean():.2%} churn)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 3. PREPROCESSOR — fit on TRAIN ONLY
# ============================================================

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    One ColumnTransformer that does both jobs at once:
      - OneHotEncoder on categorical columns
      - StandardScaler on numeric columns

    handle_unknown='ignore' on the encoder means if a live
    customer arrives with a category value never seen in
    training (e.g. a new payment method added later), the
    encoder won't crash — it just encodes that customer as
    all-zeros for that column, rather than the whole prediction
    request failing.

    drop='first' avoids the "dummy variable trap" (one category
    per feature is redundant since it's implied when all others
    are 0) — same effect as the old project's pd.get_dummies(drop_first=True),
    but as a persisted, reusable object instead of hand-derived logic.
    """
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude='object').columns.tolist()

    print(f"✅ Preprocessor: {len(categorical_cols)} categorical columns, "
          f"{len(numeric_cols)} numeric columns")

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols),
            ('num', StandardScaler(), numeric_cols),
        ]
    )

    preprocessor.fit(X_train)

    return preprocessor


# ============================================================
# 4. EVALUATE ON VALIDATION SET
# ============================================================

def evaluate_model(model, X_val_transformed, y_val) -> dict:
    """Compute the metrics we'll use to compare the three contenders."""
    y_prob = model.predict_proba(X_val_transformed)[:, 1]
    y_pred = model.predict(X_val_transformed)

    return {
        'auc': round(roc_auc_score(y_val, y_prob), 4),
        'f1': round(f1_score(y_val, y_pred), 4),
        'precision': round(precision_score(y_val, y_pred), 4),
        'recall': round(recall_score(y_val, y_pred), 4),
    }


# ============================================================
# 5. TRAIN + LOG ONE MODEL TO MLFLOW
# ============================================================

def train_and_log_model(model, params, model_name,
                         X_train_transformed, y_train,
                         X_val_transformed, y_val,
                         preprocessor):
    """
    Train one model, evaluate it on validation, and log
    everything to MLflow: params, metrics, the model itself,
    and the shared preprocessor artifact.
    """
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train_transformed, y_train)

        metrics = evaluate_model(model, X_val_transformed, y_val)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # MLflow defaults to 'skops' serialization, which audits models
        # for potentially unsafe types before trusting them (a real
        # security feature — pickle files CAN execute arbitrary code
        # if loaded from an untrusted source). It flags tree-based
        # models (RandomForest, XGBoost) by default. Since we trained
        # this model ourselves, moments ago, on our own machine, we
        # explicitly trust it and use standard pickle serialization
        # instead of triggering that audit.
        mlflow.sklearn.log_model(model, "model", serialization_format="pickle")

        # Save + log the preprocessor once per run so this run is
        # fully self-contained (model + the exact transformer that
        # matches it, bundled together in MLflow).
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, 'models/preprocessor.joblib')
        mlflow.log_artifact('models/preprocessor.joblib')

        print(f"\n{'='*45}")
        print(f"Run: {model_name}")
        print(f"{'='*45}")
        for k, v in params.items():
            print(f"  {k:<25}: {v}")
        print(f"  ---")
        for k, v in metrics.items():
            print(f"  {k:<25}: {v}")

        return metrics


# ============================================================
# 6. MAIN — build data, split, preprocess, train 3 contenders
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("TRAINING PIPELINE")
    print("=" * 50)

    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    # Save test set (still untransformed — evaluate.py will transform
    # it itself using the saved preprocessor, keeping test data
    # completely separate from anything used during training/tuning)
    os.makedirs('data/processed', exist_ok=True)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    print(f"✅ Held-out test set saved to data/processed/ (untouched until evaluate.py)")

    # Class imbalance handling — different mechanism per model type:
    # LR/RF use class_weight='balanced' (built into sklearn).
    # XGBoost has no class_weight param; it uses scale_pos_weight,
    # the ratio of negative-to-positive examples in training data.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # ── Contender 1: Logistic Regression (linear baseline) ──
    train_and_log_model(
        model=LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        params={'model_type': 'LogisticRegression', 'C': 1.0,
                'max_iter': 1000, 'class_weight': 'balanced'},
        model_name='LogisticRegression',
        X_train_transformed=X_train_transformed, y_train=y_train,
        X_val_transformed=X_val_transformed, y_val=y_val,
        preprocessor=preprocessor
    )

    # ── Contender 2: Random Forest (bagging ensemble) ──
    train_and_log_model(
        model=RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced', random_state=RANDOM_STATE),
        params={'model_type': 'RandomForest', 'n_estimators': 200,
                'max_depth': 10, 'class_weight': 'balanced'},
        model_name='RandomForest',
        X_train_transformed=X_train_transformed, y_train=y_train,
        X_val_transformed=X_val_transformed, y_val=y_val,
        preprocessor=preprocessor
    )

    # ── Contender 3: XGBoost (boosting ensemble) ──
    # n_jobs=1 is deliberate, not a performance oversight: XGBoost's
    # default multi-threaded training can produce slightly different
    # results run-to-run even with random_state fixed, because
    # floating-point summation order depends on thread scheduling.
    # Single-threaded execution removes that non-determinism source,
    # which matters for genuinely reproducible training.
    train_and_log_model(
        model=XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            eval_metric='logloss', n_jobs=1),
        params={'model_type': 'XGBoost', 'n_estimators': 200, 'max_depth': 5,
                'learning_rate': 0.1, 'scale_pos_weight': round(scale_pos_weight, 3)},
        model_name='XGBoost',
        X_train_transformed=X_train_transformed, y_train=y_train,
        X_val_transformed=X_val_transformed, y_val=y_val,
        preprocessor=preprocessor
    )

    print("\n✅ All 3 contenders trained and logged to MLflow.")
    print("Run 'mlflow ui' to compare them in the browser.")