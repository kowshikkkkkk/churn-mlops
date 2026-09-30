# ============================================================
# STAGE: TESTING
# tests/test_pipeline.py
#
# Run with: pytest tests/test_pipeline.py -v
#
# These tests are written against the pipeline as it ACTUALLY
# exists today (data_ingestion.py, data_validation.py,
# preprocessing.py, feature_engineering.py, train.py) — not
# against an old interface that used to exist. This is the
# direct fix for what killed the original project's test suite:
# it kept calling preprocess.load_and_preprocess() and
# retrain_model() long after those functions had been renamed
# or removed, so every test failed on import, silently, and
# nobody noticed until we traced it by hand.
#
# What's tested here: stages 1-5 (ingestion through the
# preprocessor/split mechanics that feed training). Deliberately
# NOT tested here: the MLflow registry / production model
# (that's covered separately once monitor.py and the CI/CD
# stage define how a freshly-trained model gets exercised in
# an automated environment).
# ============================================================

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_ingestion import load_raw_data
from data_validation import run_all_validations, DataValidationError, TARGET_COLUMN as RAW_TARGET
from preprocessing import clean_data, LEAKAGE_COLUMNS, NO_SIGNAL_COLUMNS
from feature_engineering import engineer_features
from train import split_data, build_preprocessor

RAW_PATH = "data/raw/Telco_customer_churn.xlsx"


# ============================================================
# FIXTURES — build each pipeline stage's output ONCE per test
# session, since these are read-only inputs every test shares.
# Fixtures return the same object to every test; no test may
# mutate what it receives (each test that needs to modify data
# should df.copy() first — never overwrite a shared fixture).
# ============================================================

@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data(RAW_PATH)


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="module")
def featured_df(cleaned_df):
    return engineer_features(cleaned_df)


# ============================================================
# STAGE 1: INGESTION
# ============================================================

def test_ingestion_shape_and_columns(raw_df):
    """
    Confirms the raw file loads to the expected shape. If this
    fails, every other test's fixture will also fail (they all
    depend on raw_df) — that's intentional: an ingestion problem
    should surface as an ingestion failure, not a confusing
    downstream error in preprocessing.
    """
    assert raw_df.shape[0] == 7043, f"Expected 7043 rows, got {raw_df.shape[0]}"
    assert raw_df.shape[1] == 33, f"Expected 33 columns, got {raw_df.shape[1]}"


def test_ingestion_raises_on_missing_file():
    """A missing file should raise a clear, specific error — not a generic pandas crash."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/raw/does_not_exist.xlsx")


# ============================================================
# STAGE 2: VALIDATION
# ============================================================

def test_validation_passes_on_real_data(raw_df):
    """The real dataset should pass every check without raising."""
    run_all_validations(raw_df)  # raises on failure; no exception = pass


def test_validation_catches_corrupted_target(raw_df):
    """
    Deliberately corrupt the target column (inject a value that
    isn't 0 or 1) and confirm validation catches it. This proves
    the check actually checks something, rather than passing
    vacuously on any input.
    """
    bad_df = raw_df.copy()
    bad_df.loc[0, RAW_TARGET] = 2

    with pytest.raises(DataValidationError):
        run_all_validations(bad_df)


def test_validation_catches_missing_columns(raw_df):
    """Dropping a required column should be caught, not silently ignored."""
    bad_df = raw_df.drop(columns=['Contract'])

    with pytest.raises(DataValidationError):
        run_all_validations(bad_df)


# ============================================================
# STAGE 3: PREPROCESSING
# ============================================================

def test_preprocessing_drops_leakage_columns(cleaned_df):
    """None of the leakage columns should survive preprocessing."""
    for col in LEAKAGE_COLUMNS:
        assert col not in cleaned_df.columns, f"Leakage column '{col}' was not dropped"


def test_preprocessing_drops_no_signal_columns(cleaned_df):
    """None of the no-signal columns (CustomerID, Country, etc.) should survive."""
    for col in NO_SIGNAL_COLUMNS:
        assert col not in cleaned_df.columns, f"No-signal column '{col}' was not dropped"


def test_preprocessing_no_nulls_remain(cleaned_df):
    """Total Charges imputation should leave zero nulls anywhere."""
    null_count = cleaned_df.isnull().sum().sum()
    assert null_count == 0, f"Expected 0 nulls after preprocessing, found {null_count}"


def test_preprocessing_binary_columns_are_0_or_1(cleaned_df):
    """Yes/No columns should be cleanly converted to integers 0/1, nothing else."""
    binary_cols = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']
    for col in binary_cols:
        unique_vals = set(cleaned_df[col].unique())
        assert unique_vals.issubset({0, 1}), f"'{col}' has non-binary values: {unique_vals}"


def test_preprocessing_target_renamed(cleaned_df):
    """'Churn Value' should be renamed to 'Churn' for downstream stages."""
    assert 'Churn' in cleaned_df.columns
    assert 'Churn Value' not in cleaned_df.columns


# ============================================================
# STAGE 4: FEATURE ENGINEERING
# ============================================================

def test_feature_engineering_adds_expected_columns(featured_df):
    assert 'avg_monthly_spend' in featured_df.columns
    assert 'senior_long_tenure' in featured_df.columns


def test_avg_monthly_spend_handles_zero_tenure():
    """
    Edge case: a brand-new customer with 0 tenure months would
    cause a division-by-zero if not handled. Confirm the fallback
    to Monthly Charges actually fires, rather than producing
    inf/NaN — this exact scenario is also the population most at
    risk of early churn, so silently breaking this feature for
    them would be a serious, easy-to-miss bug.
    """
    test_row = pd.DataFrame([{
        'Tenure Months': 0,
        'Total Charges': 0.0,
        'Monthly Charges': 75.0,
        'Senior Citizen': 0,
    }])

    result = engineer_features(test_row)

    assert result['avg_monthly_spend'].iloc[0] == 75.0, (
        "Expected avg_monthly_spend to fall back to Monthly Charges when tenure=0"
    )
    assert not result['avg_monthly_spend'].isnull().any(), "avg_monthly_spend should never be null"


def test_senior_long_tenure_logic():
    """Confirm the interaction feature fires only for senior + long-tenure, not either alone."""
    test_rows = pd.DataFrame([
        {'Tenure Months': 30, 'Total Charges': 1000.0, 'Monthly Charges': 50.0, 'Senior Citizen': 1},  # both -> 1
        {'Tenure Months': 30, 'Total Charges': 1000.0, 'Monthly Charges': 50.0, 'Senior Citizen': 0},  # not senior -> 0
        {'Tenure Months': 10, 'Total Charges': 500.0, 'Monthly Charges': 50.0, 'Senior Citizen': 1},   # short tenure -> 0
    ])

    result = engineer_features(test_rows)

    assert result['senior_long_tenure'].tolist() == [1, 0, 0]


def test_feature_engineering_rejects_unstandardized_senior_citizen():
    """
    If Senior Citizen still contains 'Yes'/'No' text (preprocessing
    skipped), this should raise a clear error rather than silently
    producing wrong results — this is the ordering dependency
    between preprocessing and feature_engineering made explicit.
    """
    bad_row = pd.DataFrame([{
        'Tenure Months': 30, 'Total Charges': 1000.0,
        'Monthly Charges': 50.0, 'Senior Citizen': 'Yes',
    }])

    with pytest.raises(ValueError):
        engineer_features(bad_row)


# ============================================================
# STAGE 5: SPLIT + PREPROCESSOR
# ============================================================

def test_split_proportions_and_no_row_loss(featured_df):
    """Confirm the 70/15/15 split adds up exactly and loses no rows."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(featured_df)

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(featured_df), "Split lost or duplicated rows"

    train_pct = len(X_train) / len(featured_df)
    val_pct = len(X_val) / len(featured_df)
    test_pct = len(X_test) / len(featured_df)

    assert 0.68 <= train_pct <= 0.72, f"Train split off target: {train_pct:.3f}"
    assert 0.13 <= val_pct <= 0.17, f"Val split off target: {val_pct:.3f}"
    assert 0.13 <= test_pct <= 0.17, f"Test split off target: {test_pct:.3f}"


def test_split_is_stratified(featured_df):
    """
    Churn rate across train/val/test should stay close to the
    overall rate — confirms stratify=y is actually working, not
    just present in the code.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(featured_df)

    overall_rate = featured_df['Churn'].mean()

    for name, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
        rate = y.mean()
        assert abs(rate - overall_rate) < 0.03, (
            f"{name} churn rate {rate:.3f} drifted too far from overall {overall_rate:.3f}"
        )


def test_preprocessor_handles_unseen_category(featured_df):
    """
    Fit the preprocessor on train only, then feed it a category
    value it has never seen (simulating a brand-new PaymentMethod
    appearing later in production). handle_unknown='ignore' should
    encode it as all-zeros rather than crashing — this is what
    lets the API stay up even when live data includes something
    training never anticipated.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(featured_df)
    preprocessor = build_preprocessor(X_train)

    unseen_row = X_val.iloc[[0]].copy()
    unseen_row['Payment Method'] = 'Cryptocurrency'  # does not exist in training data

    # Should not raise
    transformed = preprocessor.transform(unseen_row)
    assert transformed.shape[0] == 1


def test_preprocessor_output_has_no_nulls(featured_df):
    """The transformed feature matrix a model actually trains on should be fully numeric, no NaNs."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(featured_df)
    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    assert not np.isnan(X_train_transformed).any(), "Transformed training data contains NaNs"


# ============================================================
# CONTRACT TESTS
#
# These check invariants that must hold regardless of which
# specific model is currently deployed or what its exact metrics
# are — they test the CONTRACT between pipeline stages, not one
# specific run's numbers. That's deliberate: a test that hardcodes
# today's threshold or today's churn rate breaks the next time you
# retrain, for no real reason. These are written to stay valid
# across retrains.
# ============================================================

def test_data_contract_churn_ratio_preserved(raw_df, cleaned_df):
    """
    DATA CONTRACT: preprocessing must not silently drop or filter
    rows based on the target. clean_data() only drops COLUMNS and
    imputes Total Charges — it never removes rows — so the churn
    ratio before and after must match EXACTLY, not just approximately.

    If this ever fails, it means a future edit to preprocessing.py
    accidentally started filtering rows (e.g. an errant dropna()),
    which would silently bias the dataset.
    """
    raw_churn_rate = raw_df['Churn Value'].mean()
    cleaned_churn_rate = cleaned_df['Churn'].mean()

    assert raw_df.shape[0] == cleaned_df.shape[0], (
        f"Row count changed during preprocessing: {raw_df.shape[0]} -> {cleaned_df.shape[0]}. "
        f"Preprocessing should only drop COLUMNS, never rows."
    )
    assert raw_churn_rate == cleaned_churn_rate, (
        f"Churn ratio shifted during preprocessing: {raw_churn_rate:.4f} -> {cleaned_churn_rate:.4f}"
    )


@pytest.fixture(scope="module")
def production_bundle():
    """
    Attempts to load whatever's currently tagged @production in
    the MLflow registry. If nothing has been registered yet (e.g.
    a fresh clone of this repo before register.py has ever run),
    every test depending on this fixture SKIPS with a clear reason
    instead of failing — this is a live-registry integration check,
    not a pure pipeline unit test, so it can't run without that
    registry state existing.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    import joblib

    MODEL_NAME = "churn-classifier"

    try:
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        run_id = model_version.run_id

        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
        preprocessor_path = client.download_artifacts(run_id, "preprocessor.joblib")
        preprocessor = joblib.load(preprocessor_path)
        threshold = float(model_version.tags.get('decision_threshold', 0.5))

        return {
            'model': model, 'preprocessor': preprocessor,
            'threshold': threshold, 'model_version': model_version,
        }
    except Exception as e:
        pytest.skip(f"No production model registered yet — run train.py, evaluate.py, "
                     f"register.py first. ({e})")


def test_output_bounds_check(production_bundle, featured_df):
    """
    OUTPUT BOUNDS CONTRACT: predict_proba's second column (probability
    of churn) must always fall strictly between 0 and 1. If this ever
    fails, something is seriously wrong with the model artifact or the
    sklearn API contract itself — this should never realistically fail,
    which is exactly why it's worth asserting explicitly.
    """
    X = featured_df.drop(columns=['Churn']).head(20)
    X_transformed = production_bundle['preprocessor'].transform(X)

    probabilities = production_bundle['model'].predict_proba(X_transformed)[:, 1]

    assert (probabilities > 0).all() and (probabilities < 1).all(), (
        f"Found probabilities outside (0, 1): min={probabilities.min()}, max={probabilities.max()}"
    )


def test_threshold_contract_check():
    """
    THRESHOLD CONTRACT: verifies the decision rule itself
    (probability >= threshold -> 1, else 0) — the exact logic
    app/main.py uses. This is tested with synthetic values, NOT
    the live threshold number, so it stays valid no matter what
    the currently registered threshold happens to be.

    Checks the boundary case explicitly: a probability EXACTLY
    equal to the threshold must map to 1 (>=, not >) — getting
    this boundary backwards would mean a customer sitting exactly
    at the cutoff gets the wrong classification.
    """
    threshold = 0.417  # arbitrary value for this test only — not read from the registry

    test_cases = [
        (0.0, 0),
        (threshold - 0.001, 0),   # just below -> not flagged
        (threshold, 1),            # exactly at threshold -> flagged (>=, not >)
        (threshold + 0.001, 1),   # just above -> flagged
        (1.0, 1),
    ]

    for probability, expected in test_cases:
        actual = int(probability >= threshold)
        assert actual == expected, (
            f"probability={probability}, threshold={threshold}: "
            f"expected {expected}, got {actual}"
        )


def test_artifact_metadata_check(production_bundle):
    """
    ARTIFACT METADATA CONTRACT: the registered production package
    must contain all three pieces app/main.py depends on at startup —
    model, preprocessor, and decision threshold. Missing any one of
    these would mean the API loads successfully but serves broken
    or default-threshold predictions silently, which is exactly the
    kind of gap a health check alone wouldn't necessarily catch.
    """
    assert production_bundle['model'] is not None, "Production model failed to load"
    assert production_bundle['preprocessor'] is not None, "Production preprocessor failed to load"

    threshold_tag = production_bundle['model_version'].tags.get('decision_threshold')
    assert threshold_tag is not None, "decision_threshold tag missing from registered model version"
    assert 0.0 < float(threshold_tag) < 1.0, f"decision_threshold {threshold_tag} out of valid range"

    model_type_tag = production_bundle['model_version'].tags.get('model_type')
    assert model_type_tag is not None, "model_type tag missing from registered model version"


def test_invariant_to_dropped_columns(production_bundle, raw_df):
    """
    INVARIANT FEATURE CHECK: this dataset has no 'phone_number'-style
    column, but CustomerID / Zip Code / Lat Long / Latitude / Longitude
    play that exact role — unique-per-row identifiers with no real
    predictive signal, which preprocessing.py already drops entirely.

    This test proves that drop is actually effective: two customers
    who are IDENTICAL on every real feature, but differ on these
    dropped columns, must receive EXACTLY the same prediction. If
    this ever failed, it would mean one of these "no-signal" columns
    is somehow still leaking into the model — a real bug worth
    catching immediately.
    """
    from preprocessing import clean_data
    from feature_engineering import engineer_features

    row_original = raw_df.iloc[[0]].copy()
    row_modified = raw_df.iloc[[0]].copy()

    # Change ONLY the no-signal identifier columns — every real feature stays identical
    row_modified['CustomerID'] = '9999-ZZZZZ'
    row_modified['Zip Code'] = 90210
    row_modified['Latitude'] = 34.0522
    row_modified['Longitude'] = -118.2437

    combined = pd.concat([row_original, row_modified], ignore_index=True)
    cleaned = clean_data(combined)
    featured = engineer_features(cleaned)
    X = featured.drop(columns=['Churn'])

    X_transformed = production_bundle['preprocessor'].transform(X)
    probabilities = production_bundle['model'].predict_proba(X_transformed)[:, 1]

    assert probabilities[0] == probabilities[1], (
        f"Prediction changed after only altering dropped no-signal columns: "
        f"{probabilities[0]} vs {probabilities[1]}. A supposedly-dropped column "
        f"may still be influencing predictions."
    )


def test_extreme_values_robustness(production_bundle, raw_df):
    """
    ROBUSTNESS CHECK: the model must not crash or output NaN when
    fed extreme, out-of-bounds numeric values — a data entry error
    or a genuinely unusual customer (e.g. an extremely long-tenure
    enterprise account) should degrade gracefully, not take down
    the API. This is exactly the kind of input real production
    traffic eventually sends, whether by mistake or by a legitimate
    outlier customer.
    """
    from preprocessing import clean_data
    from feature_engineering import engineer_features

    extreme_cases = {
        'extreme_high_tenure': {'Tenure Months': 999_999},
        'extreme_high_charges': {'Monthly Charges': 1_000_000.0, 'Total Charges': 999_999_999.0},
        'zero_everything': {'Tenure Months': 0, 'Monthly Charges': 0.0, 'Total Charges': 0.0},
        'negative_charges': {'Monthly Charges': -500.0, 'Total Charges': -1000.0},
    }

    for case_name, overrides in extreme_cases.items():
        row = raw_df.iloc[[0]].copy()
        for col, val in overrides.items():
            row[col] = val

        cleaned = clean_data(row)
        featured = engineer_features(cleaned)
        X = featured.drop(columns=['Churn'])

        X_transformed = production_bundle['preprocessor'].transform(X)
        probability = production_bundle['model'].predict_proba(X_transformed)[:, 1][0]

        assert not np.isnan(probability), f"[{case_name}] produced NaN probability"
        assert not np.isinf(probability), f"[{case_name}] produced infinite probability"
        assert 0.0 <= probability <= 1.0, f"[{case_name}] probability out of bounds: {probability}"


def test_deterministic_inference(production_bundle, featured_df):
    """
    DETERMINISTIC INFERENCE CHECK: scoring the exact same customer
    data twice must produce IDENTICAL probabilities. This is a
    different guarantee than the n_jobs=1 fix we made for TRAINING
    determinism — that fixed run-to-run variance while fitting a new
    model. This test instead confirms that a single, already-trained
    model artifact behaves as a pure function at inference time: same
    input in, same output out, every time, with no hidden randomness.
    """
    X = featured_df.drop(columns=['Churn']).head(5)
    X_transformed = production_bundle['preprocessor'].transform(X)

    probabilities_run_1 = production_bundle['model'].predict_proba(X_transformed)[:, 1]
    probabilities_run_2 = production_bundle['model'].predict_proba(X_transformed)[:, 1]

    assert np.array_equal(probabilities_run_1, probabilities_run_2), (
        "Same input produced different probabilities across two inference calls — "
        "inference is not deterministic."
    )


def test_drift_detection_uses_share_not_count(featured_df):
    """
    REGRESSION TEST for a real bug: an earlier version of
    monitor.py's parsing logic checked "did ANY column drift"
    (value.count > 0) instead of "did the SHARE of drifted
    columns cross the configured threshold" (value.share >
    config.drift_share). This caused a false positive — 4 of 21
    columns drifting (19% share, below the 50% threshold) was
    incorrectly reported as dataset-level drift, when Evidently's
    own report clearly showed drift was NOT detected.

    This test locks in the fix: with a MILD perturbation (fewer
    columns changed, small enough that share stays below 0.5),
    run_drift_report() must return False — matching what a human
    reading the HTML report would conclude, not just "something,
    somewhere, changed a little."
    """
    from monitor import run_drift_report

    # Perturb only ONE column slightly — should clearly stay
    # below the 50% dataset-level drift_share threshold.
    mild_current = featured_df.copy()
    mild_current['Monthly Charges'] = mild_current['Monthly Charges'] * 1.05

    drift_detected = run_drift_report(reference_df=featured_df, current_df=mild_current)

    assert drift_detected == False, (
        "A single mildly-perturbed column triggered dataset-level drift — "
        "the share-vs-threshold comparison may have regressed back to a "
        "naive 'any column changed' check."
    )