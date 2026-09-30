# ============================================================
# STAGE 9: MONITORING (DRIFT DETECTION)
# src/monitor.py
#
# Job of this file: compare a REFERENCE dataset (historical
# training data) against a CURRENT dataset (new incoming data)
# and report whether the feature distributions have drifted.
#
# WHY THIS RUNS ON ENGINEERED FEATURES, NOT ENCODED FEATURES:
# Drift is checked on the same shape of data build_preprocessor()
# takes as input (post feature_engineering.py, pre one-hot
# encoding) — NOT on the fully-transformed numeric matrix the
# model actually trains on. "Has the distribution of Contract
# values shifted" is something a human can read and act on.
# "Has one-hot column #14 shifted" means nothing without
# cross-referencing the encoder. Same features the model
# depends on, but in a form worth actually looking at.
#
# HONEST LIMITATION, STATED PLAINLY:
# There is no live incoming customer stream to compare against
# here — this is a portfolio project with one static historical
# dataset. So "current" data below is REAL historical data with
# artificial perturbations injected, simulating what drift would
# look like. This is a legitimate way to prove the MECHANISM
# works correctly; it does not mean real drift has been observed.
# In a real deployment, "current_df" would be genuinely new
# customer data collected since the model was trained.
# ============================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

from train import build_dataset

REFERENCE_PATH_HINT = "built fresh from build_dataset() each run"
DRIFT_REPORT_PATH = "reports/drift_report.html"


def simulate_current_data(reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a synthetic 'current' dataset by perturbing a copy of
    the reference data. Explicitly simulated — see module docstring.

    The specific perturbations chosen (charges up, tenure down)
    loosely simulate a scenario like "a price increase went into
    effect, and the customer base skews newer" — a plausible real
    trigger for drift, even though this specific instance is fabricated.
    """
    current = reference_df.copy()
    current['Monthly Charges'] = current['Monthly Charges'] * 1.3
    current['Tenure Months'] = current['Tenure Months'] * 0.7
    current['Total Charges'] = current['Total Charges'] * 1.2
    current['avg_monthly_spend'] = current['avg_monthly_spend'] * 1.25
    return current


def run_drift_report(reference_df: pd.DataFrame = None,
                      current_df: pd.DataFrame = None) -> bool:
    """
    Compare reference vs current feature distributions using
    Evidently's DataDriftPreset, save an HTML report, and return
    whether dataset-level drift was detected.

    Accepts optional reference_df/current_df so retrain.py (or a
    future real monitoring job) can pass in ACTUAL new data instead
    of the synthetic simulation — the simulation is only the
    default when nothing is provided, not the only mode this
    function can run in.

    The target column ('Churn') is deliberately excluded from
    both dataframes before comparison — drift monitoring checks
    whether INCOMING FEATURE data looks different, and a genuinely
    new customer being scored has no known label yet. Comparing
    label distributions wouldn't be checking what this function
    is meant to check.
    """
    if reference_df is None:
        print("No reference data provided — building from source (historical training data)")
        reference_df = build_dataset()

    if current_df is None:
        print("No current data provided — using SIMULATED drift for demonstration")
        current_df = simulate_current_data(reference_df)

    reference_features = reference_df.drop(columns=['Churn'], errors='ignore')
    current_features = current_df.drop(columns=['Churn'], errors='ignore')

    report = Report([DataDriftPreset()])
    result = report.run(current_data=current_features, reference_data=reference_features)

    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    result.save_html(DRIFT_REPORT_PATH)
    print(f"✅ Drift report saved to {DRIFT_REPORT_PATH}")

    try:
        result_dict = result.dict()

        # metrics[0] is always the dataset-level DriftedColumnsCount metric.
        # Its config carries the SHARE THRESHOLD (default 0.5 = 50% of
        # columns must drift for dataset-level drift to be flagged), and
        # its value carries the ACTUAL share observed. The correct check
        # is share > threshold — NOT "did any single column drift," which
        # is a much more sensitive (and mismatched) question. Getting this
        # wrong previously caused a false positive: 4/21 columns drifting
        # (19% share) was incorrectly reported as "drift detected," when
        # Evidently's own report clearly stated drift was NOT detected at
        # the dataset level (19% is below the 50% threshold).
        dataset_metric = result_dict['metrics'][0]
        drift_share = dataset_metric['value']['share']
        drift_share_threshold = dataset_metric['config']['drift_share']
        drifted_count = dataset_metric['value']['count']

        drift_detected = drift_share > drift_share_threshold

        print(f"   Drifted columns: {int(drifted_count)} "
              f"(share={drift_share:.3f}, threshold={drift_share_threshold})")

    except Exception as e:
        print(f"⚠️  Could not parse structured drift result ({e}) — falling back to True (assume drift, flag for review)")
        drift_detected = True

    if drift_detected:
        print("⚠️  DRIFT DETECTED")
    else:
        print("✅ No drift detected")

    return drift_detected


if __name__ == "__main__":
    print("=" * 50)
    print("DRIFT MONITORING")
    print("=" * 50)

    drift_detected = run_drift_report()

    print("=" * 50)
    print(f"Result: drift_detected={drift_detected}")
    print(f"Report: {DRIFT_REPORT_PATH}")
    print("=" * 50)