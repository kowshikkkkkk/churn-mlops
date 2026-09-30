# ============================================================
# STAGE 10: RETRAINING TRIGGER
# src/retrain.py
#
# Job of this file: check for drift, and if found, chain
# train -> evaluate -> register in sequence — using the same
# functions those files' own __main__ blocks call, not a
# separate reimplementation.
#
# Why direct function imports instead of subprocess calls
# (e.g. subprocess.run(["python", "src/train.py"])):
# calling the actual Python functions keeps this readable,
# lets errors propagate as real Python exceptions instead of
# parsed subprocess exit codes, and means there is exactly ONE
# implementation of each stage's logic — this file, train.py's
# own script entry point, and (later) an Airflow DAG can all
# call the same functions without duplicating orchestration code.
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))

from monitor import run_drift_report
from train import run_training_pipeline
from evaluate import run_evaluation_pipeline
from register import run_registration_pipeline


def retrain_pipeline():
    print("=" * 50)
    print("RETRAINING PIPELINE")
    print("=" * 50)

    print("\n🔍 Checking for data drift...")
    drift_detected = run_drift_report()

    if not drift_detected:
        print("\n✅ No drift detected — model is healthy, no retraining needed.")
        return

    print("\n⚠️  Drift detected — triggering retraining pipeline...")

    print("\n--- Step 1/3: Training ---")
    run_training_pipeline()

    print("\n--- Step 2/3: Evaluation (governance gate) ---")
    approved = run_evaluation_pipeline()

    if not approved:
        print("\n❌ Retraining produced no model that passed governance checks.")
        print("   Production model remains unchanged — nothing to register.")
        return

    print("\n--- Step 3/3: Registration ---")
    run_registration_pipeline()

    print("\n" + "=" * 50)
    print("✅ Retraining complete — new model is live in production.")
    print("=" * 50)


if __name__ == "__main__":
    retrain_pipeline()