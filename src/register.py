# ============================================================
# STAGE 7: MODEL REGISTRATION
# src/register.py
#
# Job of this file:
#   1. Read models/approved_model.json (written by evaluate.py —
#      the run_id of whichever model passed governance)
#   2. Register that run's model in the MLflow Model Registry
#      as a new version
#   3. Point a "production" ALIAS at that new version
#
# THIS IS THE FIX FOR THE ORIGINAL PROJECT'S WORST BUG:
# The old app/main.py did:
#     mlflow.sklearn.load_model("models:/churn-classifier/1")
# — hardcoded to version 1, forever. Every retrain after that
# was invisible to the serving API, no matter what got approved.
#
# Aliases solve this properly. An alias (e.g. "production") is
# a NAMED POINTER that can be reassigned to any version at any
# time. app/main.py will load:
#     mlflow.sklearn.load_model("models:/churn-classifier@production")
# — which always resolves to whatever version currently holds
# that alias. Promoting a new model just means moving the
# pointer here; the serving code never needs to change or know
# a specific version number.
#
# (Aliases are MLflow's current recommended approach, replacing
# the older "stage" system — Staging/Production/Archived — which
# is deprecated as of MLflow 2.9+.)
# ============================================================

import mlflow
from mlflow.tracking import MlflowClient
import json
import os

MODEL_NAME = "churn-classifier"
PRODUCTION_ALIAS = "production"
APPROVED_MODEL_PATH = "models/approved_model.json"


def load_approval_record() -> dict:
    """Read the decision evaluate.py already made — this file doesn't re-decide anything."""
    if not os.path.exists(APPROVED_MODEL_PATH):
        raise FileNotFoundError(
            f"No approval record found at {APPROVED_MODEL_PATH}. "
            f"Run evaluate.py first — a model must pass governance "
            f"before it can be registered."
        )

    with open(APPROVED_MODEL_PATH, 'r') as f:
        record = json.load(f)

    print(f"✅ Loaded approval record: {record['model_type']} "
          f"(run_id={record['run_id']})")
    return record


def register_model_version(run_id: str) -> "mlflow.entities.model_registry.ModelVersion":
    """
    Register the approved run's model artifact as a new version
    of MODEL_NAME in the registry. If MODEL_NAME doesn't exist
    yet, MLflow creates it automatically as version 1; otherwise
    this becomes the next incrementing version.
    """
    model_uri = f"runs:/{run_id}/model"

    print(f"\nRegistering model...")
    print(f"  Model URI: {model_uri}")

    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print(f"✅ Registered as '{MODEL_NAME}' version {result.version}")
    return result


def attach_metadata(version: str, record: dict) -> None:
    """
    Attach the decision threshold and key test metrics as tags
    on this specific model version. This is what lets app/main.py
    know not just WHICH model to load, but what threshold to use
    with it — without the API needing its own hardcoded copy of
    that number.
    """
    client = MlflowClient()

    tags = {
        'model_type': record['model_type'],
        'decision_threshold': str(record['threshold']),
        'test_auc': str(record['test_metrics']['auc']),
        'test_recall': str(record['test_metrics']['recall']),
        'test_precision': str(record['test_metrics']['precision']),
        'test_f1': str(record['test_metrics']['f1']),
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name=MODEL_NAME, version=version, key=key, value=value
        )

    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=(
            f"{record['model_type']}, threshold={record['threshold']}, "
            f"test recall={record['test_metrics']['recall']}, "
            f"test precision={record['test_metrics']['precision']}"
        )
    )

    print(f"✅ Attached metadata tags: {list(tags.keys())}")


def promote_to_production(version: str) -> None:
    """
    Point the 'production' alias at this version. If another
    version currently holds this alias, it's automatically moved
    — MLflow enforces that an alias points to exactly one version
    at a time, so there's never ambiguity about which model is live.
    """
    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME, alias=PRODUCTION_ALIAS, version=version
    )

    print(f"\n✅ Alias '@{PRODUCTION_ALIAS}' now points to "
          f"'{MODEL_NAME}' version {version}")
    print(f"   Serving code loads: models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")


def run_registration_pipeline():
    """
    Full registration orchestration: read the approval record,
    register the model, attach metadata tags, promote via alias.

    Raises FileNotFoundError if no approval record exists (i.e.
    evaluate.py hasn't approved a model) — callers like retrain.py
    should only invoke this after confirming run_evaluation_pipeline()
    returned True.
    """
    print("=" * 50)
    print("MODEL REGISTRATION")
    print("=" * 50)

    record = load_approval_record()
    registered_version = register_model_version(record['run_id'])
    attach_metadata(registered_version.version, record)
    promote_to_production(registered_version.version)

    print("\n" + "=" * 50)
    print("REGISTRATION SUMMARY")
    print("=" * 50)
    print(f"Model         : {MODEL_NAME}")
    print(f"Version       : {registered_version.version}")
    print(f"Alias         : @{PRODUCTION_ALIAS}")
    print(f"Model type    : {record['model_type']}")
    print(f"Threshold     : {record['threshold']}")
    print(f"Test recall   : {record['test_metrics']['recall']}")
    print(f"Test precision: {record['test_metrics']['precision']}")

    return registered_version


if __name__ == "__main__":
    run_registration_pipeline()