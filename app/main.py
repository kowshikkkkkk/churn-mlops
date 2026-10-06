# ============================================================
# STAGE 8: SERVING
# app/main.py
# ============================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib
import traceback
import warnings
warnings.filterwarnings('ignore')

from app.schemas import CustomerInput, PredictionResponse, HealthResponse
from feature_engineering import engineer_features

MODEL_NAME = "churn-classifier"
PRODUCTION_ALIAS = "production"

app = FastAPI(
    title="Churn Prediction API",
    description="MLOps pipeline for customer churn prediction",
    version="2.0.0"
)


def load_production_bundle():
    """
    Returns a dict with model, preprocessor, threshold, and
    metadata — or a dict with model=None if loading fails, so
    the app can still start and report an honest /health status
    instead of crashing outright.
    """
    try:
        client = MlflowClient()

        model_version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        run_id = model_version.run_id
        version = model_version.version

        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")

        preprocessor_path = client.download_artifacts(run_id, "preprocessor.joblib")
        preprocessor = joblib.load(preprocessor_path)

        threshold = float(model_version.tags.get('decision_threshold', 0.5))
        model_type = model_version.tags.get('model_type', 'unknown')

        print(f"✅ Loaded production bundle: {MODEL_NAME} v{version} "
              f"({model_type}, threshold={threshold})")

        return {
            'model': model,
            'preprocessor': preprocessor,
            'threshold': threshold,
            'version': version,
            'model_type': model_type,
        }

    except Exception as e:
        # DEBUG: full traceback, not just str(e) — we need to see exactly
        # which call inside this try block is failing and why.
        print(f"❌ Failed to load production bundle: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return {
            'model': None, 'preprocessor': None, 'threshold': 0.5,
            'version': 'none', 'model_type': 'none',
        }


bundle = load_production_bundle()


def build_customer_dataframe(customer: CustomerInput) -> pd.DataFrame:
    row = {
        'Gender': customer.Gender,
        'Senior Citizen': customer.SeniorCitizen,
        'Partner': customer.Partner,
        'Dependents': customer.Dependents,
        'Tenure Months': customer.TenureMonths,
        'Phone Service': customer.PhoneService,
        'Multiple Lines': customer.MultipleLines,
        'Internet Service': customer.InternetService,
        'Online Security': customer.OnlineSecurity,
        'Online Backup': customer.OnlineBackup,
        'Device Protection': customer.DeviceProtection,
        'Tech Support': customer.TechSupport,
        'Streaming TV': customer.StreamingTV,
        'Streaming Movies': customer.StreamingMovies,
        'Contract': customer.Contract,
        'Paperless Billing': customer.PaperlessBilling,
        'Payment Method': customer.PaymentMethod,
        'Monthly Charges': customer.MonthlyCharges,
        'Total Charges': customer.TotalCharges,
    }
    return pd.DataFrame([row])


def get_risk_level(probability: float):
    if probability >= 0.7:
        return "High", "Urgent retention call required"
    elif probability >= 0.4:
        return "Medium", "Send retention offer"
    else:
        return "Low", "No action needed"


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running", "status": "healthy"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if bundle['model'] is not None else "degraded",
        model_loaded=bundle['model'] is not None,
        preprocessor_loaded=bundle['preprocessor'] is not None,
        model_name=MODEL_NAME,
        model_version=str(bundle['version']),
        model_type=bundle['model_type'],
        decision_threshold=bundle['threshold'],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if bundle['model'] is None or bundle['preprocessor'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check /health")

    try:
        df = build_customer_dataframe(customer)
        df = engineer_features(df)
        X_transformed = bundle['preprocessor'].transform(df)
        probability = float(bundle['model'].predict_proba(X_transformed)[:, 1][0])
        prediction = int(probability >= bundle['threshold'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    risk_level, recommendation = get_risk_level(probability)

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        risk_level=risk_level,
        recommendation=recommendation,
        model_version=str(bundle['version']),
        decision_threshold=bundle['threshold'],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)