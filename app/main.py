# ============================================================
# STAGE 8: SERVING
# app/main.py
#
# Three fixes from the original project's postmortem, all
# proven here:
#
# 1. ALIAS-BASED LOADING — loads "models:/churn-classifier@production",
#    never a hardcoded version number. Whatever register.py most
#    recently promoted is what gets served, automatically.
#
# 2. SHARED PREPROCESSOR — downloads and loads the EXACT fitted
#    ColumnTransformer from the winning training run, instead of
#    hand-rewriting one-hot encoding logic in this file. Same
#    object, same transformation, every time.
#
# 3. SERVER-SIDE FEATURE ENGINEERING — calls the real
#    engineer_features() from src/feature_engineering.py on raw
#    input, rather than asking the API caller to pre-compute
#    avg_monthly_spend / senior_long_tenure themselves.
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


# ============================================================
# LOAD PRODUCTION MODEL BUNDLE AT STARTUP
# Resolves the @production alias to a specific version, then
# pulls that version's model, matching preprocessor, and tuned
# decision threshold — all three from the SAME training run.
# ============================================================

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
        print(f"❌ Failed to load production bundle: {e}")
        return {
            'model': None, 'preprocessor': None, 'threshold': 0.5,
            'version': 'none', 'model_type': 'none',
        }


bundle = load_production_bundle()


# ============================================================
# BUILD FEATURES FROM RAW CUSTOMER INPUT
#
# This function's whole job is to turn the API's raw input
# fields into the SAME shape of dataframe that
# preprocessing.clean_data() produces from the raw dataset —
# so engineer_features() (imported directly from src, not
# reimplemented here) can be called identically to how
# train.py calls it.
# ============================================================

def build_customer_dataframe(customer: CustomerInput) -> pd.DataFrame:
    """
    Map API field names -> the internal column names/format
    clean_data() produces, so engineer_features() and the
    preprocessor both see exactly the shape they expect.
    """
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


# ============================================================
# ENDPOINTS
# ============================================================

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
        # Raw input -> same shape clean_data() produces
        df = build_customer_dataframe(customer)

        # SAME function training used — no reimplemented formulas here
        df = engineer_features(df)

        # SAME fitted transformer training used — no hand-written encoding here
        X_transformed = bundle['preprocessor'].transform(df)

        probability = float(bundle['model'].predict_proba(X_transformed)[:, 1][0])

        # Use the TUNED threshold from evaluate.py, not a naive 0.5 default —
        # this is the whole point of having threshold-tuned for recall in
        # the first place. model.predict() would silently ignore that work.
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