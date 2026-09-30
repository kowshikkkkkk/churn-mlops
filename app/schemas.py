# ============================================================
# API SCHEMAS
# app/schemas.py
#
# Input schema asks ONLY for fields a caller would actually
# know about a live customer — raw facts, nothing derived.
#
# Specifically NOT included: avg_monthly_spend, senior_long_tenure.
# These are ENGINEERED features computed from raw fields by
# src/feature_engineering.py during training. The old version of
# this project asked the API CALLER to supply these directly —
# which meant training and serving could silently drift apart
# the moment the formula changed in one place but not the other.
# This version computes them server-side, in main.py, by calling
# the exact same engineer_features() function training used.
# ============================================================

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Raw customer fields — everything a live customer record would have."""

    Gender: str = Field(default="Male", description="Male or Female")
    SeniorCitizen: int = Field(default=0, description="1 if senior citizen, else 0")
    Partner: int = Field(default=0, description="1 if has partner, else 0")
    Dependents: int = Field(default=0, description="1 if has dependents, else 0")
    TenureMonths: float = Field(default=12.0, description="Months as a customer")
    PhoneService: int = Field(default=1, description="1 if has phone service, else 0")
    MultipleLines: str = Field(default="No", description="Yes / No / No phone service")
    InternetService: str = Field(default="Fiber optic", description="DSL / Fiber optic / No")
    OnlineSecurity: str = Field(default="No", description="Yes / No / No internet service")
    OnlineBackup: str = Field(default="No", description="Yes / No / No internet service")
    DeviceProtection: str = Field(default="No", description="Yes / No / No internet service")
    TechSupport: str = Field(default="No", description="Yes / No / No internet service")
    StreamingTV: str = Field(default="No", description="Yes / No / No internet service")
    StreamingMovies: str = Field(default="No", description="Yes / No / No internet service")
    Contract: str = Field(default="Month-to-month", description="Month-to-month / One year / Two year")
    PaperlessBilling: int = Field(default=1, description="1 if paperless billing, else 0")
    PaymentMethod: str = Field(default="Electronic check")
    MonthlyCharges: float = Field(default=65.0)
    TotalCharges: float = Field(default=780.0)

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "SeniorCitizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "TenureMonths": 12.0,
                "PhoneService": 1,
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": 1,
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 1026.0
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    recommendation: str
    model_version: str
    decision_threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    model_name: str
    model_version: str
    model_type: str
    decision_threshold: float