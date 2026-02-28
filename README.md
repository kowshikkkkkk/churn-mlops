# Churn MLOps Pipeline

End-to-end MLOps pipeline for customer churn prediction.

## Project Structure
- `data/` — Raw and processed datasets
- `src/` — Training, evaluation and prediction scripts
- `app/` — FastAPI serving endpoint
- `models/` — Saved model artifacts
- `reports/` — Evaluation and drift reports

## Pipeline Steps
1. Data preparation and feature engineering
2. Experiment tracking with MLflow
3. Model registry and versioning
4. REST API serving with FastAPI
5. Docker containerization
6. Data drift monitoring with Evidently AI

## Tech Stack
Python | Scikit-learn | MLflow | FastAPI | Docker | Evidently AI