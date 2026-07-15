import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# ── Load artifacts once at startup ──────────────────────────────────────────
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/xgb_calibrated.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/threshold.txt") as f:
    THRESHOLD = float(f.read().strip())

FEATURE_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure",
    "OnlineSecurity", "TechSupport", "Contract", "PaperlessBilling",
    "MonthlyCharges",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts telecom customer churn probability using a calibrated XGBoost model.",
    version="1.0.0",
)


# ── Request / Response schemas ───────────────────────────────────────────────
class CustomerInput(BaseModel):
    senior_citizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, else 0")
    partner: int = Field(..., ge=0, le=1, description="1 if has partner")
    dependents: int = Field(..., ge=0, le=1, description="1 if has dependents")
    tenure: int = Field(..., ge=0, description="Months as customer")
    online_security: int = Field(..., ge=0, le=1, description="1 if has online security add-on")
    tech_support: int = Field(..., ge=0, le=1, description="1 if has tech support add-on")
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: int = Field(..., ge=0, le=1, description="1 if paperless billing enabled")
    monthly_charges: float = Field(..., gt=0, description="Monthly bill in USD")
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    model_config = {"json_schema_extra": {
        "example": {
            "senior_citizen": 0, "partner": 1, "dependents": 0,
            "tenure": 12, "online_security": 0, "tech_support": 0,
            "contract": "Month-to-month", "paperless_billing": 1,
            "monthly_charges": 79.85, "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        }
    }}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_predicted: bool
    risk_tier: str
    threshold_used: float


class BatchItem(BaseModel):
    customer_id: str
    input: CustomerInput


class BatchResponse(BaseModel):
    results: list[dict]


# ── Helper ───────────────────────────────────────────────────────────────────
def build_feature_row(c: CustomerInput) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLS}
    row["SeniorCitizen"] = c.senior_citizen
    row["Partner"] = c.partner
    row["Dependents"] = c.dependents
    row["tenure"] = c.tenure
    row["OnlineSecurity"] = c.online_security
    row["TechSupport"] = c.tech_support
    row["Contract"] = CONTRACT_MAP[c.contract]
    row["PaperlessBilling"] = c.paperless_billing
    row["MonthlyCharges"] = c.monthly_charges
    row[f"InternetService_{c.internet_service}"] = 1
    row[f"PaymentMethod_{c.payment_method}"] = 1
    return pd.DataFrame([row], columns=FEATURE_COLS)


def risk_tier(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    if prob >= 0.50:
        return "Medium"
    return "Low"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model": "XGBoost Calibrated Churn", "threshold": THRESHOLD}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    try:
        df = build_feature_row(customer)
        X_scaled = scaler.transform(df)
        prob = float(model.predict_proba(X_scaled)[0, 1])
        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_predicted=prob >= THRESHOLD,
            risk_tier=risk_tier(prob),
            threshold_used=THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(items: list[BatchItem]):
    if len(items) > 500:
        raise HTTPException(status_code=400, detail="Batch limit is 500 customers.")
    results = []
    for item in items:
        df = build_feature_row(item.input)
        X_scaled = scaler.transform(df)
        prob = float(model.predict_proba(X_scaled)[0, 1])
        results.append({
            "customer_id": item.customer_id,
            "churn_probability": round(prob, 4),
            "churn_predicted": prob >= THRESHOLD,
            "risk_tier": risk_tier(prob),
        })
    results.sort(key=lambda x: x["churn_probability"], reverse=True)
    return BatchResponse(results=results)
