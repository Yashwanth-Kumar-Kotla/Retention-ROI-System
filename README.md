# Retention ROI System

Predicts telecom customer churn and provides a Streamlit dashboard for model inspection and batch scoring.

This repository contains:
- a FastAPI prediction service (api/main.py)
- a Streamlit dashboard (app/dashboard.py)
- requirements (requirements.txt)
- supporting docs (metrics.md, understand.md)
- a model threshold file (models/threshold.txt referenced by code)

Features
- REST prediction endpoints:
  - GET / and GET /health (health checks) — api/main.py
  - POST /predict — single-customer prediction using a calibrated XGBoost model (api/main.py)
  - POST /predict/batch — batch scoring (max 500 customers) (api/main.py)
- Streamlit dashboard with three pages:
  - Model Dashboard — KPI cards, confusion matrix, churn probability distribution, business impact estimator, risk tier breakdown (app/dashboard.py)
  - Single Customer — interactive form to score one customer and show gauge + decision + guidance (app/dashboard.py)
  - Batch Score — view and filter held-out test set scores and download CSV (app/dashboard.py)

Requirements
- Python packages listed in requirements.txt (examples from the file):
  - pandas, numpy, scikit-learn, imbalanced-learn, lightgbm, xgboost, matplotlib, seaborn
  - fastapi, uvicorn, streamlit
  - plus jupyter, ipykernel, python-multipart
- See requirements.txt for exact pinned versions.

Installation
1. Create and activate a Python environment.
2. Install dependencies:
   pip install -r requirements.txt

Model artifacts required
The code expects a models/ directory with the following files (paths referenced in api/main.py and app/dashboard.py):
- models/scaler.pkl
- models/xgb_calibrated.pkl
- models/threshold.txt
- models/test_scored.csv (used by the Streamlit dashboard for the test set display)

If any of the above are missing the API or dashboard will fail to start (the code opens these files at startup).

Running

Streamlit dashboard
- Start the dashboard (loads model artifacts and test set):  
  streamlit run app/dashboard.py

FastAPI prediction service
- Start the API (api/main.py defines FastAPI app):  
  uvicorn api.main:app --reload

API — endpoints and schemas (from api/main.py)

- GET /
  - Returns a small health object with the loaded threshold:
    { "status": "ok", "model": "XGBoost Calibrated Churn", "threshold": <value> }

- GET /health
  - Returns { "status": "ok" }

- POST /predict
  - Request body: CustomerInput (fields validated by Pydantic). An example embedded in the code:
    {
      "senior_citizen": 0,
      "partner": 1,
      "dependents": 0,
      "tenure": 12,
      "online_security": 0,
      "tech_support": 0,
      "contract": "Month-to-month",
      "paperless_billing": 1,
      "monthly_charges": 79.85,
      "internet_service": "Fiber optic",
      "payment_method": "Electronic check"
    }
  - Response model: PredictionResponse with fields:
    - churn_probability: float (rounded to 4 decimals)
    - churn_predicted: bool (probability >= threshold read from models/threshold.txt)
    - risk_tier: str ("High", "Medium", "Low" as determined by probability)
    - threshold_used: float

- POST /predict/batch
  - Request body: list of BatchItem objects. Each BatchItem has:
    - customer_id: string
    - input: CustomerInput (same schema as single predict)
  - Batch limit: the endpoint returns HTTP 400 if more than 500 items (enforced in api/main.py).
  - Response: BatchResponse containing results list. Each result dict contains:
    - customer_id
    - churn_probability (rounded to 4 decimals)
    - churn_predicted (boolean)
    - risk_tier
  - Results are sorted by churn_probability in descending order.

Streamlit dashboard — pages and main behaviors (from app/dashboard.py)

- Model Dashboard
  - Displays KPI metrics computed from models/test_scored.csv (AUC-ROC and AUC-PR are shown as fixed numeric strings in the code: "0.8297" and "0.6164").
  - Shows confusion matrix and a histogram of calibrated churn probabilities (uses THRESHOLD read from models/threshold.txt).
  - Business impact estimator: slider to set assumed annual CLV per customer and metrics showing churners caught, revenue protected, and missed revenue.
  - Risk tier breakdown bar chart using bins Low (<50%), Medium (50–75%), High (>75%).

- Single Customer
  - Interactive form to enter customer attributes (tenure, monthly charges, contract, internet service, payment method, binary flags for online security, tech support, senior citizen, partner, dependents, paperless billing).
  - On submit the app builds features, scales with scaler.pkl and predicts using xgb_calibrated.pkl.
  - Displays churn probability, decision (CHURN or RETAIN), risk tier, a gauge-style bar, and contextual messages suggesting retention action when churn is predicted.

- Batch Score
  - Uses models/test_scored.csv to show held-out test set results.
  - Controls to filter by risk tier and minimum churn probability.
  - Shows a table with Churn Probability, Actual Churn, Predicted Churn, Risk Tier, and Result codes (TP, FN, FP, TN).
  - Download filtered results as CSV button is provided.

Project structure (top-level files referenced)
- api/main.py — FastAPI app and prediction endpoints
- app/dashboard.py — Streamlit dashboard with three pages
- requirements.txt — pinned dependencies used by the project
- metrics.md — (present in repo; contents not summarized here)
- understand.md — (present in repo; contents not summarized here)
- models/threshold.txt — threshold read at runtime by both API and dashboard

Notes and constraints
- Both API and dashboard load model artifacts at startup; ensure the models/ files listed above exist and are accessible relative to the repository root.
- The API /predict/batch endpoint enforces a maximum of 500 customers per request.
- The risk tier logic and threshold usage are implemented in code (see api/main.py and app/dashboard.py).

If you need examples of request/response payloads or help preparing the required model artifacts, share the artifacts or ask for assistance.