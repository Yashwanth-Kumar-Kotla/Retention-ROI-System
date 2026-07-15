# Retention ROI System — Complete Project Walkthrough

## What This Project Does (The Short Answer)

This project builds a machine learning system that predicts **which telecom customers are about to cancel their subscription (churn)** — before they actually do. A business can use this to target at-risk customers with retention offers (discounts, free upgrades, calls from support), saving revenue that would otherwise be lost.

Without this system: the business finds out a customer churned *after* they left. Nothing can be done.  
With this system: the business gets a ranked list of at-risk customers *weeks in advance*, with a probability score per customer.

---

## Phase 1 — Understanding the Data (Notebook: 01_EDA_and_Feature_Engineering)

### Dataset
- **Source**: IBM Telco Customer Churn dataset
- **Size**: 7,043 customers × 21 features
- **Target**: `Churn` column (Yes = left the company, No = stayed)
- **Churn rate**: ~26.5% of customers churned — this is **imbalanced** data (more non-churners than churners)

### What Each Feature Means
| Feature | What It Tells Us |
|---|---|
| `tenure` | How many months the customer has been with the company |
| `Contract` | Month-to-month, One year, or Two year — a strong churn predictor |
| `MonthlyCharges` | How much they pay per month |
| `InternetService` | DSL, Fiber optic, or No internet |
| `OnlineSecurity`, `TechSupport` | Add-on services (customers without these churn more) |
| `PaymentMethod` | Electronic check customers churn at much higher rates |
| `SeniorCitizen` | Binary flag (0/1) |
| `Partner`, `Dependents` | Social tie-ins that reduce churn |

### Data Cleaning Steps
1. **TotalCharges fix**: 11 customers had `tenure=0` (brand new), causing `TotalCharges` to be blank. Filled with 0.
2. **Binary encoding**: All Yes/No columns converted to 1/0 (e.g., `Churn: Yes → 1, No → 0`).
3. **No-service categories collapsed**: "No internet service" and "No phone service" → treated as "No" since they carry the same meaning.
4. **Contract ordinal encoding**: Month-to-month=0, One year=1, Two year=2 (natural order).
5. **One-hot encoding**: `InternetService` and `PaymentMethod` had multiple categories → expanded into binary columns with `pd.get_dummies`.

### Features Dropped (Why)
| Dropped | Reason |
|---|---|
| `customerID` | Unique ID, no predictive value |
| `gender` | Low correlation with churn in heatmap |
| `TotalCharges` | Near-perfectly correlated with `tenure × MonthlyCharges` (multicollinearity) |
| `PhoneService` | Very low variance / low correlation |
| `MultipleLines` | Low added signal after collapsing "No phone service" |
| `OnlineBackup`, `DeviceProtection`, `StreamingTV`, `StreamingMovies` | Redundant with OnlineSecurity/TechSupport; dropped after correlation analysis |

### Final Feature Set (16 features saved to `data/master_dataset_v1.csv`)
`SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `OnlineSecurity`, `TechSupport`, `Contract`, `PaperlessBilling`, `MonthlyCharges`, `InternetService_DSL`, `InternetService_Fiber optic`, `InternetService_No`, `PaymentMethod_Bank transfer`, `PaymentMethod_Credit card`, `PaymentMethod_Electronic check`, `PaymentMethod_Mailed check`

---

## Phase 2 — Solving the Imbalanced Data Problem

### The Problem
26.5% churn vs 73.5% no-churn. A naive model can achieve 73.5% accuracy by just predicting "no churn" for everyone — but that's useless. You'd miss every churner.

### The Fix: SMOTE
**SMOTE (Synthetic Minority Over-sampling Technique)** creates *synthetic* examples of churners during training (not by copying — by interpolating between real churner examples). This balances the classes so the model learns churner patterns equally.

- Before SMOTE: 4,139 no-churn, 1,493 churn (training set)
- After SMOTE: 4,139 no-churn, 4,139 churn (balanced)

**Important**: SMOTE is applied *only to training data*. Test data is never touched — that would be data leakage.

---

## Phase 3 — Model Selection (Notebook: 02_modeling)

### Models Tried
| Model | AUC-ROC | F1 (Churn) | Notes |
|---|---|---|---|
| Logistic Regression | 0.72 | 0.57 | Fast but underpowered for complex patterns |
| XGBoost | 0.83 | 0.60 | Best performer — gradient boosted trees |
| LightGBM | ~0.83 | ~0.61 | Similar to XGBoost, tested for comparison |

**Winner: XGBoost** — selected for its superior AUC-ROC and business-cost-optimized threshold performance.

### Why AUC-PR (not accuracy) is the primary metric
- **Accuracy** is misleading on imbalanced data
- **AUC-ROC** measures overall discrimination (good general metric)
- **AUC-PR (Precision-Recall)** is the primary metric here because it's most sensitive to performance on the minority class (churners) — which is exactly what the business cares about

Final XGBoost AUC-PR: **0.6164** (baseline random model = 0.265 = churn rate)

---

## Phase 4 — Calibration

### What Calibration Means
Raw XGBoost gives probability scores, but they aren't *true* probabilities. For example, it might output 0.85 for something that only happens 60% of the time.

**Calibrated probabilities = the output score matches the real-world frequency.**  
If the model says a customer has a 70% churn probability, roughly 70% of such customers should actually churn.

### Why It Matters for Business
A retention manager needs to know: "Of the customers we flag today, how many will *actually* churn?" Calibrated scores answer this. Uncalibrated scores cannot be trusted for resource allocation.

### How It Was Done
Used `CalibratedClassifierCV` with `method='isotonic'` (non-parametric — works better than Platt scaling when you have enough data). The base XGBoost was frozen (`FrozenEstimator`) and the calibration layer was fit on a held-out calibration set.

**Result**: Brier Score improved from uncalibrated → 0.1613 (lower = better, 0 = perfect, 0.5 = random)

---

## Phase 5 — Threshold Optimization

### The Default Problem
Classifiers default to threshold = 0.5: if churn probability > 0.5 → predict churn. But this isn't optimal for all business goals.

**Business cost analysis showed threshold=0.10 minimized total cost** — but at precision=40%, 60% of interventions would be wasted on non-churners.

### The Fix
Constrained optimization: find the threshold that **maximizes F1-score while keeping precision ≥ 50%** (so at least half your interventions are on real churners).

**Final threshold: 0.5393**

---

## Phase 6 — Final Results

### Test Set Performance (1,409 customers, never seen during training)

| Metric | Value |
|---|---|
| AUC-ROC | **0.8297** |
| AUC-PR | **0.6164** |
| Brier Score | 0.1613 |
| Accuracy | 77% |
| Churn Precision | 55.4% |
| Churn Recall | 69.5% |
| Churn F1 | 62% |

### Confusion Matrix
```
                 Predicted: No Churn    Predicted: Churn
Actual: No Churn      826 (TN)              209 (FP)
Actual: Churn         114 (FN)              260 (TP)
```

**What this means in plain English**:
- Out of 374 customers who actually churned, the model caught **260 (69.5%)** in time for intervention
- It incorrectly flagged **209 non-churners** (false alarms — these receive unnecessary offers)
- It missed **114 churners** (false negatives — these are the revenue losses we couldn't prevent)

### Business Impact Estimate
- Average Monthly Charges: $64.76/customer
- Assumed CLV saved per retained customer: $777/year (12 months × avg charge)
- **Revenue protected per 1,409-customer cohort**: 260 × $777 = **$202,051**

---

## What's Saved and How to Use It

All production artifacts are in `models/`:

| File | Purpose |
|---|---|
| `xgb_calibrated.pkl` | Production model — loads and predicts calibrated churn probabilities |
| `xgb_base.pkl` | Raw model — use for SHAP feature importance explanations |
| `scaler.pkl` | StandardScaler — must be applied to new data before predicting |
| `threshold.txt` | Contains 0.5393 — apply this to convert probabilities to binary predictions |
| `test_scored.csv` | Test set with all predictions — ready for dashboard or analysis |

### Inference Recipe (using saved models)
```python
import pickle, pandas as pd

# Load
scaler    = pickle.load(open("models/scaler.pkl", "rb"))
model     = pickle.load(open("models/xgb_calibrated.pkl", "rb"))
threshold = float(open("models/threshold.txt").read())

# Score new customers (same 16-feature format as master_dataset_v1.csv)
X_new_scaled = scaler.transform(X_new)
churn_prob   = model.predict_proba(X_new_scaled)[:, 1]
churn_flag   = (churn_prob >= threshold).astype(int)
```

---

## Why This Project Matters

| Without the Model | With the Model |
|---|---|
| Reactive: intervene after customer leaves | Proactive: intervene before they decide to leave |
| 0% churners saved | ~69.5% of churners identified in advance |
| No prioritization of retention spend | Score-ranked list lets teams focus on highest-risk customers |
| No quantified ROI for retention budget | Calibrated probabilities let you calculate expected revenue saved per campaign |

The system turns a **reactive customer service problem** into a **data-driven revenue protection strategy**.
