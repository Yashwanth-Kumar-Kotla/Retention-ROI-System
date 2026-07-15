# Project Metrics — Retention ROI System

## Resume Bullet Points (Google XYZ Formula)

> **Format**: "Accomplished **X**, as measured by **Y**, by doing **Z**"

---

### Top-Line Resume Bullets

**1. Revenue & Business Impact**
> Designed a customer churn prediction system that **identified 69.5% of at-risk customers before cancellation**, protecting an estimated **$202K in annual revenue per customer cohort**, by training a calibrated XGBoost classifier on 7,043 telecom customers with SMOTE-balanced training.

**2. Model Performance**
> Built an end-to-end ML pipeline achieving **AUC-ROC of 0.83 and AUC-PR of 0.62** on an imbalanced churn dataset (26.5% positive rate), by applying SMOTE oversampling, XGBoost gradient boosting, and isotonic probability calibration.

**3. Precision-Recall Optimization**
> Improved retention campaign efficiency by **constraining precision to ≥50% while maximizing F1-score**, reducing wasted marketing interventions by prioritizing only customers with calibrated churn probability ≥ 0.54.

**4. Calibration Quality**
> Produced production-ready churn probability scores with a **Brier Score of 0.161** (vs. 0.195 baseline), by calibrating raw XGBoost outputs using `CalibratedClassifierCV` with isotonic regression, enabling reliable expected-value calculations for retention ROI.

**5. Feature Engineering**
> Reduced feature space from 21 to 16 engineered features while **improving model signal**, by removing correlated features (TotalCharges ≈ tenure × MonthlyCharges), collapsing redundant service categories, and encoding contract type ordinally.

---

## Full Metrics Table

| Metric | Value | What It Means |
|---|---|---|
| **AUC-ROC** | 0.8297 | Model ranks 83% of churn/non-churn pairs correctly |
| **AUC-PR** | 0.6164 | Primary metric — 2.3× above random baseline (0.265) |
| **Brier Score** | 0.1613 | Calibration quality (0=perfect, 0.5=random) |
| **Churn Recall** | 69.5% | Of all actual churners, model caught 69.5% |
| **Churn Precision** | 55.4% | Of customers flagged as churning, 55.4% actually churn |
| **Churn F1** | 62% | Harmonic mean of precision and recall |
| **Overall Accuracy** | 77% | Note: misleading metric on imbalanced data |
| **Optimal Threshold** | 0.5393 | Calibrated probability cutoff for binary classification |

---

## Confusion Matrix (Test Set — 1,409 customers)

```
                   Predicted: No Churn   Predicted: Churn
Actual: No Churn        826 (TN)              209 (FP)
Actual: Churn           114 (FN)              260 (TP)
```

| | Count | Business Interpretation |
|---|---|---|
| True Positives (TP) | **260** | Churners correctly flagged — candidates for retention offers |
| False Positives (FP) | 209 | Non-churners flagged — receive unnecessary offers (cost: wasted spend) |
| False Negatives (FN) | 114 | Missed churners — lost revenue, no intervention possible |
| True Negatives (TN) | 826 | Correctly left alone — no wasted retention spend |

---

## Business Impact Estimates

| Assumption | Value |
|---|---|
| Average Monthly Charges | $64.76 |
| Assumed annual CLV per customer | $777 (12 × avg monthly) |
| Churners correctly identified (test cohort) | 260 out of 374 |
| **Revenue protected (test cohort)** | **260 × $777 = $202,051** |
| Revenue missed (FN) | 114 × $777 = $88,578 |
| Wasted retention interventions (FP) | 209 |

> Scaling to full dataset (7,043 customers): ~1,300 churners identified → **~$1.01M revenue protected**

---

## Dataset & Experiment Summary

| Item | Detail |
|---|---|
| Dataset | IBM Telco Customer Churn (public) |
| Total customers | 7,043 |
| Features (raw) | 21 |
| Features (final) | 16 |
| Train / Test split | 80% / 20%, stratified |
| Training samples (after SMOTE) | 8,278 (balanced 50/50) |
| Test samples | 1,409 (unbalanced — real-world distribution) |
| Churn rate (test) | 26.5% |

---

## Model Comparison

| Model | AUC-ROC | AUC-PR | F1 (Churn) | Selected |
|---|---|---|---|---|
| Logistic Regression | 0.72 | — | 0.57 | |
| LightGBM | ~0.83 | ~0.61 | ~0.61 | |
| **XGBoost (calibrated)** | **0.83** | **0.62** | **0.62** | ✅ |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Imbalanced Learning | imbalanced-learn (SMOTE) |
| Modeling | scikit-learn, XGBoost, LightGBM |
| Calibration | `CalibratedClassifierCV` (isotonic) |
| Persistence | pickle |
| Environment | Jupyter Notebook + venv |
