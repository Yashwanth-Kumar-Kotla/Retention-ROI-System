import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Retention Dashboard",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).parent.parent


# ── Load artifacts (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler = pickle.load(open(ROOT / "models/scaler.pkl", "rb"))
    model  = pickle.load(open(ROOT / "models/xgb_calibrated.pkl", "rb"))
    threshold = float(open(ROOT / "models/threshold.txt").read().strip())
    return scaler, model, threshold


@st.cache_data
def load_test_data():
    return pd.read_csv(ROOT / "models/test_scored.csv")


scaler, model, THRESHOLD = load_artifacts()
test_df = load_test_data()

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


def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLS}
    row["SeniorCitizen"]   = inputs["senior_citizen"]
    row["Partner"]         = inputs["partner"]
    row["Dependents"]      = inputs["dependents"]
    row["tenure"]          = inputs["tenure"]
    row["OnlineSecurity"]  = inputs["online_security"]
    row["TechSupport"]     = inputs["tech_support"]
    row["Contract"]        = CONTRACT_MAP[inputs["contract"]]
    row["PaperlessBilling"]= inputs["paperless_billing"]
    row["MonthlyCharges"]  = inputs["monthly_charges"]
    row[f"InternetService_{inputs['internet_service']}"] = 1
    row[f"PaymentMethod_{inputs['payment_method']}"]     = 1
    return pd.DataFrame([row], columns=FEATURE_COLS)


def risk_color(prob):
    if prob >= 0.75:
        return "#e74c3c"   # red
    if prob >= THRESHOLD:
        return "#f39c12"   # orange
    return "#27ae60"       # green


def risk_label(prob):
    if prob >= 0.75:
        return "🔴 HIGH RISK"
    if prob >= THRESHOLD:
        return "🟠 MEDIUM RISK"
    return "🟢 LOW RISK"


# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Model Dashboard", "🔍 Single Customer", "📋 Batch Score"])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MODEL DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Model Dashboard":
    st.title("Churn Retention ROI Dashboard")
    st.caption("XGBoost + Isotonic Calibration | Telco Customer Dataset")

    # ── KPI cards ────────────────────────────────────────────────────────────
    y_true = test_df["actual_churn"]
    y_pred = test_df["predicted_churn"]
    y_prob = test_df["churn_prob_calibrated"]

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC-ROC",  "0.8297")
    c2.metric("AUC-PR",   "0.6164", help="Primary metric — 2.3× above random (0.265)")
    c3.metric("Recall",   f"{recall*100:.1f}%", help="Churners correctly identified")
    c4.metric("Precision",f"{precision*100:.1f}%")
    c5.metric("F1 Score", f"{f1*100:.1f}%")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    with col_left:
        st.subheader("Confusion Matrix")
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"], ax=ax,
            annot_kws={"size": 14},
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(f"Threshold = {THRESHOLD:.4f}", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Probability distribution ─────────────────────────────────────────────
    with col_right:
        st.subheader("Churn Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.hist(
            y_prob[y_true == 0], bins=40, alpha=0.6,
            color="#3498db", label="No Churn", density=True,
        )
        ax2.hist(
            y_prob[y_true == 1], bins=40, alpha=0.6,
            color="#e74c3c", label="Churn", density=True,
        )
        ax2.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold ({THRESHOLD:.2f})")
        ax2.set_xlabel("Calibrated Churn Probability")
        ax2.set_ylabel("Density")
        ax2.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # ── Business impact ───────────────────────────────────────────────────────
    st.subheader("Business Impact Estimate")
    avg_clv = st.slider("Assumed annual CLV per customer ($)", 200, 2000, 777, 50)
    bi1, bi2, bi3 = st.columns(3)
    bi1.metric("Churners Caught", f"{tp} / {tp+fn}",
               help="Customers who would have churned and were correctly flagged")
    bi2.metric("Revenue Protected", f"${tp * avg_clv:,.0f}",
               delta=f"{recall*100:.1f}% recall rate")
    bi3.metric("Missed Revenue", f"${fn * avg_clv:,.0f}",
               delta=f"-{fn} churners not caught", delta_color="inverse")

    st.divider()

    # ── Risk tier breakdown ───────────────────────────────────────────────────
    st.subheader("Risk Tier Breakdown (Test Set)")
    tiers = pd.cut(
        test_df["churn_prob_calibrated"],
        bins=[0, 0.5, 0.75, 1.0],
        labels=["Low (<50%)", "Medium (50–75%)", "High (>75%)"],
    ).value_counts().sort_index()
    fig3, ax3 = plt.subplots(figsize=(6, 2.5))
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    ax3.barh(tiers.index, tiers.values, color=colors)
    for i, v in enumerate(tiers.values):
        ax3.text(v + 5, i, str(v), va="center", fontsize=11)
    ax3.set_xlabel("Number of Customers")
    ax3.set_xlim(0, tiers.max() * 1.15)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE CUSTOMER PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Customer":
    st.title("Single Customer Churn Predictor")
    st.caption("Enter customer details to get a real-time churn risk score.")

    with st.form("customer_form"):
        st.subheader("Customer Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure         = st.number_input("Tenure (months)", 0, 120, 12)
            monthly_charges= st.number_input("Monthly Charges ($)", 10.0, 200.0, 65.0, step=0.5)
            contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment_method   = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            online_security = st.selectbox("Online Security", [0, 1], format_func=lambda x: "Yes" if x else "No")
            tech_support    = st.selectbox("Tech Support",    [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col3:
            senior_citizen  = st.selectbox("Senior Citizen",   [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner         = st.selectbox("Has Partner",       [0, 1], format_func=lambda x: "Yes" if x else "No")
            dependents      = st.selectbox("Has Dependents",    [0, 1], format_func=lambda x: "Yes" if x else "No")
            paperless_billing=st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        inputs = dict(
            senior_citizen=senior_citizen, partner=partner, dependents=dependents,
            tenure=tenure, online_security=online_security, tech_support=tech_support,
            contract=contract, paperless_billing=paperless_billing,
            monthly_charges=monthly_charges, internet_service=internet_service,
            payment_method=payment_method,
        )
        df_row  = build_feature_row(inputs)
        X_scaled = scaler.transform(df_row)
        prob    = float(model.predict_proba(X_scaled)[0, 1])
        churn   = prob >= THRESHOLD

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob*100:.1f}%")
        r2.metric("Decision",          "CHURN" if churn else "RETAIN")
        r3.metric("Risk Tier",         risk_label(prob).split()[-2] + " " + risk_label(prob).split()[-1])

        # Gauge-style bar
        fig, ax = plt.subplots(figsize=(6, 0.8))
        ax.barh([""], [1], color="#ecf0f1", height=0.5)
        ax.barh([""], [prob], color=risk_color(prob), height=0.5)
        ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, THRESHOLD, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", f"Threshold\n{THRESHOLD:.2f}", "75%", "100%"])
        ax.set_yticks([])
        ax.set_title(f"Churn Risk: {prob*100:.1f}%", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if churn:
            st.warning(f"⚠️ This customer is predicted to **churn** (probability {prob*100:.1f}% ≥ threshold {THRESHOLD*100:.1f}%). Consider a retention offer.")
        else:
            st.success(f"✅ This customer is predicted to **stay** (probability {prob*100:.1f}% < threshold {THRESHOLD*100:.1f}%). No immediate action needed.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH SCORING
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Score":
    st.title("Batch Scoring — Test Set Results")
    st.caption(f"Showing {len(test_df):,} customers from the held-out test set. Sorted by churn risk.")

    # Controls
    c1, c2 = st.columns([1, 3])
    risk_filter = c1.selectbox("Filter by Risk Tier", ["All", "High (>75%)", "Medium (50–75%)", "Low (<50%)"])
    min_prob    = c2.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)

    display = test_df[["churn_prob_calibrated", "actual_churn", "predicted_churn"]].copy()
    display.columns = ["Churn Probability", "Actual Churn", "Predicted Churn"]
    display["Risk Tier"] = display["Churn Probability"].apply(
        lambda p: "High" if p >= 0.75 else ("Medium" if p >= THRESHOLD else "Low")
    )
    display["Result"] = display.apply(
        lambda r: "✅ TP" if r["Actual Churn"] == 1 and r["Predicted Churn"] == 1
        else ("❌ FN" if r["Actual Churn"] == 1 and r["Predicted Churn"] == 0
        else ("⚠️ FP" if r["Actual Churn"] == 0 and r["Predicted Churn"] == 1
        else "✓ TN")),
        axis=1,
    )

    if risk_filter == "High (>75%)":
        display = display[display["Risk Tier"] == "High"]
    elif risk_filter == "Medium (50–75%)":
        display = display[display["Risk Tier"] == "Medium"]
    elif risk_filter == "Low (<50%)":
        display = display[display["Risk Tier"] == "Low"]

    display = display[display["Churn Probability"] >= min_prob]
    display = display.sort_values("Churn Probability", ascending=False).reset_index(drop=True)
    display.index += 1

    st.metric("Customers shown", len(display))
    st.dataframe(
        display.style.format({"Churn Probability": "{:.1%}"}),
        use_container_width=True,
        height=450,
    )

    csv = display.to_csv(index=True).encode()
    st.download_button("Download filtered results as CSV", csv, "churn_scores.csv", "text/csv")
