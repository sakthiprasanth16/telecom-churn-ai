"""
frontend/app.py
----------------
Streamlit UI for a customer-service representative to get a real-time churn
prediction for a customer.

Architecture (per project spec):
    User -> Streamlit -> FastAPI -> Pydantic validation -> Preprocessing
    -> Already-loaded ML model -> Prediction -> Explanation -> MongoDB logging
    -> Response to Streamlit

This file only talks to the FastAPI backend over HTTP -- it never loads the
model or touches MongoDB directly. That separation matters: Streamlit can be
redeployed, scaled, or swapped out independently of the prediction service.

BACKEND_URL is read from an environment variable rather than hardcoded, so
this same code works locally (http://127.0.0.1:8000) and in production
(the real deployed Render URL) without any code changes -- only the
environment variable differs between environments.
"""

from __future__ import annotations

import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header + backend health check
# ---------------------------------------------------------------------------
st.title("📊 Telecom Customer Churn Prediction")
st.caption("Real-time churn risk assessment for customer-service representatives")


def check_backend_health() -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        return None
    return None


health = check_backend_health()

if health is None:
    st.error(
        f"⚠️ Cannot reach the prediction API at `{BACKEND_URL}`. "
        "Make sure the FastAPI backend is running."
    )
    st.stop()
elif not health.get("model_loaded"):
    st.error("⚠️ The prediction API is reachable, but the model failed to load.")
    st.stop()
# If everything is healthy, we say nothing -- no need to show a status row
# every time. A failure still surfaces clearly above, since st.stop() halts
# the rest of the page from rendering.

# ---------------------------------------------------------------------------
# Customer input form
# ---------------------------------------------------------------------------
st.subheader("Customer Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=8)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=560.0, step=1.0)

    submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Submit -> call FastAPI -> display result
# ---------------------------------------------------------------------------
if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the prediction API: {e}")
        st.stop()

    if response.status_code == 422:
        st.error("The API rejected this input as invalid:")
        for err in response.json().get("detail", []):
            field = " -> ".join(str(p) for p in err.get("loc", []))
            st.write(f"- **{field}**: {err.get('msg')}")
        st.stop()
    elif response.status_code != 200:
        st.error(f"Prediction failed (HTTP {response.status_code}): {response.text}")
        st.stop()

    result = response.json()

    st.divider()
    st.subheader("Prediction Result")

    risk = result["risk"]
    risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    risk_color = risk_colors.get(risk, "⚪")

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Prediction", result["prediction"])
    result_col2.metric("Churn Probability", f"{result['probability']*100:.1f}%")
    result_col3.metric("Risk Level", f"{risk_color} {risk}")

    st.progress(min(result["probability"], 1.0))

    if risk == "HIGH":
        st.error(
            "⚠️ **High churn risk.** Consider proactive retention outreach "
            "(loyalty offer, contract upgrade discussion, service review call)."
        )
    elif risk == "MEDIUM":
        st.warning("🟡 **Moderate churn risk.** Worth monitoring; a light-touch check-in may help.")
    else:
        st.success("🟢 **Low churn risk.** No action needed at this time.")

    st.divider()
    st.subheader("Why this prediction?")
    st.caption("Top factors influencing this customer's risk score:")

    for factor in result.get("top_factors", []):
        icon = "⬆️" if factor["direction"] == "increases_risk" else "⬇️"
        st.write(f"{icon} {factor['human_readable']}")

    if not result.get("top_factors"):
        st.caption("No explanation available for this model type.")

    st.divider()
    meta_col1, meta_col2 = st.columns(2)
    meta_col1.caption(f"Model version: `{result['model_version']}`")
    meta_col2.caption(f"Prediction API latency: **{result['latency_ms']} ms**")
    st.caption(
        "Note: this is the FastAPI inference latency only, measured server-side. "
        "It does not include network time between this app and the API."
    )