"""
frontend/pages/1_Monitoring.py
-------------------------------
Monitoring dashboard. Streamlit auto-detects files in a `pages/` folder and
adds them to the sidebar navigation, so this shows up as a "Monitoring" page
alongside the main "app" (Prediction) page automatically -- no extra routing
code needed.

This page reads aggregated data from the backend's GET /metrics and
GET /model-info endpoints -- it never queries MongoDB directly, keeping the
same separation of concerns as the prediction page (Streamlit only ever
talks to FastAPI).

Some sections are placeholders for now, honestly labeled as such, because
their underlying features (drift detection, fairness auditing, business
metrics) are later phases in the build order (12, 15, 16) that haven't been
built yet. Per the project's engineering rules, this page does not fabricate
numbers for features that don't exist yet.
"""

from __future__ import annotations

import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Churn Monitoring", page_icon="📈", layout="centered")

st.title("📈 Model Monitoring")
st.caption("Live view of prediction activity and model health")


def get_json(path: str) -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        return None
    return None


health = get_json("/health")
metrics = get_json("/metrics")
model_info = get_json("/model-info")

if health is None:
    st.error(f"⚠️ Cannot reach the prediction API at `{BACKEND_URL}`.")
    st.stop()

# ---------------------------------------------------------------------------
# System health
# ---------------------------------------------------------------------------
st.subheader("System Health")
h1, h2, h3 = st.columns(3)
h1.metric("API", "🟢 Online" if health.get("status") == "ok" else "🔴 Degraded")
h2.metric("Model Loaded", "🟢 Yes" if health.get("model_loaded") else "🔴 No")
h3.metric("Database", "🟢 Connected" if health.get("mongo_connected") else "🟡 Degraded")

st.divider()

# ---------------------------------------------------------------------------
# Prediction activity
# ---------------------------------------------------------------------------
st.subheader("Prediction Activity")

if metrics is None or metrics.get("total_predictions", 0) == 0:
    st.info(
        "No predictions logged yet. Make a prediction on the main page, "
        "then come back here to see it reflected in these stats."
    )
else:
    a1, a2, a3 = st.columns(3)
    a1.metric("Predictions Today", metrics["predictions_today"])
    a2.metric("Total Predictions (all time)", metrics["total_predictions"])
    a3.metric("Avg. API Latency", f"{metrics['avg_latency_ms']} ms")

    st.divider()

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.write("**Prediction Distribution**")
        churn = metrics["churn_count"]
        no_churn = metrics["no_churn_count"]
        total = churn + no_churn
        if total > 0:
            st.write(f"🔴 CHURN: {churn} ({churn/total*100:.1f}%)")
            st.progress(churn / total)
            st.write(f"🟢 NO_CHURN: {no_churn} ({no_churn/total*100:.1f}%)")
            st.progress(no_churn / total)

    with dist_col2:
        st.write("**Risk Level Breakdown**")
        rb = metrics["risk_breakdown"]
        risk_total = rb["HIGH"] + rb["MEDIUM"] + rb["LOW"]
        if risk_total > 0:
            st.write(f"🔴 HIGH: {rb['HIGH']} ({rb['HIGH']/risk_total*100:.1f}%)")
            st.progress(rb["HIGH"] / risk_total)
            st.write(f"🟡 MEDIUM: {rb['MEDIUM']} ({rb['MEDIUM']/risk_total*100:.1f}%)")
            st.progress(rb["MEDIUM"] / risk_total)
            st.write(f"🟢 LOW: {rb['LOW']} ({rb['LOW']/risk_total*100:.1f}%)")
            st.progress(rb["LOW"] / risk_total)

st.divider()

# ---------------------------------------------------------------------------
# Model performance (from the last training run's held-out test metrics)
# ---------------------------------------------------------------------------
st.subheader("Model Performance")
st.caption("Metrics from the model's evaluation on held-out test data at training time.")

if model_info:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{model_info['precision']*100:.1f}%")
    m2.metric("Recall", f"{model_info['recall']*100:.1f}%")
    m3.metric("F1 Score", f"{model_info['f1']*100:.1f}%")
    m4.metric("ROC-AUC", f"{model_info['roc_auc']*100:.1f}%")

    st.caption(
        f"Model: `{model_info['model_type']}` (version `{model_info['version']}`) "
        f"· Trained: {model_info['trained_at'][:10]} · "
        f"Dataset: `{model_info['dataset_version']}` · Features: `{model_info['feature_version']}` · "
        f"Status: `{model_info['status']}`"
    )
else:
    st.warning("Could not load model info.")

st.divider()

# ---------------------------------------------------------------------------
# Not yet built - honestly labeled placeholders
# ---------------------------------------------------------------------------
st.subheader("Data & Model Drift")

drift = get_json("/drift")
_STATUS_ICONS = {"no_drift": "🟢", "moderate_drift": "🟡", "significant_drift": "🔴"}

if drift is None:
    st.warning("Could not load drift report from the API.")
elif drift.get("status") == "insufficient_data":
    st.info(
        f"ℹ️ {drift.get('message', 'Not enough recent predictions yet for a drift check.')}"
    )
else:
    status = drift["status"]
    icon = _STATUS_ICONS.get(status, "⚪")

    d1, d2, d3 = st.columns(3)
    d1.metric("Drift Status", f"{icon} {status.replace('_', ' ').title()}")
    d2.metric("Max PSI", drift.get("overall_max_psi"))
    d3.metric("Recent Predictions Checked", drift.get("n_current_rows"))

    drifted = drift.get("drifted_features") or []
    if drifted:
        st.warning(f"**Drifted features:** {', '.join(drifted)}")
    else:
        st.caption("No individual features currently show drift.")

    with st.expander("Per-feature PSI breakdown"):
        all_features = {**drift.get("numeric_features", {}), **drift.get("categorical_features", {})}
        for feat, result in all_features.items():
            feat_icon = _STATUS_ICONS.get(result["status"], "⚪")
            st.write(f"{feat_icon} **{feat}** — PSI: {result['psi']} ({result['status'].replace('_', ' ')})")
            if result.get("unseen_categories"):
                st.caption(f"  New categories not seen in training: {result['unseen_categories']}")

    thresholds = drift.get("thresholds") or {}
    if thresholds:
        st.caption(
            f"Thresholds: PSI < {thresholds.get('moderate')} = no drift · "
            f"{thresholds.get('moderate')}–{thresholds.get('significant')} = moderate · "
            f"≥ {thresholds.get('significant')} = significant"
        )

st.subheader("Fairness Metrics")

fairness = get_json("/fairness")
if fairness is None:
    st.warning("Could not load fairness report from the API.")
else:
    st.caption(
        "Audited attributes: gender and SeniorCitizen (closest available proxy for "
        "age -- this dataset has no location field)."
    )

    st.write("**Training-time audit** (ground-truth-backed, from the active model's last training run)")
    if fairness.get("training_report") is None:
        st.info(f"ℹ️ {fairness.get('training_report_note', 'Not available for the active model.')}")
    else:
        for attr, report in fairness["training_report"].items():
            flag_icon = "🔴" if report["flagged"] else "🟢"
            st.write(f"{flag_icon} **{attr}**")
            cols = st.columns(3)
            cols[0].metric("Selection rate ratio", report["disparities"].get("selection_rate_ratio"))
            cols[1].metric("Recall ratio", report["disparities"].get("recall_ratio"))
            cols[2].metric("FPR ratio", report["disparities"].get("false_positive_rate_ratio"))
            with st.expander(f"{attr} — per-group breakdown"):
                st.json(report)

    st.divider()
    st.write("**Live snapshot** (recent predictions -- selection rate only, no ground truth available yet)")
    st.caption(
        "This can only show demographic parity (are some groups flagged as "
        "high-risk more often?) -- not recall or false-positive-rate parity, "
        "which require knowing who actually churned."
    )
    if fairness.get("live_snapshot") is None:
        st.info(f"ℹ️ {fairness.get('live_snapshot_note', 'Not enough recent predictions yet.')}")
    else:
        for attr, report in fairness["live_snapshot"].items():
            flag_icon = "🔴" if report["flagged"] else "🟢"
            st.write(
                f"{flag_icon} **{attr}** — selection rate ratio: "
                f"{report['selection_rate_ratio']} · groups: {report['selection_rate']}"
            )

st.subheader("Business Impact")

biz = get_json("/business-impact")
if biz is None:
    st.warning("Could not load business impact report from the API.")
elif biz.get("training_report") is None:
    st.info(f"ℹ️ {biz.get('training_report_note', 'Not available for the active model.')}")
else:
    report = biz["training_report"]
    st.caption(
        "⚠️ SIMULATED, not measured — retention rate, intervention cost, and "
        "customer lifetime are adjustable business assumptions, not verified "
        "campaign results. Only the confusion matrix and average monthly "
        "revenue below come from real data."
    )

    a1, a2, a3 = st.columns(3)
    a1.metric("Avg. monthly revenue", f"${report['assumptions']['avg_monthly_revenue']}")
    a2.metric("Assumed retention success", f"{report['assumptions']['retention_success_rate']*100:.0f}%")
    a3.metric("Assumed cost/contact", f"${report['assumptions']['intervention_cost_per_contact']}")

    st.write("**Scenario comparison** (compare by ROI, not just net benefit — see note below)")
    scenario_rows = []
    for name, s in report["scenarios"].items():
        scenario_rows.append({
            "Scenario": name.replace("_", " ").title(),
            "Customers contacted": s["customers_flagged"],
            "Est. retained": s["expected_customers_retained"],
            "Revenue saved": f"${s['estimated_revenue_saved']:,.0f}",
            "Cost": f"${s['estimated_intervention_cost']:,.0f}",
            "Net benefit": f"${s['estimated_net_benefit']:,.0f}",
            "ROI": f"{s['roi']}x" if s["roi"] is not None else "—",
        })
    st.table(scenario_rows)
    st.caption(f"Missed revenue at risk (false negatives): ${report['missed_revenue_at_risk']:,.0f}")

    if biz.get("scaled_report"):
        scaled = biz["scaled_report"]
        with st.expander(f"Scaled projection to {scaled['target_population']:,} customers"):
            st.caption(scaled["note"])
            scaled_rows = [
                {
                    "Scenario": name.replace("_", " ").title(),
                    "Net benefit": f"${s['estimated_net_benefit']:,.0f}",
                    "ROI": f"{s['roi']}x" if s["roi"] is not None else "—",
                }
                for name, s in scaled["scenarios"].items()
            ]
            st.table(scaled_rows)

    if biz.get("live_projection"):
        lp = biz["live_projection"]
        with st.expander("Live projection (from recent prediction volume)"):
            st.caption(lp["note"])
            st.write(
                f"Based on {lp['based_on_total_predictions']} logged predictions, "
                f"{lp['based_on_churn_flagged']} flagged CHURN, at "
                f"{lp['training_precision_used']*100:.1f}% training-time precision:"
            )
            st.write(f"Projected net benefit so far: **${lp['projected_net_benefit']:,.0f}**")
    elif biz.get("live_projection_note"):
        st.caption(f"Live projection: {biz['live_projection_note']}")
