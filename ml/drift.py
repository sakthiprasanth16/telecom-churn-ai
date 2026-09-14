"""
ml/drift.py
-----------
Statistical data drift detection using PSI (Population Stability Index) --
a lightweight, well-established technique that avoids pulling in a heavy
dependency like Evidently, per the project spec's guidance to use "Evidently
or a lightweight statistical drift implementation."

Why PSI and not just "compare averages": a shift in the mean can hide a
shift in the shape of the distribution (e.g. the average stays the same but
the data becomes bimodal). PSI compares the full distribution, bucket by
bucket, which catches that kind of shift too.

Why PSI, specifically, and not a KS-test or chi-square test: PSI is the
standard metric used in industry (especially credit risk / churn modeling)
specifically because it comes with widely-accepted, interpretable
thresholds (see PSI_THRESHOLDS below) that a non-statistician stakeholder
can be told directly, which matters for a monitoring dashboard a
customer-service ops manager might look at, not just a data scientist.

What this module compares: the RAW, Kaggle-shaped input fields (tenure,
MonthlyCharges, Contract, etc.) -- i.e. exactly what a customer-service rep
types into the Streamlit form and what the FastAPI backend logs into
MongoDB's `predictions` collection. It does NOT compare the engineered
features (tenure_bucket, avg_monthly_spend, etc.), because those are
mechanically derived from the raw fields -- if the raw fields haven't
drifted, the engineered features can't have drifted either, so comparing
both would be redundant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Numeric fields get bucketed into quantile bins before PSI is computed.
REFERENCE_NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Categorical fields are compared bucket-for-bucket on their existing categories.
REFERENCE_CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

N_QUANTILE_BINS = 10

# Industry-standard PSI interpretation thresholds (widely used in credit
# risk / churn modeling, e.g. Siddiqi 2006 and common industry practice):
#   PSI < 0.10           -> no significant drift
#   0.10 <= PSI < 0.25    -> moderate drift, worth investigating
#   PSI >= 0.25           -> significant drift, action recommended
# These are configurable (not hardcoded blindly) via the constants below so
# they can be tuned per-deployment, but the defaults are the standard,
# defensible starting point rather than an arbitrary made-up number.
PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25

# Minimum number of recent predictions required before a drift report is
# considered statistically meaningful. Below this, PSI on a tiny sample is
# noisy and can flag "drift" that's really just small-sample randomness.
MIN_SAMPLE_SIZE = 30

_EPSILON = 1e-4  # avoids divide-by-zero / log(0) for empty buckets


def _psi_from_proportions(ref_props: np.ndarray, cur_props: np.ndarray) -> float:
    ref = np.clip(ref_props, _EPSILON, None)
    cur = np.clip(cur_props, _EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def build_reference_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build the reference (baseline) distribution snapshot from the training
    dataset. Run ONCE, offline, whenever a new model is trained on a new
    dataset version -- this is what "reference data" means throughout this
    module. Saved to disk so it doesn't need to be recomputed on every
    drift check.
    """
    stats: dict[str, Any] = {"numeric": {}, "categorical": {}, "n_reference_rows": len(df)}

    for col in REFERENCE_NUMERIC_FEATURES:
        series = df[col].dropna()
        # Quantile-based bin edges so each reference bucket has ~equal mass.
        quantiles = np.linspace(0, 1, N_QUANTILE_BINS + 1)
        edges = np.unique(series.quantile(quantiles).values)
        if len(edges) < 2:
            # Degenerate case (constant column) -- skip, nothing to compare.
            continue
        counts, _ = np.histogram(series, bins=edges)
        proportions = (counts / counts.sum()).tolist()
        stats["numeric"][col] = {
            "bin_edges": edges.tolist(),
            "reference_proportions": proportions,
            "reference_mean": float(series.mean()),
            "reference_std": float(series.std()),
        }

    for col in REFERENCE_CATEGORICAL_FEATURES:
        series = df[col].astype(str)
        value_counts = series.value_counts(normalize=True)
        stats["categorical"][col] = {
            "reference_proportions": value_counts.to_dict(),
        }

    return stats


def save_reference_stats(stats: dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_reference_stats(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _numeric_feature_drift(feature_stats: dict, current_series: pd.Series) -> dict[str, Any]:
    edges = np.array(feature_stats["bin_edges"])
    ref_props = np.array(feature_stats["reference_proportions"])

    current_series = current_series.dropna()
    counts, _ = np.histogram(current_series, bins=edges)
    total = counts.sum()
    cur_props = counts / total if total > 0 else np.zeros_like(ref_props, dtype=float)

    psi = _psi_from_proportions(ref_props, cur_props)

    return {
        "psi": round(psi, 4),
        "reference_mean": feature_stats["reference_mean"],
        "current_mean": round(float(current_series.mean()), 2) if len(current_series) else None,
        "status": _psi_status(psi),
    }


def _categorical_feature_drift(feature_stats: dict, current_series: pd.Series) -> dict[str, Any]:
    ref_dist: dict[str, float] = feature_stats["reference_proportions"]
    categories = list(ref_dist.keys())

    current_series = current_series.astype(str)
    cur_dist = current_series.value_counts(normalize=True).to_dict()

    ref_props = np.array([ref_dist.get(c, 0.0) for c in categories])
    cur_props = np.array([cur_dist.get(c, 0.0) for c in categories])

    # Any category present now but never seen in reference data is itself
    # a signal worth surfacing (e.g. a new payment method being offered).
    unseen_categories = [c for c in cur_dist if c not in ref_dist]

    psi = _psi_from_proportions(ref_props, cur_props)

    return {
        "psi": round(psi, 4),
        "status": _psi_status(psi),
        "unseen_categories": unseen_categories,
    }


def _psi_status(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant_drift"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate_drift"
    return "no_drift"


def detect_drift(reference_stats: dict[str, Any], current_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compare `current_df` (recent live prediction inputs) against the saved
    reference distribution. Returns a per-feature breakdown plus an overall
    status. `current_df` should contain the same raw columns the reference
    was built from.
    """
    n_current = len(current_df)

    if n_current < MIN_SAMPLE_SIZE:
        return {
            "status": "insufficient_data",
            "n_reference_rows": reference_stats.get("n_reference_rows"),
            "n_current_rows": n_current,
            "message": (
                f"Only {n_current} recent predictions available; need at "
                f"least {MIN_SAMPLE_SIZE} for a statistically meaningful "
                "drift comparison. This is not the same as 'no drift' -- "
                "it means there isn't yet enough data to tell."
            ),
            "numeric_features": {},
            "categorical_features": {},
        }

    numeric_results = {}
    for col, feature_stats in reference_stats.get("numeric", {}).items():
        if col in current_df.columns:
            numeric_results[col] = _numeric_feature_drift(feature_stats, current_df[col])

    categorical_results = {}
    for col, feature_stats in reference_stats.get("categorical", {}).items():
        if col in current_df.columns:
            categorical_results[col] = _categorical_feature_drift(feature_stats, current_df[col])

    all_psi = [v["psi"] for v in numeric_results.values()] + [v["psi"] for v in categorical_results.values()]
    max_psi = max(all_psi) if all_psi else 0.0
    overall_status = _psi_status(max_psi)

    drifted_features = [
        col for col, v in {**numeric_results, **categorical_results}.items()
        if v["status"] != "no_drift"
    ]

    return {
        "status": overall_status,
        "overall_max_psi": round(max_psi, 4),
        "n_reference_rows": reference_stats.get("n_reference_rows"),
        "n_current_rows": n_current,
        "drifted_features": drifted_features,
        "numeric_features": numeric_results,
        "categorical_features": categorical_results,
        "thresholds": {
            "moderate": PSI_MODERATE_THRESHOLD,
            "significant": PSI_SIGNIFICANT_THRESHOLD,
        },
    }
