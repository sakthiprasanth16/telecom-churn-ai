"""
ml/fairness.py
---------------
Fairness auditing (project spec Phase 15 / Question 2 item 8: "The model
must not discriminate against customers based on protected attributes like
age, gender, or location. You need to audit for fairness and mitigate
biases.")

This dataset has no location/state field, so the two sensitive attributes
audited here are the two the project actually has: `gender` and
`SeniorCitizen` (the closest available proxy for age -- flagged during EDA
as worth watching specifically). The same logic, duplicated intentionally,
also appears inline in notebooks/03_model_training.ipynb's Section 12 --
see ml/preprocess.py's docstring for why notebooks stay self-contained
rather than importing this module.

Two audit modes, because they answer genuinely different questions and
need different data:

1. `audit_with_ground_truth()` -- valid ONLY where true labels exist (the
   held-out test split at training time; see ml/train.py). A real fairness
   audit: selection rate (demographic parity), recall/true positive rate,
   and false positive rate per group, plus a disparate impact ratio for
   each -- covering demographic parity AND both halves of "equalized odds."

2. `audit_selection_rate_only()` -- for LIVE production predictions, which
   have no ground truth yet (backend/database.py's `actual_outcome` field
   is always None -- no labeled-feedback loop exists). Without true
   labels, only demographic parity (selection rate) can be measured at
   all; recall and false-positive-rate parity literally can't be computed
   without knowing who actually churned. Labeled as a narrower, proxy
   signal everywhere it's surfaced (see backend/main.py's GET /fairness).

Why a disparate impact RATIO (min-group / max-group) rather than a raw
difference: it's scale-invariant and matches the "four-fifths rule" (ratio
>= 0.80), a standard (if informal outside strict legal contexts) threshold
used in US employment/credit-fairness practice -- same reasoning ml/drift.py
uses for PSI over a raw mean difference: an interpretable, industry-
recognized cutoff, not an arbitrary number invented for this project.

Also provided: `suggest_group_thresholds()`, a documented, OPTIONAL bias
-mitigation technique (post-processing per-group probability thresholds
targeting equal recall). This is a RECOMMENDATION surfaced in reports --
see its docstring for why it is never applied inside backend/predictor.py's
live serving path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

SENSITIVE_ATTRIBUTES = ["gender", "SeniorCitizen"]

# The "four-fifths rule" -- see module docstring.
DISPARITY_RATIO_THRESHOLD = 0.80

# A group's rate below this sample size is too noisy to trust -- same
# reasoning as ml/drift.py's MIN_SAMPLE_SIZE.
MIN_GROUP_SIZE = 30


def _disparity_ratio(group_values: dict[str, float]) -> float | None:
    values = [v for v in group_values.values() if v is not None]
    # A ratio needs at least two groups to compare -- with only one group
    # meeting MIN_GROUP_SIZE, min/max collapse to the same value and would
    # trivially return 1.0 ("no disparity"), which misrepresents "we
    # couldn't check" as "we checked and it's fine."
    if len(values) < 2 or max(values) == 0:
        return None
    return round(min(values) / max(values), 4)


def audit_with_ground_truth(
    sensitive_series: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Full fairness audit for ONE sensitive attribute, using true labels --
    only valid where ground truth exists (a held-out test set). If
    `y_proba` is given, also includes a mitigation recommendation (see
    suggest_group_thresholds()).
    """
    sensitive_series = sensitive_series.astype(str)
    groups = sensitive_series.unique().tolist()

    selection_rate: dict[str, float] = {}
    recall: dict[str, float] = {}
    false_positive_rate: dict[str, float] = {}
    group_sizes: dict[str, int] = {}

    for group in groups:
        mask = (sensitive_series == group).values
        group_sizes[group] = int(mask.sum())
        if group_sizes[group] < MIN_GROUP_SIZE:
            continue  # too small to report a trustworthy rate

        y_true_g = np.asarray(y_true)[mask]
        y_pred_g = np.asarray(y_pred)[mask]

        selection_rate[group] = round(float(y_pred_g.mean()), 4)
        recall[group] = round(float(recall_score(y_true_g, y_pred_g, zero_division=0)), 4)

        negatives_mask = y_true_g == 0
        if negatives_mask.sum() > 0:
            false_positive_rate[group] = round(float(y_pred_g[negatives_mask].mean()), 4)

    disparities = {
        "selection_rate_ratio": _disparity_ratio(selection_rate),
        "recall_ratio": _disparity_ratio(recall),
        "false_positive_rate_ratio": _disparity_ratio(false_positive_rate),
    }
    flagged = any(
        ratio is not None and ratio < DISPARITY_RATIO_THRESHOLD for ratio in disparities.values()
    )

    report = {
        "group_sizes": group_sizes,
        "selection_rate": selection_rate,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "disparities": disparities,
        "flagged": flagged,
    }

    if y_proba is not None:
        report["mitigation"] = suggest_group_thresholds(
            sensitive_series, y_true, y_proba, report
        )

    return report


def suggest_group_thresholds(
    sensitive_series: pd.Series,
    y_true: pd.Series,
    y_proba: np.ndarray,
    report: dict[str, Any],
) -> dict[str, Any]:
    """
    Demonstrates ONE standard post-processing bias-mitigation technique:
    per-group probability thresholds targeting equal recall across groups
    (lowering the threshold for an under-served group brings its detection
    rate up to match the best-served group).

    This is a RECOMMENDATION for discussion, never applied inside
    backend/predictor.py's live serving path, which uses one 0.5 threshold
    for every customer. Adopting per-group thresholds is itself a
    fairness-vs-consistency trade-off: two customers with an identical
    predicted risk score could be treated differently depending on which
    group they're in. There's a well-known result showing several fairness
    definitions (e.g. equal recall and equal selection rate) generally
    can't all be satisfied at once except in special cases -- which one to
    prioritize is a business/ethics decision, not a purely technical one.
    """
    valid_groups = [g for g, size in report["group_sizes"].items() if size >= MIN_GROUP_SIZE]
    if len(valid_groups) < 2 or not report["recall"]:
        return {"note": "Not enough groups with sufficient sample size to suggest thresholds."}

    sensitive_series = sensitive_series.astype(str)
    target_recall = max(report["recall"].values())

    suggested_thresholds: dict[str, float] = {}
    simulated_selection_rate: dict[str, float] = {}

    for group in valid_groups:
        mask = (sensitive_series == group).values
        y_true_g = np.asarray(y_true)[mask]
        y_proba_g = np.asarray(y_proba)[mask]

        candidate_thresholds = sorted(np.unique(y_proba_g), reverse=True)
        chosen_t = candidate_thresholds[-1] if candidate_thresholds else 0.5
        for t in candidate_thresholds:
            preds = (y_proba_g >= t).astype(int)
            if recall_score(y_true_g, preds, zero_division=0) >= target_recall:
                chosen_t = t
                break

        suggested_thresholds[group] = round(float(chosen_t), 4)
        simulated_selection_rate[group] = round(float((y_proba_g >= chosen_t).mean()), 4)

    return {
        "strategy": "per-group probability threshold adjustment (post-processing), targeting equal recall",
        "target_recall": round(target_recall, 4),
        "suggested_thresholds": suggested_thresholds,
        "simulated_selection_rate_after_mitigation": simulated_selection_rate,
        "simulated_selection_rate_ratio_after_mitigation": _disparity_ratio(simulated_selection_rate),
        "note": "Recommendation only -- NOT applied in the live serving path.",
    }


def audit_selection_rate_only(sensitive_series: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """
    Demographic-parity-only audit for live production predictions, where
    no ground truth exists yet. See module docstring for why recall and
    false-positive-rate parity can't be computed here.
    """
    sensitive_series = sensitive_series.astype(str)
    groups = sensitive_series.unique().tolist()

    selection_rate: dict[str, float] = {}
    group_sizes: dict[str, int] = {}

    for group in groups:
        mask = (sensitive_series == group).values
        group_sizes[group] = int(mask.sum())
        if group_sizes[group] < MIN_GROUP_SIZE:
            continue
        selection_rate[group] = round(float(np.asarray(y_pred)[mask].mean()), 4)

    ratio = _disparity_ratio(selection_rate)
    return {
        "group_sizes": group_sizes,
        "selection_rate": selection_rate,
        "selection_rate_ratio": ratio,
        "flagged": ratio is not None and ratio < DISPARITY_RATIO_THRESHOLD,
    }


def audit_all_attributes_with_ground_truth(
    raw_df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, Any]:
    """Runs audit_with_ground_truth() for every attribute in SENSITIVE_ATTRIBUTES present in raw_df."""
    return {
        attr: audit_with_ground_truth(raw_df[attr], y_true, y_pred, y_proba)
        for attr in SENSITIVE_ATTRIBUTES
        if attr in raw_df.columns
    }


def audit_all_attributes_selection_rate_only(raw_df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, Any]:
    """Runs audit_selection_rate_only() for every attribute in SENSITIVE_ATTRIBUTES present in raw_df."""
    return {
        attr: audit_selection_rate_only(raw_df[attr], y_pred)
        for attr in SENSITIVE_ATTRIBUTES
        if attr in raw_df.columns
    }
