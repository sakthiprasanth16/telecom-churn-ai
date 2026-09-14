"""
ml/train.py
-----------
Reusable model-training logic, mirroring notebooks/03_model_training.ipynb's
pipeline exactly: LogisticRegression, class_weight='balanced', and --
important -- the same Pipeline/ColumnTransformer shape backend/predictor.py's
explainability code depends on. `predictor.py._top_factors()` reaches
directly into `pipeline.named_steps["preprocessor"].named_transformers_["num"]`
and `pipeline.named_steps["classifier"].coef_`, so a candidate model produced
here must keep those exact step names and the numeric-features-first column
order (ml.preprocess.NUMERIC_FEATURES + CATEGORICAL_FEATURES) or a promoted
candidate would silently break the "Why this prediction?" explanations.

Why this exists as a module rather than only living in the notebook: the
notebook is intentionally self-contained (project rule -- see
ml/preprocess.py's docstring for the same reasoning applied to feature
engineering: notebooks are for reading/explaining, never imported from).
Automated retraining (Phase 13's LangGraph workflow, ml/retraining_graph.py)
needs to call this same training logic programmatically -- on a drift
trigger or a schedule -- without a human re-running a notebook cell by cell.

If this drifts out of sync with notebook 03, an automated retrain could
produce a materially different model than what the notebook documents and
the interview explains. Treat this file as the source of truth; if the
model architecture changes here, update notebook 03 to match (or better,
have the notebook narrate this file's logic rather than duplicate it).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES, engineer_features
from ml.fairness import audit_all_attributes_with_ground_truth
from ml.business_impact import compute_confusion_and_revenue

# Matches the Kaggle Telco Customer Churn dataset's label column exactly
# (values "Yes"/"No"). If your data/processed/churn_cleaned_v1.csv uses a
# different column name or encoding, update this constant -- don't
# hand-duplicate this file, per the same reasoning as ml/preprocess.py.
TARGET_COLUMN = "Churn"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def _build_pipeline() -> Pipeline:
    """
    Same shape as notebook 03 and as backend/predictor.py assumes: a
    ColumnTransformer named "preprocessor" with a "num" transformer
    (StandardScaler, numeric features first) and a "cat" transformer
    (OneHotEncoder), feeding a LogisticRegression named "classifier".
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(clean_df: pd.DataFrame) -> dict[str, Any]:
    """
    Trains one candidate model from an already-cleaned dataset -- same raw,
    Kaggle-shaped columns as data/processed/churn_cleaned_v1.csv, including
    the TARGET_COLUMN label. Feature engineering is run here via
    ml.preprocess.engineer_features(), exactly like predictor.py does for a
    single live prediction, so training and inference never diverge.

    Returns the fitted pipeline plus held-out test metrics in the same
    shape as models/model_v1_metadata.json's metric fields, so a candidate
    can be compared apples-to-apples against the current production model
    (see ml/retraining_graph.py's evaluate_candidate step).

    Raises a clear ValueError (rather than a confusing KeyError deep inside
    sklearn) if the expected label column isn't present -- an automated
    retraining run should fail loudly and diagnosably, not silently train
    on the wrong thing.
    """
    if TARGET_COLUMN not in clean_df.columns:
        raise ValueError(
            f"Expected label column '{TARGET_COLUMN}' not found in training data. "
            f"Available columns: {list(clean_df.columns)}"
        )

    df = engineer_features(clean_df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_raw = df[TARGET_COLUMN]
    # Checking `dtype == object` alone is not reliable for detecting a
    # string label column on pandas >= 2.x/3.x, which may store strings
    # under a dedicated StringDtype rather than legacy `object` -- using
    # is_numeric_dtype() to decide the branch instead avoids silently
    # trying (and failing) to int-cast "Yes"/"No" strings.
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
    else:
        y = (y_raw == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    # Ground-truth-backed fairness audit (Phase 15) -- computed on the same
    # held-out test set, since X_test already contains the raw "gender" and
    # "SeniorCitizen" columns the audit needs (see ml/fairness.py). This is
    # the only point in the pipeline with true labels to check against --
    # backend/main.py's GET /fairness can only do a narrower, live
    # demographic-parity check for the same reason.
    fairness_report = audit_all_attributes_with_ground_truth(X_test, y_test, y_pred, y_proba)

    # Business-impact inputs (Phase 16) -- the confusion matrix and average
    # monthly revenue are real, data-derived numbers; the dollar-figure
    # SIMULATION built from these (with configurable business assumptions)
    # happens later, in ml.business_impact.simulate_business_impact(), so
    # a caller (GET /business-impact) can recompute it with different
    # assumptions without retraining. See ml/business_impact.py's docstring.
    business_impact_inputs = compute_confusion_and_revenue(y_test, y_pred, X_test["MonthlyCharges"])

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fairness_report": fairness_report,
        "business_impact_inputs": business_impact_inputs,
    }
