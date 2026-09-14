"""
ml/preprocess.py
-----------------
Shared feature-engineering logic.

Why this file exists (and isn't just duplicated inside notebook 02 and the backend):
the whole point of `02_feature_engineering.ipynb` was to *design* these features and
validate them against the EDA findings. Once they were validated, that logic needs to
be reusable in exactly one place, because it will be called twice in the running
system:

1. Offline, when retraining a model on a fresh batch of data (ml/train.py, and the
   LangGraph retraining workflow later).
2. Online, on every single incoming prediction request in the FastAPI backend
   (backend/predictor.py).

If this logic lived only inside the notebook, someone would eventually hand-copy it
into the backend, and the two copies would drift the moment either one changed. That
drift is exactly what rule #9 in the project spec warns against: "do not manually
preprocess training and inference differently." So this file is the notebook's logic,
lifted out unchanged, callable from anywhere.

`engineer_features()` intentionally mirrors the cell-by-cell logic in
`notebooks/02_feature_engineering.ipynb` exactly. If you change one, change the other,
or better: re-run notebook 02 against this function directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature groups, in the exact form the trained model pipeline expects them.
# These lists must match what was used in 03_model_training.ipynb, since the
# saved model's ColumnTransformer was fit against columns in this shape.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "num_services_subscribed",
    "contract_risk_score",
    "SeniorCitizen",
    "is_new_customer",
    "has_internet_and_phone",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]

MODEL_FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

_TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
_TENURE_LABELS = ["0-12", "12-24", "24-36", "36-48", "48-60", "60-72"]

_CONTRACT_RISK_MAP = {
    "Month-to-month": 2,
    "One year": 1,
    "Two year": 0,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the six engineered features on top of already-clean input.

    Accepts a DataFrame containing at least the original Kaggle-shaped columns
    (tenure, MonthlyCharges, TotalCharges, Contract, PhoneService, InternetService,
    and the individual service Yes/No columns). Returns a new DataFrame with the
    engineered columns added; does not mutate the input.

    This function assumes `df` is already clean (no blank TotalCharges, no
    "No internet/phone service" placeholder categories) -- for a single live
    prediction request this is enforced by Pydantic validation in
    backend/schemas.py instead of the notebook-01-style batch cleaning, since a
    single incoming customer record has no duplicates/outliers/blank strings to
    detect in the first place.
    """
    df = df.copy()

    # tenure_bucket
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=_TENURE_BINS, labels=_TENURE_LABELS, include_lowest=True
    )
    # A brand-new customer request could have tenure outside all bins only if
    # tenure < 0, which Pydantic already rejects -- but guard defensively so a
    # missing bucket never silently breaks the ColumnTransformer's one-hot encoder.
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)

    # is_new_customer
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    # num_services_subscribed
    df["num_services_subscribed"] = (df[_SERVICE_COLS] == "Yes").sum(axis=1)

    # has_internet_and_phone
    df["has_internet_and_phone"] = (
        (df["PhoneService"] == "Yes") & (df["InternetService"] != "No")
    ).astype(int)

    # avg_monthly_spend
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # contract_risk_score
    df["contract_risk_score"] = df["Contract"].map(_CONTRACT_RISK_MAP)

    return df


def to_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run feature engineering and return columns in the exact order the saved
    model pipeline's ColumnTransformer expects.
    """
    engineered = engineer_features(df)
    return engineered[MODEL_FEATURE_ORDER]
