"""
backend/predictor.py
---------------------
Loads the trained model pipeline ONCE at import time (which happens once, at
FastAPI startup) and exposes a single `predict()` function used by the /predict
endpoint. The model is never reloaded from disk per-request -- this is what
"the model must be loaded once when FastAPI starts" means in practice.

As of Phase 14, WHICH model file gets loaded is decided by
backend/versioning.py's registry (models/version_registry.json), not a
hardcoded path -- see that module's docstring. `reload()` below is the one
deliberate exception to "loaded once, never reloaded": it's called only by
an explicit admin action (POST /promote-candidate or POST /rollback in
backend/main.py), never from the request path, so this still doesn't
violate the "not reloaded per request" rule -- a promotion/rollback is a
rare, human-triggered event, not part of serving a prediction.

Explainability here uses the model's own coefficients (Logistic Regression) or
feature importances (tree-based models) rather than SHAP, per the project spec's
guidance: "If SHAP causes unnecessary complexity, use model feature importance or
another explainability approach appropriate to the selected model." For a linear
model, coefficients x feature value is an exact, cheap, and honest explanation of
that specific prediction (a per-instance local explanation), not just a global
importance ranking -- so it's appropriate here, not a downgrade.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from ml.preprocess import to_model_input, NUMERIC_FEATURES
from backend.versioning import get_active_model_paths

# Human-readable labels for the top contributing factors shown to a
# non-technical customer-service representative.
_FEATURE_LABELS = {
    "contract_risk_score": "Contract type",
    "tenure": "How long they've been a customer",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges to date",
    "avg_monthly_spend": "Average monthly spend",
    "num_services_subscribed": "Number of services subscribed",
    "SeniorCitizen": "Senior citizen status",
    "is_new_customer": "New customer (under 6 months)",
    "has_internet_and_phone": "Bundled phone + internet",
}


class ModelNotLoadedError(RuntimeError):
    pass


class ChurnPredictor:
    """Wraps the saved pipeline + metadata. Instantiated once at module import."""

    def __init__(self):
        self._pipeline = None
        self._metadata: dict = {}
        self._load_error: str | None = None
        self.reload()

    def reload(self) -> None:
        """
        Re-reads the active version from backend/versioning.py's registry
        and (re)loads that model + metadata. Called once at import time
        (via __init__), and again -- deliberately -- after an admin
        promotes a candidate or rolls back. See this class's docstring for
        why that doesn't violate the "load once, don't reload per request"
        rule.
        """
        try:
            model_path, metadata_path = get_active_model_paths()
            self._pipeline = joblib.load(model_path)
            with open(metadata_path) as f:
                self._metadata = json.load(f)
            self._load_error = None
        except Exception as e:  # noqa: BLE001 - we want to capture and report any load failure
            self._load_error = str(e)
            self._pipeline = None
            self._metadata = {}

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def metadata(self) -> dict:
        return self._metadata

    def _risk_category(self, probability: float) -> str:
        if probability >= 0.6:
            return "HIGH"
        if probability >= 0.3:
            return "MEDIUM"
        return "LOW"

    def _top_factors(self, raw_df: pd.DataFrame, model_input: pd.DataFrame, top_n: int = 4) -> list[dict]:
        """
        Produce a small, human-readable explanation for this single prediction.

        For the numeric block we use standardized-coefficient x scaled-value as the
        per-feature contribution (a local, additive explanation for linear models).
        This intentionally stays at the numeric-feature level rather than exploding
        every one-hot category, because a customer-service rep needs "contract type
        matters a lot here", not twelve near-zero one-hot coefficients.
        """
        classifier = self._pipeline.named_steps.get("classifier")
        preprocessor = self._pipeline.named_steps.get("preprocessor")

        if classifier is None or not hasattr(classifier, "coef_"):
            # Non-linear model selected (e.g. Random Forest/Gradient Boosting):
            # fall back to global feature_importances_ as a coarser explanation.
            if hasattr(classifier, "feature_importances_"):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    importances = classifier.feature_importances_
                    order = np.argsort(importances)[::-1][:top_n]
                    return [
                        {
                            "feature": feature_names[i],
                            "contribution": float(importances[i]),
                            "direction": "increases_risk",
                            "human_readable": f"{feature_names[i]} is one of the model's most influential factors overall",
                        }
                        for i in order
                    ]
                except Exception:
                    return []
            return []

        # Linear model path: use standardized numeric coefficients x scaled value.
        scaler = preprocessor.named_transformers_["num"]
        scaled_values = scaler.transform(raw_df[NUMERIC_FEATURES])[0]
        coefs = classifier.coef_[0][: len(NUMERIC_FEATURES)]
        contributions = coefs * scaled_values

        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        factors = []
        for i in order:
            feat_name = NUMERIC_FEATURES[i]
            contrib = float(contributions[i])
            factors.append({
                "feature": feat_name,
                "contribution": round(contrib, 4),
                "direction": "increases_risk" if contrib > 0 else "decreases_risk",
                "human_readable": self._describe_factor(feat_name, contrib, raw_df),
            })
        return factors

    def _describe_factor(self, feat_name: str, contribution: float, raw_df: pd.DataFrame) -> str:
        label = _FEATURE_LABELS.get(feat_name, feat_name)
        direction = "increases" if contribution > 0 else "reduces"
        return f"{label} {direction} this customer's churn risk"

    def predict(self, customer_dict: dict) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError(self._load_error or "Model not loaded")

        start = time.perf_counter()

        raw_df = pd.DataFrame([customer_dict])
        model_input = to_model_input(raw_df)

        probability = float(self._pipeline.predict_proba(model_input)[0, 1])
        prediction = "CHURN" if probability >= 0.5 else "NO_CHURN"
        risk = self._risk_category(probability)
        top_factors = self._top_factors(model_input, model_input)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk": risk,
            "model_version": self._metadata.get("version", "unknown"),
            "latency_ms": round(latency_ms, 2),
            "top_factors": top_factors,
        }


# Instantiated once, at import time -- i.e. once when the FastAPI app starts.
# Which file it actually loads is decided by backend/versioning.py's
# registry, not hardcoded here.
predictor = ChurnPredictor()
