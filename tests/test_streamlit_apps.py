"""
tests/test_streamlit_apps.py
-----------------------------
Streamlit AppTest suite (project spec Phase 17: "additional testing" --
covers the frontend, not just the FastAPI backend that tests/test_api.py
already exercises). Run with:

    pytest tests/test_streamlit_apps.py -v

AppTest (streamlit.testing.v1) runs the actual app script headlessly --
no browser, no running Streamlit server -- and lets us inspect rendered
elements and simulate widget interaction directly. See
https://docs.streamlit.io/develop/concepts/app-testing for the framework
this is built on.

Both frontend/app.py and frontend/pages/1_Monitoring.py talk to the
FastAPI backend over real HTTP (`requests.get`/`requests.post`) rather
than importing backend code directly -- see app.py's own docstring for
why that separation matters. These tests mock `requests.get`/`requests.post`
directly rather than running a live FastAPI server, so they test the
Streamlit rendering logic in isolation: given a specific backend response,
does the page show the right thing? A real running backend is exercised
separately by tests/test_api.py and by manual end-to-end testing (Phase 20).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves a relative path against the file that CALLS
# it (this test file's own location), NOT the current working directory --
# so plain "frontend/app.py" would incorrectly resolve to
# tests/frontend/app.py. Using absolute paths computed from this file's
# own location sidesteps that entirely and works regardless of where
# pytest is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(_PROJECT_ROOT / "frontend" / "app.py")
MONITORING_PATH = str(_PROJECT_ROOT / "frontend" / "pages" / "1_Monitoring.py")


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    return mock


HEALTHY = {"status": "ok", "model_loaded": True, "model_version": "v1", "mongo_connected": True}

VALID_PREDICTION_RESPONSE = {
    "prediction": "CHURN",
    "probability": 0.72,
    "risk": "HIGH",
    "model_version": "v1",
    "latency_ms": 12.3,
    "top_factors": [
        {
            "feature": "contract_risk_score",
            "contribution": 0.51,
            "direction": "increases_risk",
            "human_readable": "Contract type increases this customer's churn risk",
        }
    ],
}


class TestPredictionApp:
    def test_app_loads_without_exception_when_backend_healthy(self):
        with patch("requests.get", return_value=_mock_response(HEALTHY)):
            at = AppTest.from_file(APP_PATH, default_timeout=15)
            at.run()
        assert not at.exception

    def test_app_shows_error_when_backend_unreachable(self):
        import requests

        with patch("requests.get", side_effect=requests.exceptions.RequestException("no connection")):
            at = AppTest.from_file(APP_PATH, default_timeout=15)
            at.run()
        # app.py catches this itself (st.error + st.stop()) -- it must never
        # surface as an uncaught AppTest exception.
        assert not at.exception
        assert len(at.error) > 0

    def test_app_shows_error_when_model_not_loaded(self):
        degraded = {"status": "degraded", "model_loaded": False, "mongo_connected": True}
        with patch("requests.get", return_value=_mock_response(degraded)):
            at = AppTest.from_file(APP_PATH, default_timeout=15)
            at.run()
        assert len(at.error) > 0

    def test_form_submission_with_default_values_shows_prediction_result(self):
        with patch("requests.get", return_value=_mock_response(HEALTHY)), patch(
            "requests.post", return_value=_mock_response(VALID_PREDICTION_RESPONSE)
        ):
            at = AppTest.from_file(APP_PATH, default_timeout=15)
            at.run()
            assert not at.exception
            # Submits the form with whatever default widget values app.py sets.
            at.button[0].click().run()

        assert not at.exception
        assert any("CHURN" in str(m.value) for m in at.metric)

    def test_form_submission_shows_422_validation_errors_from_backend(self):
        validation_error_response = _mock_response(
            {"detail": [{"loc": ["body", "tenure"], "msg": "value is not a valid integer"}]},
            status_code=422,
        )
        with patch("requests.get", return_value=_mock_response(HEALTHY)), patch(
            "requests.post", return_value=validation_error_response
        ):
            at = AppTest.from_file(APP_PATH, default_timeout=15)
            at.run()
            at.button[0].click().run()

        assert not at.exception
        assert len(at.error) > 0


class TestMonitoringPage:
    @staticmethod
    def _default_get_side_effect(url: str, **kwargs):
        if url.endswith("/health"):
            return _mock_response(HEALTHY)
        if url.endswith("/metrics"):
            return _mock_response(
                {
                    "total_predictions": 0,
                    "predictions_today": 0,
                    "avg_latency_ms": 0.0,
                    "churn_count": 0,
                    "no_churn_count": 0,
                    "risk_breakdown": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
                }
            )
        if url.endswith("/model-info"):
            return _mock_response(
                {
                    "version": "v1",
                    "model_type": "LogisticRegression",
                    "dataset_version": "dataset_v1",
                    "feature_version": "features_v1",
                    "trained_at": "2026-09-11T03:22:50",
                    "precision": 0.5076,
                    "recall": 0.8048,
                    "f1": 0.6225,
                    "roc_auc": 0.8447,
                    "status": "production",
                }
            )
        if url.endswith("/drift"):
            return _mock_response(
                {
                    "status": "insufficient_data",
                    "n_reference_rows": 7043,
                    "n_current_rows": 0,
                    "message": "Only 0 recent predictions available; need at least 30.",
                    "numeric_features": {},
                    "categorical_features": {},
                    "drifted_features": [],
                }
            )
        if url.endswith("/fairness"):
            return _mock_response(
                {
                    "training_report": None,
                    "training_report_note": "The active model's metadata has no fairness_report.",
                    "live_snapshot": None,
                    "live_snapshot_note": "Only 0 recent predictions available; need at least 30.",
                    "n_current_rows": 0,
                }
            )
        if url.endswith("/business-impact"):
            return _mock_response(
                {
                    "training_report": None,
                    "training_report_note": "The active model's metadata has no business_impact_inputs.",
                    "scaled_report": None,
                    "live_projection": None,
                    "live_projection_note": "Unavailable -- same reason as training_report_note above.",
                }
            )
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    def test_monitoring_page_loads_without_exception(self):
        with patch("requests.get", side_effect=self._default_get_side_effect):
            at = AppTest.from_file(MONITORING_PATH, default_timeout=15)
            at.run()
        assert not at.exception

    def test_monitoring_page_shows_system_health(self):
        with patch("requests.get", side_effect=self._default_get_side_effect):
            at = AppTest.from_file(MONITORING_PATH, default_timeout=15)
            at.run()
        assert not at.exception
        assert any("Online" in str(m.value) for m in at.metric)

    def test_monitoring_page_shows_insufficient_data_messages_gracefully(self):
        # With zero live predictions, drift/fairness/business-impact should
        # all render an informational note, not crash or show fabricated
        # numbers -- this is the same honesty requirement enforced in
        # ml/drift.py, ml/fairness.py, and ml/business_impact.py themselves.
        with patch("requests.get", side_effect=self._default_get_side_effect):
            at = AppTest.from_file(MONITORING_PATH, default_timeout=15)
            at.run()
        assert not at.exception
        assert len(at.info) > 0

    def test_monitoring_page_handles_unreachable_backend(self):
        import requests

        with patch("requests.get", side_effect=requests.exceptions.RequestException("no connection")):
            at = AppTest.from_file(MONITORING_PATH, default_timeout=15)
            at.run()
        assert not at.exception
        assert len(at.error) > 0
