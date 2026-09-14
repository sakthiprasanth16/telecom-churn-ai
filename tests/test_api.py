"""
tests/test_api.py
------------------
Automated tests for the FastAPI backend. Run with:

    pytest tests/test_api.py -v

Requires a trained model at models/model_v1.pkl (produced by
notebooks/03_model_training.ipynb) to already exist -- these are integration
tests against the real loaded pipeline, not mocked unit tests, because the
thing most likely to break in this project is the interaction between
feature engineering, the saved pipeline's expected columns, and Pydantic
validation -- and a mock would hide exactly that.
"""

from fastapi.testclient import TestClient
import os
import pytest

from backend.main import app
from backend.predictor import predictor

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 8,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 684.0,
}

LOW_RISK_PAYLOAD = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 65, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 55.0,
    "TotalCharges": 3500.0,
}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_reports_model_loaded(self):
        r = client.get("/health")
        body = r.json()
        assert body["model_loaded"] is True
        assert body["status"] == "ok"


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self):
        r = client.get("/model-info")
        assert r.status_code == 200

    def test_model_info_has_expected_fields(self):
        body = client.get("/model-info").json()
        for field in ["version", "model_type", "roc_auc", "precision", "recall", "f1"]:
            assert field in body


class TestPredictEndpointValid:
    def test_predict_returns_200(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_predict_response_shape(self):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["prediction"] in ("CHURN", "NO_CHURN")
        assert 0.0 <= body["probability"] <= 1.0
        assert body["risk"] in ("LOW", "MEDIUM", "HIGH")
        assert "model_version" in body
        assert "latency_ms" in body
        assert isinstance(body["top_factors"], list)

    def test_predict_latency_is_fast(self):
        # Sub-100ms target is for the model inference itself; TestClient adds
        # its own overhead, so we assert generously here (the reported
        # latency_ms field is what should be watched in real deployment).
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["latency_ms"] < 500

    def test_high_risk_profile_predicts_higher_probability_than_low_risk(self):
        high = client.post("/predict", json=VALID_PAYLOAD).json()
        low = client.post("/predict", json=LOW_RISK_PAYLOAD).json()
        assert high["probability"] > low["probability"]
        assert low["prediction"] == "NO_CHURN"
        assert low["risk"] == "LOW"

    def test_top_factors_not_empty_for_linear_model(self):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert len(body["top_factors"]) > 0
        for factor in body["top_factors"]:
            assert factor["direction"] in ("increases_risk", "decreases_risk")


class TestDriftEndpoint:
    """
    Uses the same shared TestClient/mongomock instance as the rest of this
    file, so /predict calls made anywhere above in the test session also
    seed the `predictions` collection /drift reads from. That's why this
    class first fires enough /predict calls itself to guarantee at least
    MIN_SAMPLE_SIZE (30) rows exist, rather than depending on test order.
    """

    def _seed_predictions(self, n: int = 40) -> None:
        for _ in range(n):
            client.post("/predict", json=VALID_PAYLOAD)

    def test_drift_returns_200(self):
        self._seed_predictions()
        r = client.get("/drift")
        assert r.status_code == 200

    def test_drift_response_shape(self):
        self._seed_predictions()
        body = client.get("/drift").json()
        assert body["status"] in ("no_drift", "moderate_drift", "significant_drift", "insufficient_data")
        assert "n_current_rows" in body
        assert isinstance(body["drifted_features"], list)
        assert isinstance(body["numeric_features"], dict)
        assert isinstance(body["categorical_features"], dict)
        for feature_result in {**body["numeric_features"], **body["categorical_features"]}.values():
            assert "psi" in feature_result
            assert feature_result["status"] in ("no_drift", "moderate_drift", "significant_drift")

    def test_drift_with_identical_repeated_payload_reports_low_or_no_drift(self):
        # Sending the exact same customer profile repeatedly is an extreme
        # (zero-variance) case, not representative of real traffic, but it
        # should never crash and should always return a valid status.
        self._seed_predictions(n=50)
        body = client.get("/drift").json()
        assert body["status"] in ("no_drift", "moderate_drift", "significant_drift")

    def test_drift_limit_below_min_sample_size_rejected(self):
        r = client.get("/drift", params={"limit": 5})
        assert r.status_code == 422

    def test_drift_limit_above_max_rejected(self):
        r = client.get("/drift", params={"limit": 10000})
        assert r.status_code == 422

    def test_drift_logs_report_to_mongo(self):
        # log_drift_report() is fired via BackgroundTasks; TestClient runs
        # background tasks synchronously before returning, so it should
        # already be persisted by the time this request completes.
        self._seed_predictions()
        client.get("/drift")
        from backend.database import database
        assert database.is_connected
        reports = list(database._db["drift_reports"].find({}))
        assert len(reports) > 0


class TestRetrainEndpoint:
    """
    Requires data/processed/churn_cleaned_v1.csv to exist (same file
    notebooks 01-03 already produced) -- see ml/retraining_graph.py's
    module docstring for why this graph deliberately retrains on that same
    base dataset rather than fabricating new labeled data.
    """

    def test_retrain_forced_returns_200_and_trains_a_candidate(self):
        r = client.post("/retrain", params={"force": True})
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] in ("candidate_approved", "candidate_rejected")
        assert body["candidate_metrics"] is not None
        assert body["current_metrics"] is not None
        for metric in ("precision", "recall", "f1", "roc_auc"):
            assert metric in body["candidate_metrics"]

    def test_retrain_without_force_reports_a_valid_decision(self):
        r = client.post("/retrain", params={"force": False})
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] in ("skipped_no_drift", "candidate_approved", "candidate_rejected", "error")
        assert body["drift_report"] is not None

    def test_retrain_never_touches_the_production_model_file(self):
        # A candidate must never overwrite backend/predictor.py's loaded
        # model -- promotion is Phase 14's job, not this workflow's.
        before = predictor.metadata.get("version")
        client.post("/retrain", params={"force": True})
        assert predictor.metadata.get("version") == before

    def test_retrain_limit_below_min_sample_size_rejected(self):
        r = client.post("/retrain", params={"limit": 5})
        assert r.status_code == 422

    def test_retrain_limit_above_max_rejected(self):
        r = client.post("/retrain", params={"limit": 10000})
        assert r.status_code == 422

    def test_retrain_logs_report_to_mongo(self):
        client.post("/retrain", params={"force": True})
        from backend.database import database
        assert database.is_connected
        reports = list(database._db["retraining_runs"].find({}))
        assert len(reports) > 0


class TestVersioningEndpoints:
    """
    Snapshots backend/versioning.py's registry file (and the set of files
    in models/) before each test and restores both afterward, so running
    this suite repeatedly doesn't leave an ever-growing pile of promoted
    versions on disk -- each test starts and ends from the same registry
    state. Requires data/processed/churn_cleaned_v1.csv to exist (same
    dependency as TestRetrainEndpoint), since some tests trigger a real
    /retrain to produce a candidate to promote.
    """

    @pytest.fixture(autouse=True)
    def _snapshot_and_restore_registry(self):
        from backend import versioning

        registry_before = versioning.load_registry()
        files_before = set(os.listdir(versioning.MODELS_DIR))

        yield

        versioning._write_registry(registry_before)
        for fname in os.listdir(versioning.MODELS_DIR):
            if fname not in files_before:
                os.remove(versioning.MODELS_DIR / fname)
        predictor.reload()

    def _ensure_candidate_exists(self):
        client.post("/retrain", params={"force": True})

    def test_model_versions_returns_200_and_includes_v1(self):
        r = client.get("/model-versions")
        assert r.status_code == 200
        body = r.json()
        assert "v1" in body["versions"]
        assert body["active_version"] in body["versions"]

    def test_promote_candidate_creates_new_version_and_updates_active(self):
        before = client.get("/model-versions").json()["active_version"]
        self._ensure_candidate_exists()

        r = client.post("/promote-candidate")
        assert r.status_code == 200
        body = r.json()
        assert body["previous_version"] == before
        assert body["active_version"] != before

        # The new version should now actually be what's being served.
        assert client.get("/model-info").json()["version"] == body["active_version"]

    def test_promote_candidate_without_a_candidate_returns_409(self):
        from backend import versioning
        for p in (versioning.CANDIDATE_MODEL_PATH, versioning.CANDIDATE_METADATA_PATH):
            if p.exists():
                p.unlink()

        r = client.post("/promote-candidate")
        assert r.status_code == 409

    def test_rollback_reverts_to_previous_version(self):
        before = client.get("/model-versions").json()["active_version"]
        self._ensure_candidate_exists()
        client.post("/promote-candidate")

        r = client.post("/rollback")
        assert r.status_code == 200
        assert r.json()["active_version"] == before
        assert client.get("/model-info").json()["version"] == before

    def test_rollback_to_unknown_version_returns_409(self):
        r = client.post("/rollback", params={"to_version": "v999"})
        assert r.status_code == 409

    def test_rollback_to_already_active_version_returns_409(self):
        active = client.get("/model-versions").json()["active_version"]
        r = client.post("/rollback", params={"to_version": active})
        assert r.status_code == 409

    def test_predictions_still_work_after_a_full_promote_rollback_cycle(self):
        self._ensure_candidate_exists()
        client.post("/promote-candidate")
        client.post("/rollback")

        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200
        assert r.json()["prediction"] in ("CHURN", "NO_CHURN")


class TestFairnessEndpoint:
    """
    Requires data/processed/churn_cleaned_v1.csv to exist (same dependency
    as TestRetrainEndpoint) for the training_report path to have real
    content after a retrain+promote cycle. Uses the same registry/model-
    file snapshot-and-restore fixture as TestVersioningEndpoints so
    promoting a candidate here doesn't leave files behind between runs.
    """

    @pytest.fixture(autouse=True)
    def _snapshot_and_restore_registry(self):
        from backend import versioning

        registry_before = versioning.load_registry()
        files_before = set(os.listdir(versioning.MODELS_DIR))

        yield

        versioning._write_registry(registry_before)
        for fname in os.listdir(versioning.MODELS_DIR):
            if fname not in files_before:
                os.remove(versioning.MODELS_DIR / fname)
        predictor.reload()

    def test_fairness_returns_200(self):
        r = client.get("/fairness")
        assert r.status_code == 200

    def test_fairness_response_shape(self):
        body = client.get("/fairness").json()
        assert "n_current_rows" in body
        if body["training_report"] is None:
            assert body.get("training_report_note") is not None
        for report in (body.get("live_snapshot") or {}).values():
            assert "selection_rate_ratio" in report
            assert "flagged" in report

    def test_fairness_gets_a_training_report_after_promoting_a_retrained_candidate(self):
        client.post("/retrain", params={"force": True})
        client.post("/promote-candidate")

        body = client.get("/fairness").json()
        assert body["training_report"] is not None
        assert "gender" in body["training_report"] or "SeniorCitizen" in body["training_report"]
        for attr_report in body["training_report"].values():
            assert "disparities" in attr_report
            assert "flagged" in attr_report

    def test_fairness_limit_validation(self):
        r = client.get("/fairness", params={"limit": 5})
        assert r.status_code == 422


class TestBusinessImpactEndpoint:
    """
    Uses the same registry/model-file snapshot-and-restore fixture as
    TestVersioningEndpoints and TestFairnessEndpoint, since some tests here
    promote a retrained candidate to get a fresh business_impact_inputs.
    """

    @pytest.fixture(autouse=True)
    def _snapshot_and_restore_registry(self):
        from backend import versioning

        registry_before = versioning.load_registry()
        files_before = set(os.listdir(versioning.MODELS_DIR))

        yield

        versioning._write_registry(registry_before)
        for fname in os.listdir(versioning.MODELS_DIR):
            if fname not in files_before:
                os.remove(versioning.MODELS_DIR / fname)
        predictor.reload()

    def test_business_impact_returns_200(self):
        r = client.get("/business-impact")
        assert r.status_code == 200

    def test_business_impact_response_shape(self):
        body = client.get("/business-impact").json()
        if body["training_report"] is not None:
            for name in ("model_targeted", "contact_everyone", "do_nothing"):
                assert name in body["training_report"]["scenarios"]
                scenario = body["training_report"]["scenarios"][name]
                assert "estimated_net_benefit" in scenario
                assert "roi" in scenario
        else:
            assert body["training_report_note"] is not None

    def test_business_impact_after_retrain_and_promote_has_real_data(self):
        client.post("/retrain", params={"force": True})
        client.post("/promote-candidate")

        body = client.get("/business-impact").json()
        assert body["training_report"] is not None
        report = body["training_report"]

        cm = report["confusion_matrix"]
        assert cm["true_positive"] + cm["false_positive"] + cm["false_negative"] + cm["true_negative"] > 0

        model_targeted = report["scenarios"]["model_targeted"]
        contact_everyone = report["scenarios"]["contact_everyone"]
        # do_nothing must always be all zeros regardless of the data.
        assert report["scenarios"]["do_nothing"]["estimated_net_benefit"] == 0
        # Model-targeted should contact no more customers than a blanket campaign.
        assert model_targeted["customers_flagged"] <= contact_everyone["customers_flagged"]

    def test_business_impact_custom_assumptions_change_the_numbers(self):
        client.post("/retrain", params={"force": True})
        client.post("/promote-candidate")

        default_body = client.get("/business-impact").json()
        custom_body = client.get(
            "/business-impact",
            params={"retention_success_rate": 0.9, "intervention_cost_per_contact": 1.0},
        ).json()

        default_benefit = default_body["training_report"]["scenarios"]["model_targeted"]["estimated_net_benefit"]
        custom_benefit = custom_body["training_report"]["scenarios"]["model_targeted"]["estimated_net_benefit"]
        # A much higher retention success rate and much lower cost should
        # never produce a WORSE (lower) net benefit.
        assert custom_benefit >= default_benefit

    def test_business_impact_scale_to_population(self):
        client.post("/retrain", params={"force": True})
        client.post("/promote-candidate")

        body = client.get("/business-impact", params={"scale_to_population_param": 10_000_000}).json()
        assert body["scaled_report"] is not None
        assert body["scaled_report"]["target_population"] == 10_000_000
        # ROI is a ratio -- scaling shouldn't change it.
        for name in body["training_report"]["scenarios"]:
            assert (
                body["scaled_report"]["scenarios"][name]["roi"]
                == body["training_report"]["scenarios"][name]["roi"]
            )


class TestMongoDownGracefulDegradation:
    """
    Directly simulates MongoDB being unreachable by clearing database._db
    (rather than actually breaking a real connection), so every endpoint's
    existing graceful-degradation code path (see database.py's docstring:
    "a logging failure must never break the customer-facing prediction
    flow") gets exercised for real, not just trusted by inspection.
    """

    @pytest.fixture(autouse=True)
    def _simulate_mongo_down(self):
        from backend.database import database

        original_db = database._db
        database._db = None
        yield
        database._db = original_db

    def test_predict_still_returns_200_when_mongo_down(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_health_reports_mongo_disconnected(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["mongo_connected"] is False

    def test_metrics_returns_empty_defaults_when_mongo_down(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.json()["total_predictions"] == 0

    def test_predictions_endpoint_returns_empty_list_when_mongo_down(self):
        r = client.get("/predictions")
        assert r.status_code == 200
        assert r.json() == []

    def test_drift_reports_insufficient_data_when_mongo_down(self):
        r = client.get("/drift")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "insufficient_data"
        assert body["n_current_rows"] == 0

    def test_fairness_reports_no_live_data_when_mongo_down(self):
        r = client.get("/fairness")
        assert r.status_code == 200
        assert r.json()["n_current_rows"] == 0

    def test_business_impact_still_returns_200_when_mongo_down(self):
        r = client.get("/business-impact")
        assert r.status_code == 200

    def test_model_versions_and_predictor_unaffected_by_mongo_down(self):
        # Versioning is file-based, not Mongo-based -- must be completely
        # unaffected (see backend/versioning.py's docstring).
        r = client.get("/model-versions")
        assert r.status_code == 200
        assert client.get("/health").json()["model_loaded"] is True


class TestPredictBoundaryValues:
    """Boundary-value analysis beyond the existing invalid-input tests --
    confirms values AT the documented limits are accepted, not just that
    values beyond them are rejected."""

    def test_max_boundary_values_accepted(self):
        payload = dict(VALID_PAYLOAD, tenure=100, MonthlyCharges=1000.0, TotalCharges=100000.0)
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_min_tenure_zero_with_zero_total_charges_accepted(self):
        payload = dict(VALID_PAYLOAD, tenure=0, TotalCharges=0.0)
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_minimum_nonzero_monthly_charges_accepted(self):
        # MonthlyCharges must be > 0 (not >= 0) per schemas.py -- confirm
        # a tiny-but-valid value is accepted, not just that 0 is rejected.
        payload = dict(VALID_PAYLOAD, MonthlyCharges=0.01)
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_extra_unexpected_field_is_ignored_not_rejected(self):
        # Pydantic v2 ignores unrecognized fields by default (CustomerInput
        # doesn't set extra="forbid") -- confirms that's the actual, current
        # behavior rather than an assumption.
        payload = dict(VALID_PAYLOAD, this_field_does_not_exist="whatever")
        r = client.post("/predict", json=payload)
        assert r.status_code == 200


class TestVersioningConcurrency:
    """
    Fires several concurrent promote_candidate() calls from real threads
    (not just sequential TestClient calls) to verify backend/versioning.py's
    lock + atomic-write fix actually prevents a lost update within a
    single process. See that module's docstring for what this does and
    does not protect against (multiple Uvicorn worker processes are out
    of scope for this test).
    """

    @pytest.fixture(autouse=True)
    def _snapshot_and_restore_registry(self):
        from backend import versioning

        registry_before = versioning.load_registry()
        files_before = set(os.listdir(versioning.MODELS_DIR))

        yield

        versioning._write_registry(registry_before)
        for fname in os.listdir(versioning.MODELS_DIR):
            if fname not in files_before:
                os.remove(versioning.MODELS_DIR / fname)
        predictor.reload()

    def test_concurrent_promotions_all_succeed_with_unique_versions(self):
        from concurrent.futures import ThreadPoolExecutor
        from backend import versioning

        client.post("/retrain", params={"force": True})

        n_threads = 8
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(versioning.promote_candidate) for _ in range(n_threads)]
            results = [f.result() for f in futures]

        versions = [r["active_version"] for r in results]
        assert len(set(versions)) == n_threads, "every concurrent promotion should get a unique version"

        registry = versioning.load_registry()
        assert registry["active_version"] in registry["versions"]
        for v in registry["history"]:
            assert v in registry["versions"], f"history references unknown version {v}"


class TestLatencyUnderVolume:
    """
    A lightweight sanity check, not a real load-test benchmark -- TestClient
    is synchronous and in-process, so it can't model true concurrent-
    connection load. See tests/load_test.py for a threaded, wall-clock
    latency/throughput report meant to be run separately, on demand.
    """

    def test_latency_stays_reasonable_across_repeated_requests(self):
        latencies = []
        for _ in range(30):
            r = client.post("/predict", json=VALID_PAYLOAD)
            assert r.status_code == 200
            latencies.append(r.json()["latency_ms"])

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        # Generous thresholds, same reasoning as test_predict_latency_is_fast:
        # the reported latency_ms is inference-only and should stay fast even
        # across repeated calls, but this isn't a strict production SLA check.
        assert p50 < 200, f"p50 latency {p50}ms unexpectedly high"
        assert p95 < 500, f"p95 latency {p95}ms unexpectedly high"


class TestPredictEndpointInvalidInput:
    def test_negative_tenure_rejected(self):
        bad = dict(VALID_PAYLOAD, tenure=-5)
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_tenure_zero_with_nonzero_total_charges_rejected(self):
        bad = dict(VALID_PAYLOAD, tenure=0, TotalCharges=500.0)
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_missing_required_field_rejected(self):
        bad = dict(VALID_PAYLOAD)
        del bad["Contract"]
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_wrong_type_rejected(self):
        bad = dict(VALID_PAYLOAD, tenure="eight")
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_categorical_value_rejected(self):
        bad = dict(VALID_PAYLOAD, Contract="Lifetime")
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_monthly_charges_zero_rejected(self):
        bad = dict(VALID_PAYLOAD, MonthlyCharges=0)
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_out_of_range_tenure_rejected(self):
        bad = dict(VALID_PAYLOAD, tenure=500)
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
