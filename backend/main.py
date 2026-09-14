"""
backend/main.py
----------------
FastAPI application.

Architecture (per project spec):
    User -> Streamlit -> FastAPI -> Pydantic validation -> Preprocessing
    -> Already-loaded ML model -> Prediction -> Explanation -> (MongoDB logging,
    added in Phase 8) -> Response to Streamlit

The model is loaded once, at import time, via `backend/predictor.py`'s module-level
`predictor` instance -- NOT reloaded on every request. FastAPI's startup event just
verifies that load succeeded and fails loudly if it didn't, rather than silently
serving broken predictions.

MongoDB logging (recording each prediction) is deliberately NOT wired in yet --
that's Phase 8 in the build order. This phase focuses on making /predict itself
correct, validated, and fast.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # Loads MONGODB_URI etc. from a local .env file, if present.
               # Must happen BEFORE backend.database is imported below, since
               # that module reads os.environ at import time.

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    CustomerInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRecord,
    MetricsResponse,
    DriftResponse,
    RetrainingReport,
    VersionRegistryResponse,
    VersionActionResponse,
    FairnessResponse,
    BusinessImpactResponse,
)
from backend.predictor import predictor, ModelNotLoadedError
from backend.database import database
from backend.versioning import (
    load_registry,
    promote_candidate as versioning_promote_candidate,
    rollback as versioning_rollback,
    VersioningError,
)
from ml.drift import load_reference_stats, detect_drift, MIN_SAMPLE_SIZE
from ml.retraining_graph import run_retraining_workflow
from ml.fairness import audit_all_attributes_selection_rate_only
from ml.business_impact import simulate_business_impact, scale_to_population, DEFAULT_ASSUMPTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

# Drift reference stats (Phase 12) are loaded ONCE at import time, same
# pattern as the model in predictor.py -- never reloaded per request. This
# is a static baseline snapshot of the training distribution; per the
# handoff notes it should only be regenerated when a new model is trained
# on a new dataset version, not reloaded casually.
DRIFT_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "models" / "drift_reference_v1.json"

try:
    _drift_reference_stats = load_reference_stats(DRIFT_REFERENCE_PATH)
    _drift_reference_load_error: str | None = None
except Exception as e:  # noqa: BLE001
    _drift_reference_stats = None
    _drift_reference_load_error = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: the model was already loaded once, at import time, by
    # backend/predictor.py's module-level `predictor` instance. We just
    # verify and log that here -- we deliberately do NOT reload it per
    # request, and we do NOT crash the app if loading failed, so /health
    # can still report "degraded" for monitoring instead of the process
    # refusing to start with no diagnostic surface.
    if predictor.is_loaded:
        logger.info(
            "Model loaded successfully: version=%s type=%s",
            predictor.metadata.get("version"),
            predictor.metadata.get("model_type"),
        )
    else:
        logger.error("Model failed to load: %s", predictor.load_error)

    if database.is_connected:
        logger.info("MongoDB connected successfully.")
    else:
        logger.warning(
            "MongoDB NOT connected (%s) -- predictions will still be served, "
            "but will not be logged.",
            database.connect_error,
        )

    if _drift_reference_stats is not None:
        logger.info(
            "Drift reference stats loaded: n_reference_rows=%s",
            _drift_reference_stats.get("n_reference_rows"),
        )
    else:
        logger.error("Drift reference stats failed to load: %s", _drift_reference_load_error)

    yield
    # Shutdown: nothing to clean up -- the model holds no open resources.


app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description=(
        "Real-time churn prediction for customer-service representatives. "
        "The model is loaded once at startup and reused for every request."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Streamlit will call this API from a different host/port (and later a different
# Render service entirely) -- CORS must be open enough for that, but this should
# be tightened to the actual deployed Streamlit URL once that's known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version=predictor.metadata.get("version") if predictor.is_loaded else None,
        mongo_connected=database.is_connected,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not currently loaded")
    m = predictor.metadata
    return ModelInfoResponse(
        version=m.get("version", "unknown"),
        model_type=m.get("model_type", "unknown"),
        dataset_version=m.get("dataset_version", "unknown"),
        feature_version=m.get("feature_version", "unknown"),
        trained_at=m.get("trained_at", "unknown"),
        precision=m.get("precision", 0.0),
        recall=m.get("recall", 0.0),
        f1=m.get("f1", 0.0),
        roc_auc=m.get("roc_auc", 0.0),
        status=m.get("status", "unknown"),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    m = database.get_monitoring_metrics()
    return MetricsResponse(**m)


@app.get("/drift", response_model=DriftResponse)
async def drift(background_tasks: BackgroundTasks, limit: int = 500) -> DriftResponse:
    """
    Compares recent live prediction inputs (pulled from MongoDB's
    `predictions` collection) against the training-time reference
    distribution baked into models/drift_reference_v1.json, using
    ml.drift.detect_drift() (PSI-based -- see that module's docstring for
    the full methodology and threshold rationale).

    `limit` controls how many recent predictions are pulled for the
    comparison; the default of 500 balances "recent enough to reflect
    current behavior" against "enough rows for PSI to be meaningful" --
    ml.drift.MIN_SAMPLE_SIZE (30) is the hard floor below which the report
    is 'insufficient_data' rather than a false 'no drift'.

    Compares raw input fields only (not engineered features) -- see
    ml/drift.py's module docstring for why that's sufficient.
    """
    if _drift_reference_stats is None:
        raise HTTPException(
            status_code=503,
            detail=f"Drift reference stats are not loaded: {_drift_reference_load_error}",
        )
    if limit < MIN_SAMPLE_SIZE or limit > 5000:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be between {MIN_SAMPLE_SIZE} and 5000",
        )

    recent_inputs = database.get_recent_prediction_inputs(limit=limit)
    current_df = pd.DataFrame(recent_inputs)
    report = detect_drift(_drift_reference_stats, current_df)

    # Logged in the background, after the response is built, so a slow or
    # unavailable database never delays this monitoring call either -- same
    # reasoning as /predict's background logging (see database.py docstring).
    background_tasks.add_task(database.log_drift_report, report)

    return DriftResponse(**report)


@app.post("/retrain", response_model=RetrainingReport)
async def retrain(background_tasks: BackgroundTasks, force: bool = False, limit: int = 500) -> RetrainingReport:
    """
    Triggers the Phase 13 LangGraph retraining workflow (ml/retraining_graph.py).
    This is an administrative/offline endpoint, deliberately separate from the
    customer-facing /predict path -- see that module's docstring for why
    LangGraph never runs inside /predict.

    `force=True` bypasses the drift check and always trains a candidate --
    useful for demonstrating the workflow without waiting for real drift to
    accumulate in a prototype with limited traffic. `limit` controls how
    many recent predictions are pulled for the drift check, same meaning as
    on GET /drift. A trained candidate is never auto-promoted to production
    -- see ml/retraining_graph.py for why that's Phase 14's job.
    """
    if limit < MIN_SAMPLE_SIZE or limit > 5000:
        raise HTTPException(status_code=422, detail=f"limit must be between {MIN_SAMPLE_SIZE} and 5000")

    recent_inputs = database.get_recent_prediction_inputs(limit=limit)

    try:
        report = run_retraining_workflow(recent_inputs=recent_inputs, force=force)
    except Exception as e:  # noqa: BLE001
        logger.exception("Retraining workflow failed")
        raise HTTPException(status_code=500, detail=f"Retraining workflow failed: {e}") from e

    # Logged in the background, after the response is built -- same reasoning
    # as /predict and /drift's background logging (see database.py docstring).
    background_tasks.add_task(database.log_retraining_run, report)

    return RetrainingReport(**report)


@app.get("/model-versions", response_model=VersionRegistryResponse)
async def model_versions() -> VersionRegistryResponse:
    """
    Read-only view of the version registry (backend/versioning.py) -- which
    version is currently active/serving, and every version ever promoted
    or rolled back to. Reads the local JSON registry directly, not
    MongoDB, since the registry (not Mongo) is the actual source of truth
    for what /predict is serving.
    """
    return VersionRegistryResponse(**load_registry())


@app.post("/promote-candidate", response_model=VersionActionResponse)
async def promote_candidate_endpoint(background_tasks: BackgroundTasks) -> VersionActionResponse:
    """
    Promotes the current candidate (models/candidate_model.pkl, produced by
    POST /retrain in Phase 13) to a new, permanent production version, and
    immediately reloads backend/predictor.py's in-memory model so /predict
    starts serving it right away -- no server restart needed.

    Deliberately a separate, explicit admin action from /retrain: Phase
    13's retraining graph never auto-promotes (see ml/retraining_graph.py's
    docstring), so a bad candidate can never reach production without a
    human choosing to call this endpoint.
    """
    try:
        result = versioning_promote_candidate()
    except VersioningError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    predictor.reload()
    if not predictor.is_loaded:
        # Extremely unlikely (the file we just wrote is unreadable), but if
        # it happens, surface it loudly rather than silently serving a
        # broken model.
        raise HTTPException(
            status_code=500,
            detail=(
                f"Promoted to {result['active_version']} but the model failed to "
                f"reload: {predictor.load_error}"
            ),
        )

    background_tasks.add_task(database.log_version_event, {"action": "promote", **result})
    return VersionActionResponse(**result)


@app.post("/rollback", response_model=VersionActionResponse)
async def rollback_endpoint(
    background_tasks: BackgroundTasks, to_version: str | None = None
) -> VersionActionResponse:
    """
    Reverts the active model version -- to `to_version` if given, otherwise
    to whichever version was active immediately before the current one --
    and reloads backend/predictor.py's in-memory model immediately.
    Answers the project spec's requirement directly: "If a new model
    performs poorly, you must be able to quickly rollback to the previous
    version."
    """
    try:
        result = versioning_rollback(to_version=to_version)
    except VersioningError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    predictor.reload()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Rolled back to {result['active_version']} but the model failed to "
                f"reload: {predictor.load_error}"
            ),
        )

    background_tasks.add_task(database.log_version_event, {"action": "rollback", **result})
    return VersionActionResponse(**result)


@app.get("/fairness", response_model=FairnessResponse)
async def fairness(background_tasks: BackgroundTasks, limit: int = 500) -> FairnessResponse:
    """
    Two-part fairness view (project spec Phase 15). See ml/fairness.py's
    module docstring for why these are fundamentally different kinds of
    evidence:

    1. `training_report` -- the ground-truth-backed audit computed on the
       held-out test set the last time the ACTIVE model was trained
       (stored in its metadata by ml/train.py -- see also
       notebooks/03_model_training.ipynb's Section 12 for the same logic,
       duplicated intentionally, run manually). `training_report_note`
       explains when this is null: the active model predates this phase
       and has no fairness_report field yet.

    2. `live_snapshot` -- a demographic-parity-ONLY proxy computed from
       recent live predictions, which have no ground truth
       (`actual_outcome` is always None -- see database.py). This can show
       one kind of imbalance (who gets flagged as high-risk more often)
       but cannot show recall/false-positive-rate parity, which requires
       knowing who actually churned.
    """
    if limit < MIN_SAMPLE_SIZE or limit > 5000:
        raise HTTPException(status_code=422, detail=f"limit must be between {MIN_SAMPLE_SIZE} and 5000")

    training_report = predictor.metadata.get("fairness_report")
    training_report_note = None
    if training_report is None:
        training_report_note = (
            "The active model's metadata has no fairness_report -- it was trained "
            "before this audit existed. Retrain (POST /retrain) and promote a "
            "candidate (POST /promote-candidate) to generate one."
        )

    recent_predictions = database.get_recent_predictions(limit=limit)
    n_current = len(recent_predictions)

    live_snapshot = None
    live_snapshot_note = None
    if n_current < MIN_SAMPLE_SIZE:
        live_snapshot_note = (
            f"Only {n_current} recent predictions available; need at least "
            f"{MIN_SAMPLE_SIZE} for a meaningful snapshot."
        )
    else:
        inputs_df = pd.DataFrame([r["input"] for r in recent_predictions])
        y_pred_binary = np.array(
            [1 if r["prediction"] == "CHURN" else 0 for r in recent_predictions]
        )
        live_snapshot = audit_all_attributes_selection_rate_only(inputs_df, y_pred_binary)
        background_tasks.add_task(
            database.log_fairness_snapshot, {"n_current_rows": n_current, "live_snapshot": live_snapshot}
        )

    return FairnessResponse(
        training_report=training_report,
        training_report_note=training_report_note,
        live_snapshot=live_snapshot,
        live_snapshot_note=live_snapshot_note,
        n_current_rows=n_current,
    )


@app.get("/business-impact", response_model=BusinessImpactResponse)
async def business_impact(
    retention_success_rate: float = DEFAULT_ASSUMPTIONS["retention_success_rate"],
    intervention_cost_per_contact: float = DEFAULT_ASSUMPTIONS["intervention_cost_per_contact"],
    customer_lifetime_months: int = DEFAULT_ASSUMPTIONS["customer_lifetime_months"],
    scale_to_population_param: int | None = None,
) -> BusinessImpactResponse:
    """
    Business-impact simulation (project spec Phase 16). See
    ml/business_impact.py's module docstring for the full honesty caveat --
    in short: `confusion_matrix` and `avg_monthly_revenue` come from the
    active model's last training run (real data); `retention_success_rate`,
    `intervention_cost_per_contact`, and `customer_lifetime_months` are
    business assumptions, overridable here via query params, defaulting to
    ml.business_impact.DEFAULT_ASSUMPTIONS. Nothing returned here is a
    measured outcome.

    `scale_to_population_param` optionally projects the report to a larger
    hypothetical customer base (e.g. 10_000_000, matching the interview
    spec's framing) -- see ml.business_impact.scale_to_population()'s
    docstring for the assumption this relies on.

    `live_projection` extrapolates from live prediction volume (GET
    /metrics' churn_count) using the training-time precision -- a rougher,
    unverified estimate, not the main event here.
    """
    inputs = predictor.metadata.get("business_impact_inputs")
    training_report_note = None
    training_report = None
    if inputs is None:
        training_report_note = (
            "The active model's metadata has no business_impact_inputs -- it was "
            "trained before this simulation existed. Retrain (POST /retrain) and "
            "promote a candidate (POST /promote-candidate) to generate one."
        )
    else:
        training_report = simulate_business_impact(
            confusion_matrix=inputs["confusion_matrix"],
            avg_monthly_revenue=inputs["avg_monthly_revenue"],
            retention_success_rate=retention_success_rate,
            intervention_cost_per_contact=intervention_cost_per_contact,
            customer_lifetime_months=customer_lifetime_months,
        )

    scaled_report = None
    if training_report is not None and scale_to_population_param is not None:
        cm = training_report["confusion_matrix"]
        source_population = sum(cm.values())
        scaled_report = scale_to_population(
            training_report, target_population=scale_to_population_param, source_population=source_population
        )

    live_projection = None
    live_projection_note = None
    if inputs is None:
        live_projection_note = "Unavailable -- same reason as training_report_note above."
    else:
        live_metrics = database.get_monitoring_metrics()
        total_predictions = live_metrics["total_predictions"]
        churn_flagged = live_metrics["churn_count"]
        cm = inputs["confusion_matrix"]
        predicted_positive = cm["true_positive"] + cm["false_positive"]
        training_precision = (
            round(cm["true_positive"] / predicted_positive, 4) if predicted_positive > 0 else None
        )

        if total_predictions == 0 or training_precision is None:
            live_projection_note = "No live predictions logged yet -- nothing to project from."
        else:
            projected_retained = churn_flagged * training_precision * retention_success_rate
            projected_revenue_saved = projected_retained * inputs["avg_monthly_revenue"] * customer_lifetime_months
            projected_cost = churn_flagged * intervention_cost_per_contact
            live_projection = {
                "based_on_total_predictions": total_predictions,
                "based_on_churn_flagged": churn_flagged,
                "training_precision_used": training_precision,
                "projected_customers_retained": round(projected_retained, 2),
                "projected_revenue_saved": round(projected_revenue_saved, 2),
                "projected_intervention_cost": round(projected_cost, 2),
                "projected_net_benefit": round(projected_revenue_saved - projected_cost, 2),
                "note": (
                    "Extrapolated from live CHURN-flagged volume times the "
                    "training-time precision and the same business assumptions "
                    "above -- not verified against real outcomes (no labeled "
                    "feedback loop exists yet)."
                ),
            }

    return BusinessImpactResponse(
        training_report=training_report,
        training_report_note=training_report_note,
        scaled_report=scaled_report,
        live_projection=live_projection,
        live_projection_note=live_projection_note,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerInput, background_tasks: BackgroundTasks) -> PredictionResponse:
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not currently loaded")

    try:
        customer_dict = customer.model_dump()
        result = predictor.predict(customer_dict)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    # Logged AFTER the response is prepared, via a background task, so a slow
    # or unavailable database never adds to the latency the customer-service
    # rep experiences. See backend/database.py docstring point 3.
    background_tasks.add_task(database.log_prediction, customer_dict, result)

    return PredictionResponse(**result)


@app.get("/predictions", response_model=list[PredictionRecord])
async def recent_predictions(limit: int = 20) -> list[PredictionRecord]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    records = database.get_recent_predictions(limit=limit)
    return [PredictionRecord(**r) for r in records]
