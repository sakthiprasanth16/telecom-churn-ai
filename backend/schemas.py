"""
backend/schemas.py
-------------------
Pydantic models for request validation and response shaping.

Validation happens here, at the API boundary, before any data reaches the model.
This is the "Validate input" step in the architecture:

    User -> Streamlit -> FastAPI -> Pydantic validation -> Preprocessing -> Model

Field-level validators reject invalid types and out-of-range values (e.g. negative
tenure, a MonthlyCharges of zero) with a clear 422 error, rather than letting bad
data reach the model and produce a nonsense prediction silently.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerInput(BaseModel):
    """
    Raw customer attributes, in the same shape as the original Kaggle dataset
    columns. This is exactly what a customer-service representative would type
    into the Streamlit form.
    """

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="1 if the customer is a senior citizen, else 0"
    )
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., gt=0, le=1000)
    TotalCharges: float = Field(..., ge=0, le=100000)

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_consistent_with_tenure(cls, v: float, info) -> float:
        # Business-rule check mirroring the cleaning rule from notebook 01:
        # TotalCharges should be 0 only for brand-new customers (tenure == 0).
        # A nonzero TotalCharges with tenure == 0 is a data-entry error, not a
        # valid edge case, so we reject it here instead of silently modeling on it.
        tenure = info.data.get("tenure")
        if tenure == 0 and v > 0:
            raise ValueError(
                "TotalCharges must be 0 when tenure is 0 (new customer not yet billed)"
            )
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class ExplanationFactor(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    human_readable: str


class PredictionResponse(BaseModel):
    prediction: Literal["CHURN", "NO_CHURN"]
    probability: float
    risk: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    latency_ms: float
    top_factors: list[ExplanationFactor] = []


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    mongo_connected: bool = False


class PredictionRecord(BaseModel):
    input: dict
    prediction: Literal["CHURN", "NO_CHURN"]
    probability: float
    risk: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    latency_ms: float
    timestamp: str
    actual_outcome: bool | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def stringify_timestamp(cls, v):
        # MongoDB returns a datetime object; convert to ISO string for the
        # JSON response so this schema doesn't need to know it's Mongo-backed.
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return v


class ModelInfoResponse(BaseModel):
    version: str
    model_type: str
    dataset_version: str
    feature_version: str
    trained_at: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    status: str


class DriftFeatureResult(BaseModel):
    psi: float
    status: Literal["no_drift", "moderate_drift", "significant_drift"]
    reference_mean: float | None = None
    current_mean: float | None = None
    unseen_categories: list[str] | None = None


class DriftThresholds(BaseModel):
    moderate: float
    significant: float


class DriftResponse(BaseModel):
    """
    Mirrors the two shapes `ml.drift.detect_drift()` can return: a full
    report once at least MIN_SAMPLE_SIZE recent predictions exist, or the
    lighter "insufficient_data" shape before that. All report-only fields
    are optional so both shapes validate against this one schema.
    """

    status: Literal["no_drift", "moderate_drift", "significant_drift", "insufficient_data"]
    n_reference_rows: int | None = None
    n_current_rows: int
    overall_max_psi: float | None = None
    drifted_features: list[str] = []
    numeric_features: dict[str, DriftFeatureResult] = {}
    categorical_features: dict[str, DriftFeatureResult] = {}
    thresholds: DriftThresholds | None = None
    message: str | None = None


class RetrainingReport(BaseModel):
    """
    Mirrors the dict shape ml.retraining_graph.run_retraining_workflow()
    returns. `drift_report` reuses DriftResponse since both are produced by
    the same ml.drift.detect_drift() function. Metric/candidate fields are
    optional because they're only populated on the branches of the graph
    that actually train a candidate (see ml/retraining_graph.py's docstring
    for the full branch map).
    """

    force: bool
    should_retrain: bool | None = None
    drift_report: DriftResponse | None = None
    candidate_metrics: dict[str, float] | None = None
    candidate_training_rows: int | None = None
    candidate_test_rows: int | None = None
    current_metrics: dict[str, float | None] | None = None
    decision: Literal["skipped_no_drift", "error", "candidate_approved", "candidate_rejected"]
    decision_reason: str | None = None


class ModelVersionInfo(BaseModel):
    model_file: str
    metadata_file: str
    status: Literal["production", "archived", "rolled_back"]
    activated_at: str
    metrics: dict[str, float | None]
    source: Literal["initial_training", "retraining_workflow"]


class VersionRegistryResponse(BaseModel):
    """Mirrors backend/versioning.py's registry file exactly."""

    active_version: str
    history: list[str]
    versions: dict[str, ModelVersionInfo]


class VersionActionResponse(BaseModel):
    """Returned by both POST /promote-candidate and POST /rollback."""

    active_version: str
    previous_version: str
    version_info: ModelVersionInfo


class MitigationRecommendation(BaseModel):
    strategy: str | None = None
    target_recall: float | None = None
    suggested_thresholds: dict[str, float] | None = None
    simulated_selection_rate_after_mitigation: dict[str, float] | None = None
    simulated_selection_rate_ratio_after_mitigation: float | None = None
    note: str | None = None


class TrainingFairnessAttributeReport(BaseModel):
    """One attribute's ground-truth-backed audit, as stored in a model's metadata."""

    group_sizes: dict[str, int]
    selection_rate: dict[str, float]
    recall: dict[str, float]
    false_positive_rate: dict[str, float]
    disparities: dict[str, float | None]
    flagged: bool
    mitigation: MitigationRecommendation | None = None


class LiveFairnessAttributeReport(BaseModel):
    """One attribute's demographic-parity-only snapshot from recent live predictions."""

    group_sizes: dict[str, int]
    selection_rate: dict[str, float]
    selection_rate_ratio: float | None = None
    flagged: bool


class FairnessResponse(BaseModel):
    """
    Two-part fairness view -- see backend/main.py's GET /fairness docstring
    for why these are fundamentally different kinds of evidence and why
    `training_report` can be null (the active model predates this phase).
    """

    training_report: dict[str, TrainingFairnessAttributeReport] | None = None
    training_report_note: str | None = None
    live_snapshot: dict[str, LiveFairnessAttributeReport] | None = None
    live_snapshot_note: str | None = None
    n_current_rows: int


class BusinessImpactConfusionMatrix(BaseModel):
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


class BusinessImpactAssumptions(BaseModel):
    avg_monthly_revenue: float
    retention_success_rate: float
    intervention_cost_per_contact: float
    customer_lifetime_months: int


class BusinessImpactScenario(BaseModel):
    customers_flagged: float
    expected_customers_retained: float
    estimated_revenue_saved: float
    estimated_intervention_cost: float
    estimated_net_benefit: float
    roi: float | None = None


class BusinessImpactReport(BaseModel):
    confusion_matrix: BusinessImpactConfusionMatrix
    assumptions: BusinessImpactAssumptions
    scenarios: dict[str, BusinessImpactScenario]
    missed_revenue_at_risk: float
    note: str


class ScaledBusinessImpactReport(BaseModel):
    scale_factor: float
    target_population: int
    source_population: int
    confusion_matrix: dict[str, float]
    scenarios: dict[str, BusinessImpactScenario]
    note: str


class BusinessImpactLiveProjection(BaseModel):
    """
    A rough projection from live prediction volume (GET /metrics'
    churn_count) times the training-time precision and the same business
    assumptions -- NOT verified against real outcomes, since no
    ground-truth feedback loop exists yet (see ml/retraining_graph.py's
    docstring on `actual_outcome`).
    """

    based_on_total_predictions: int
    based_on_churn_flagged: int
    training_precision_used: float | None = None
    projected_customers_retained: float
    projected_revenue_saved: float
    projected_intervention_cost: float
    projected_net_benefit: float
    note: str


class BusinessImpactResponse(BaseModel):
    """
    See backend/main.py's GET /business-impact docstring: `training_report`
    is a live simulation (query params can override the default
    assumptions) built from the active model's training-time confusion
    matrix; `training_report_note` explains when it's null (the active
    model predates this phase). `scaled_report` is only present if the
    caller passed `scale_to_population`. `live_projection` extrapolates
    from live prediction volume and is a rougher, unverified estimate.
    """

    training_report: BusinessImpactReport | None = None
    training_report_note: str | None = None
    scaled_report: ScaledBusinessImpactReport | None = None
    live_projection: BusinessImpactLiveProjection | None = None
    live_projection_note: str | None = None


class RiskBreakdown(BaseModel):
    HIGH: int
    MEDIUM: int
    LOW: int


class MetricsResponse(BaseModel):
    total_predictions: int
    predictions_today: int
    avg_latency_ms: float
    churn_count: int
    no_churn_count: int
    risk_breakdown: RiskBreakdown
