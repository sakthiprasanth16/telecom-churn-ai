r"""
ml/retraining_graph.py
-----------------------
LangGraph-orchestrated retraining workflow (project spec Phase 13: "LangGraph
ML lifecycle workflow (retraining orchestration)").

Deliberately kept OUT of the real-time prediction path -- per the project's
engineering rules, LangGraph never runs inside POST /predict. It only runs
when a human (or later, a scheduled job) hits POST /retrain in
backend/main.py, so a slow or failed retraining run can never affect
customer-service-facing prediction latency.

Graph shape:

    START -> check_drift --(significant drift OR forced)--> load_training_data
                         \--(no drift, not forced)---------> skip -> END

    load_training_data --(file found)-------> train_candidate -> evaluate_candidate -> END
                        \--(file missing)----> END (decision="error")

What this graph deliberately does NOT do (out of scope for Phase 13):

- It does NOT overwrite the production model. A candidate that clears the
  evaluation bar is saved to models/candidate_model.pkl /
  candidate_model_metadata.json -- never models/model_v1.pkl. Promoting a
  candidate to production, model/version tracking, and rollback are Phase
  14's job. Mixing an automatic promotion into this graph would let a bad
  automated decision take down the real-time prediction path with no human
  in the loop -- exactly what the project spec's "retraining must be done
  carefully to avoid deploying worse models" requirement warns against.

- It does NOT invent new labeled training data. A real production retrain
  would incorporate actual churn outcomes fed back over time (see
  backend/database.py's `actual_outcome` field on each logged prediction --
  currently always None, since no such feedback loop is built yet). Until
  that exists, this graph retrains on the same base dataset
  (data/processed/churn_cleaned_v1.csv) that produced model_v1, so a
  drift-triggered "retrain" mostly demonstrates the orchestration mechanics
  end-to-end -- stated here plainly rather than presented as if the model
  were learning from new customer behavior it doesn't actually have access
  to yet.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import joblib
import pandas as pd
from langgraph.graph import StateGraph, START, END

from ml.drift import load_reference_stats, detect_drift
from ml.train import train_model

logger = logging.getLogger("churn-api.retraining")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIFT_REFERENCE_PATH = PROJECT_ROOT / "models" / "drift_reference_v1.json"
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_cleaned_v1.csv"
CURRENT_METADATA_PATH = PROJECT_ROOT / "models" / "model_v1_metadata.json"
CANDIDATE_MODEL_PATH = PROJECT_ROOT / "models" / "candidate_model.pkl"
CANDIDATE_METADATA_PATH = PROJECT_ROOT / "models" / "candidate_model_metadata.json"

# A candidate must not regress more than this on either metric to be
# approved -- a small, explicit, explainable tolerance, rather than requiring
# a strict improvement (which would make retraining on essentially-the-same
# data always fail) or accepting any regression at all (which would defeat
# the point of a gate).
METRIC_REGRESSION_TOLERANCE = 0.02


class RetrainingState(TypedDict, total=False):
    force: bool
    recent_inputs: list[dict[str, Any]]
    drift_report: dict[str, Any]
    should_retrain: bool
    training_df: pd.DataFrame
    candidate_metrics: dict[str, float]
    candidate_training_rows: int
    candidate_test_rows: int
    candidate_fairness_report: dict[str, Any]
    current_metrics: dict[str, float]
    decision: str
    decision_reason: str


def _check_drift(state: RetrainingState) -> RetrainingState:
    reference_stats = load_reference_stats(DRIFT_REFERENCE_PATH)
    recent_inputs = state.get("recent_inputs", [])
    drift_report = detect_drift(reference_stats, pd.DataFrame(recent_inputs))
    should_retrain = bool(state.get("force")) or drift_report["status"] == "significant_drift"
    return {**state, "drift_report": drift_report, "should_retrain": should_retrain}


def _route_after_drift_check(state: RetrainingState) -> str:
    return "load_training_data" if state.get("should_retrain") else "skip"


def _skip(state: RetrainingState) -> RetrainingState:
    reason = f"No retraining triggered: drift status was '{state['drift_report']['status']}', and force=False."
    return {**state, "decision": "skipped_no_drift", "decision_reason": reason}


def _load_training_data(state: RetrainingState) -> RetrainingState:
    if not TRAINING_DATA_PATH.exists():
        return {
            **state,
            "decision": "error",
            "decision_reason": f"Training data not found at {TRAINING_DATA_PATH}",
        }
    df = pd.read_csv(TRAINING_DATA_PATH)
    return {**state, "training_df": df}


def _route_after_load(state: RetrainingState) -> str:
    return "error" if state.get("decision") == "error" else "train_candidate"


def _train_candidate(state: RetrainingState) -> RetrainingState:
    result = train_model(state["training_df"])

    joblib.dump(result["pipeline"], CANDIDATE_MODEL_PATH)
    metadata = {
        "version": "candidate",
        "model_type": type(result["pipeline"].named_steps["classifier"]).__name__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": result["training_rows"],
        "test_rows": result["test_rows"],
        **result["metrics"],
        "class_imbalance_handling": "class_weight='balanced'",
        "status": "candidate",
        "fairness_report": result["fairness_report"],
        "business_impact_inputs": result["business_impact_inputs"],
    }
    with open(CANDIDATE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        **state,
        "candidate_metrics": result["metrics"],
        "candidate_training_rows": result["training_rows"],
        "candidate_test_rows": result["test_rows"],
        "candidate_fairness_report": result["fairness_report"],
    }


def _evaluate_candidate(state: RetrainingState) -> RetrainingState:
    with open(CURRENT_METADATA_PATH) as f:
        current_metadata = json.load(f)
    current_metrics = {
        "precision": current_metadata.get("precision"),
        "recall": current_metadata.get("recall"),
        "f1": current_metadata.get("f1"),
        "roc_auc": current_metadata.get("roc_auc"),
    }
    candidate = state["candidate_metrics"]

    recall_ok = candidate["recall"] >= current_metrics["recall"] - METRIC_REGRESSION_TOLERANCE
    roc_auc_ok = candidate["roc_auc"] >= current_metrics["roc_auc"] - METRIC_REGRESSION_TOLERANCE

    # Fairness is surfaced in the decision reason but does NOT block
    # approval by itself this phase -- see ml/fairness.py's docstring for
    # why a disparity flag calls for human judgment (which mitigation
    # trade-off to accept) rather than an automatic reject. A human
    # reviewing this report before calling POST /promote-candidate is the
    # actual safeguard, same as recall/roc_auc never auto-promoting.
    fairness_report = state.get("candidate_fairness_report", {})
    flagged_attributes = [attr for attr, report in fairness_report.items() if report.get("flagged")]
    fairness_note = (
        f" Fairness audit flagged a disparity in: {', '.join(flagged_attributes)} "
        "(see candidate_model_metadata.json's fairness_report for details, including "
        "a suggested mitigation)."
        if flagged_attributes
        else " Fairness audit found no flagged disparities (gender, SeniorCitizen)."
    )

    if recall_ok and roc_auc_ok:
        decision = "candidate_approved"
        reason = (
            f"Candidate recall={candidate['recall']} and roc_auc={candidate['roc_auc']} are within "
            f"tolerance ({METRIC_REGRESSION_TOLERANCE}) of production (recall={current_metrics['recall']}, "
            f"roc_auc={current_metrics['roc_auc']}). Saved to {CANDIDATE_MODEL_PATH.name} for human "
            "review/promotion (Phase 14) -- NOT auto-deployed." + fairness_note
        )
    else:
        decision = "candidate_rejected"
        reason = (
            f"Candidate recall={candidate['recall']} or roc_auc={candidate['roc_auc']} regressed beyond "
            f"tolerance ({METRIC_REGRESSION_TOLERANCE}) vs production (recall={current_metrics['recall']}, "
            f"roc_auc={current_metrics['roc_auc']}). Not recommended for promotion." + fairness_note
        )

    return {**state, "current_metrics": current_metrics, "decision": decision, "decision_reason": reason}


def _build_graph():
    graph = StateGraph(RetrainingState)

    graph.add_node("check_drift", _check_drift)
    graph.add_node("skip", _skip)
    graph.add_node("load_training_data", _load_training_data)
    graph.add_node("train_candidate", _train_candidate)
    graph.add_node("evaluate_candidate", _evaluate_candidate)

    graph.add_edge(START, "check_drift")
    graph.add_conditional_edges(
        "check_drift",
        _route_after_drift_check,
        {"load_training_data": "load_training_data", "skip": "skip"},
    )
    graph.add_edge("skip", END)
    graph.add_conditional_edges(
        "load_training_data",
        _route_after_load,
        {"train_candidate": "train_candidate", "error": END},
    )
    graph.add_edge("train_candidate", "evaluate_candidate")
    graph.add_edge("evaluate_candidate", END)

    return graph.compile()


# Compiled once at import time -- same "load once, reuse" pattern as the
# model in predictor.py. The graph itself holds no per-request state.
retraining_graph = _build_graph()


def run_retraining_workflow(recent_inputs: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
    """
    Public entrypoint used by backend/main.py's POST /retrain. Runs the
    compiled graph and returns a JSON-serializable report -- the raw
    DataFrame and the input list are dropped from the returned dict since
    neither belongs in an API response or a MongoDB document.
    """
    final_state = retraining_graph.invoke({"recent_inputs": recent_inputs, "force": force})
    return {k: v for k, v in final_state.items() if k not in ("training_df", "recent_inputs")}
