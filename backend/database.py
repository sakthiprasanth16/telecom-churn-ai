"""
backend/database.py
--------------------
MongoDB Atlas connection, used for:
- Logging every prediction (the `predictions` collection)
- Reading back recent prediction history (for GET /predictions and, later,
  the Streamlit monitoring dashboard)

Design decisions:

1. Connect ONCE, at import time (same pattern as the ML model in predictor.py).
   We do not open a new connection per request -- pymongo's MongoClient is
   itself a connection pool and is meant to be created once and reused.

2. Graceful degradation. If MongoDB is unreachable (wrong URI, network issue,
   Atlas cluster paused, etc.), /predict must still return a prediction to the
   customer-service rep -- a logging failure should never block a real-time
   business-critical request. We catch connection/write errors, log a warning,
   and continue. `/health` reports `mongo_connected: false` so this is visible
   in monitoring, rather than failing silently forever.

3. Background writes. main.py calls `log_prediction_background()` via FastAPI's
   BackgroundTasks, so the MongoDB write happens AFTER the response has already
   been sent back to the customer-service rep -- database latency never adds to
   the reported prediction latency_ms. This is the "asynchronous/background
   processing for non-critical logging" scalability point from the project spec.

4. Local testing without a real Atlas cluster: if the MONGODB_URI environment
   variable is missing or set to the literal value "mongomock", this module
   uses the `mongomock` library instead of a real MongoDB connection. This
   exists ONLY so this module can be developed and tested without network
   access to Atlas. It is not a production code path -- in real deployment,
   MONGODB_URI will always be a real mongodb+srv://... connection string,
   which is required in the .env file (see .env.example).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("churn-api.database")

MONGODB_URI = os.environ.get("MONGODB_URI", "mongomock")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "telecom_churn")
# Fail fast rather than hang if Atlas is unreachable -- a stuck connection
# attempt at startup would delay the whole API from becoming healthy.
SERVER_SELECTION_TIMEOUT_MS = int(os.environ.get("MONGODB_TIMEOUT_MS", "5000"))


class Database:
    def __init__(self):
        self._client = None
        self._db = None
        self._connect_error: str | None = None
        self._connect()

    def _connect(self) -> None:
        try:
            if MONGODB_URI == "mongomock":
                # Local-testing-only path -- see module docstring point 4.
                import mongomock
                logger.warning(
                    "MONGODB_URI not set (or explicitly 'mongomock') -- using an "
                    "in-memory mock database. Predictions will NOT persist. Set a "
                    "real MONGODB_URI in your .env file for actual use."
                )
                self._client = mongomock.MongoClient()
            else:
                import pymongo
                self._client = pymongo.MongoClient(
                    MONGODB_URI,
                    serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
                )
                # Force a round-trip now, at startup, rather than discovering
                # a bad URI on the first prediction request.
                self._client.admin.command("ping")
                logger.info("Connected to MongoDB Atlas successfully.")

            self._db = self._client[MONGODB_DB_NAME]
        except Exception as e:  # noqa: BLE001
            self._connect_error = str(e)
            self._client = None
            self._db = None
            logger.error("MongoDB connection failed: %s", e)

    @property
    def is_connected(self) -> bool:
        return self._db is not None

    @property
    def connect_error(self) -> str | None:
        return self._connect_error

    def log_prediction(self, customer_input: dict[str, Any], result: dict[str, Any]) -> None:
        """
        Insert one prediction record. Never raises -- a logging failure must
        not be allowed to affect the customer-facing prediction flow. Intended
        to be called from a FastAPI BackgroundTask, after the response has
        already been sent.
        """
        if not self.is_connected:
            logger.warning("Skipping prediction log: MongoDB not connected.")
            return
        try:
            document = {
                "input": customer_input,
                "prediction": result["prediction"],
                "probability": result["probability"],
                "risk": result["risk"],
                "model_version": result["model_version"],
                "latency_ms": result["latency_ms"],
                "timestamp": datetime.now(timezone.utc),
                # Populated later, when/if a real outcome becomes known --
                # see module docstring / project spec section 13 ("New Data
                # Handling"): we do not assume an immediately available label.
                "actual_outcome": None,
            }
            self._db["predictions"].insert_one(document)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log prediction to MongoDB: %s", e)

    def get_recent_prediction_inputs(self, limit: int = 500) -> list[dict[str, Any]]:
        """
        Returns just the raw `input` field from the most recent predictions,
        newest first -- i.e. exactly the Kaggle-shaped fields a rep typed into
        Streamlit, which is what `ml.drift.detect_drift()` compares against
        the training-time reference distribution (see ml/drift.py docstring).

        Reuses the same `predictions` collection `log_prediction()` already
        writes to rather than a separate collection, since the raw input was
        already being stored there for every prediction. Empty list if the
        database is unavailable or nothing has been predicted yet -- the
        caller (the /drift endpoint) is expected to treat that the same way
        `ml.drift.detect_drift()` treats an undersized sample: as
        "insufficient_data", not as "no drift".
        """
        if not self.is_connected:
            return []
        try:
            cursor = (
                self._db["predictions"]
                .find({}, {"_id": 0, "input": 1})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return [doc["input"] for doc in cursor if "input" in doc]
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to read prediction inputs from MongoDB: %s", e)
            return []

    def log_drift_report(self, report: dict[str, Any]) -> None:
        """
        Insert one drift report snapshot into the `drift_reports` collection
        (project spec section 14). Never raises -- same graceful-degradation
        pattern as `log_prediction()`: a failure to persist a drift report
        must never break the `/drift` endpoint's response to the caller.
        Intended to be called from a FastAPI BackgroundTask, after the
        response has already been prepared.
        """
        if not self.is_connected:
            logger.warning("Skipping drift report log: MongoDB not connected.")
            return
        try:
            document = {**report, "logged_at": datetime.now(timezone.utc)}
            self._db["drift_reports"].insert_one(document)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log drift report to MongoDB: %s", e)

    def log_retraining_run(self, report: dict[str, Any]) -> None:
        """
        Insert one retraining-workflow report into the `retraining_runs`
        collection (Phase 13 -- see ml/retraining_graph.py). Same
        graceful-degradation and background-task pattern as
        `log_drift_report()`: a failure to persist this must never break
        the `/retrain` endpoint's response to the caller.
        """
        if not self.is_connected:
            logger.warning("Skipping retraining run log: MongoDB not connected.")
            return
        try:
            document = {**report, "logged_at": datetime.now(timezone.utc)}
            self._db["retraining_runs"].insert_one(document)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log retraining run to MongoDB: %s", e)

    def log_version_event(self, event: dict[str, Any]) -> None:
        """
        Insert one promote/rollback event into the `model_version_events`
        collection (Phase 14 -- see backend/versioning.py). This is an
        AUDIT TRAIL only -- backend/versioning.py's local JSON registry,
        not this collection, is the actual source of truth for which
        model is being served. If MongoDB is down, promote/rollback still
        work; only the audit record is missed (same graceful-degradation
        pattern as every other log_* method here).
        """
        if not self.is_connected:
            logger.warning("Skipping version event log: MongoDB not connected.")
            return
        try:
            document = {**event, "logged_at": datetime.now(timezone.utc)}
            self._db["model_version_events"].insert_one(document)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log version event to MongoDB: %s", e)

    def log_fairness_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        Insert one live fairness snapshot into the `fairness_snapshots`
        collection (Phase 15 -- see ml/fairness.py and GET /fairness in
        backend/main.py). Same graceful-degradation pattern as every other
        log_* method: a failure to persist this must never break the
        /fairness endpoint's response to the caller.
        """
        if not self.is_connected:
            logger.warning("Skipping fairness snapshot log: MongoDB not connected.")
            return
        try:
            document = {**snapshot, "logged_at": datetime.now(timezone.utc)}
            self._db["fairness_snapshots"].insert_one(document)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log fairness snapshot to MongoDB: %s", e)

    def get_recent_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Returns the most recent predictions, newest first. Empty list if DB unavailable."""
        if not self.is_connected:
            return []
        try:
            cursor = (
                self._db["predictions"]
                .find({}, {"_id": 0})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(cursor)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to read predictions from MongoDB: %s", e)
            return []

    def count_predictions_today(self) -> int:
        if not self.is_connected:
            return 0
        try:
            start_of_day = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return self._db["predictions"].count_documents(
                {"timestamp": {"$gte": start_of_day}}
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to count today's predictions: %s", e)
            return 0

    def get_monitoring_metrics(self) -> dict[str, Any]:
        """
        Aggregated stats for the monitoring dashboard (GET /metrics). Reads
        directly from the `predictions` collection -- this is intentionally
        computed on read, not maintained as a running counter, since a
        prototype's prediction volume is small enough that this is cheap
        and it avoids a second source of truth that could drift out of sync.
        In a high-volume production system this would instead be a
        periodically-refreshed materialized view or a time-series
        aggregation job, since scanning every prediction on every dashboard
        load would not scale to millions of predictions/day.
        """
        empty = {
            "total_predictions": 0,
            "predictions_today": 0,
            "avg_latency_ms": 0.0,
            "churn_count": 0,
            "no_churn_count": 0,
            "risk_breakdown": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        }
        if not self.is_connected:
            return empty
        try:
            coll = self._db["predictions"]
            total = coll.count_documents({})
            if total == 0:
                return empty

            today = self.count_predictions_today()

            avg_latency_cursor = coll.aggregate([
                {"$group": {"_id": None, "avg_latency": {"$avg": "$latency_ms"}}}
            ])
            avg_latency_doc = next(avg_latency_cursor, None)
            avg_latency = round(avg_latency_doc["avg_latency"], 2) if avg_latency_doc else 0.0

            churn_count = coll.count_documents({"prediction": "CHURN"})
            no_churn_count = coll.count_documents({"prediction": "NO_CHURN"})

            risk_breakdown = {
                "HIGH": coll.count_documents({"risk": "HIGH"}),
                "MEDIUM": coll.count_documents({"risk": "MEDIUM"}),
                "LOW": coll.count_documents({"risk": "LOW"}),
            }

            return {
                "total_predictions": total,
                "predictions_today": today,
                "avg_latency_ms": avg_latency,
                "churn_count": churn_count,
                "no_churn_count": no_churn_count,
                "risk_breakdown": risk_breakdown,
            }
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to compute monitoring metrics: %s", e)
            return empty


# Instantiated once, at import time -- same pattern as the model in predictor.py.
database = Database()
