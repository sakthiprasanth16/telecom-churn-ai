"""
backend/versioning.py
----------------------
Model version registry + promote/rollback logic (Phase 14: "Versioning and
Rollback"). Answers the project spec's requirement directly: "If a new
model performs poorly, you must be able to quickly rollback to the
previous version."

Design:

- models/version_registry.json is the single source of truth for which
  model version is currently "active" -- i.e. what backend/predictor.py
  loads and serves. It's a plain JSON file, not a MongoDB collection,
  because promotion, rollback, and serving must all keep working even if
  MongoDB is completely unreachable -- same principle as everywhere else
  in this project (see database.py's docstring). MongoDB is used only as
  an audit trail (see database.log_version_event), never as the mechanism
  that decides what actually gets served.

- Promoting Phase 13's candidate (models/candidate_model.pkl +
  candidate_model_metadata.json, produced by POST /retrain in
  ml/retraining_graph.py) does NOT overwrite the candidate's slot in
  place. It's copied into a new, permanent, numbered slot
  (models/model_v{N}.pkl), so every version that was ever in production
  stays on disk and in the registry -- rollback needs the actual old
  file, not just a metadata record that it once existed.

- Self-healing bootstrap: if version_registry.json doesn't exist yet (a
  deployment created before this phase existed), one is generated
  automatically from whatever model_v1.pkl/model_v1_metadata.json is
  already there, so this phase doesn't require a manual migration step --
  same "graceful, self-healing" spirit as database.py's mongomock
  fallback and ml/drift.py's insufficient_data handling.

- Nothing here ever deletes a model file. Rollback and a later "roll
  forward" both just repoint `active_version` at an existing file.

Concurrency: `_write_registry()` writes to a temp file and atomically
renames it over the real one (`os.replace`, atomic on both POSIX and
Windows), so a concurrent reader can only ever see a fully-old or
fully-new registry, never a half-written/corrupted one. `promote_candidate()`
and `rollback()` additionally hold `_registry_lock` around their
read-modify-write sequence, which prevents a lost update (two promotions
racing and one silently overwriting the other) WITHIN a single process.
This does NOT protect against multiple Uvicorn worker processes (each
has its own independent lock and its own memory) -- a production
deployment with `--workers > 1` would need either a database-backed
atomic compare-and-swap (e.g. MongoDB's findAndModify) or a cross-process
file lock, neither of which this project's registry uses today. Stated
plainly rather than silently assumed away, per this project's "never
claim untested enterprise-scale performance" rule.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REGISTRY_PATH = MODELS_DIR / "version_registry.json"
V1_MODEL_PATH = MODELS_DIR / "model_v1.pkl"
V1_METADATA_PATH = MODELS_DIR / "model_v1_metadata.json"
CANDIDATE_MODEL_PATH = MODELS_DIR / "candidate_model.pkl"
CANDIDATE_METADATA_PATH = MODELS_DIR / "candidate_model_metadata.json"

_METRIC_KEYS = ("precision", "recall", "f1", "roc_auc")

# Guards the read-modify-write sequence in promote_candidate()/rollback()
# against a lost update from two requests racing within this process. See
# the concurrency note in the module docstring for what this does and does
# NOT protect against.
_registry_lock = threading.Lock()


class VersioningError(Exception):
    """Raised for any invalid promote/rollback request. Caught in
    backend/main.py and turned into a 409, never a raw 500."""


def _write_registry(registry: dict[str, Any]) -> None:
    """
    Writes via a temp file + atomic rename (os.replace) rather than
    writing REGISTRY_PATH directly, so a concurrent reader (e.g.
    get_active_model_paths(), called by predictor.py) can only ever
    observe a complete old file or a complete new file -- never a
    half-written, corrupted one caught mid-write.
    """
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_path, REGISTRY_PATH)


def _bootstrap_registry() -> dict[str, Any]:
    if V1_METADATA_PATH.exists():
        with open(V1_METADATA_PATH) as f:
            v1_metadata = json.load(f)
    else:
        v1_metadata = {}

    now = datetime.now(timezone.utc).isoformat()
    registry = {
        "active_version": "v1",
        "history": ["v1"],
        "versions": {
            "v1": {
                "model_file": "model_v1.pkl",
                "metadata_file": "model_v1_metadata.json",
                "status": "production",
                "activated_at": v1_metadata.get("trained_at", now),
                "metrics": {k: v1_metadata.get(k) for k in _METRIC_KEYS},
                "source": "initial_training",
            }
        },
    }
    _write_registry(registry)
    return registry


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return _bootstrap_registry()
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def get_active_model_paths() -> tuple[Path, Path]:
    """
    Used by backend/predictor.py at import time (and again inside
    reload()) to know which files to load. This is the only place outside
    this module that needs to know the registry exists at all.
    """
    registry = load_registry()
    active = registry["active_version"]
    info = registry["versions"][active]
    return MODELS_DIR / info["model_file"], MODELS_DIR / info["metadata_file"]


def _next_version_id(registry: dict[str, Any]) -> str:
    existing_numbers = [
        int(v[1:]) for v in registry["versions"] if v.startswith("v") and v[1:].isdigit()
    ]
    return f"v{max(existing_numbers, default=0) + 1}"


def _promote_candidate_impl() -> dict[str, Any]:
    """
    Promotes models/candidate_model.pkl to a new, permanent production
    version. Never called automatically -- always an explicit admin
    action, triggered only via backend/main.py's POST /promote-candidate
    (see ml/retraining_graph.py's docstring for why Phase 13 never
    auto-promotes).
    """
    if not CANDIDATE_MODEL_PATH.exists() or not CANDIDATE_METADATA_PATH.exists():
        raise VersioningError(
            "No candidate model found. Call POST /retrain first to produce "
            "models/candidate_model.pkl."
        )

    registry = load_registry()
    with open(CANDIDATE_METADATA_PATH) as f:
        candidate_metadata = json.load(f)

    new_version = _next_version_id(registry)
    new_model_file = f"model_{new_version}.pkl"
    new_metadata_file = f"model_{new_version}_metadata.json"

    shutil.copyfile(CANDIDATE_MODEL_PATH, MODELS_DIR / new_model_file)

    # Don't blindly copy the candidate's metadata file -- its internal
    # "version" field still says "candidate" (see ml/retraining_graph.py),
    # which would make GET /model-info show the wrong version to a
    # customer-service rep after promotion. Patch it to the real version id.
    candidate_metadata["version"] = new_version
    candidate_metadata["status"] = "production"
    with open(MODELS_DIR / new_metadata_file, "w") as f:
        json.dump(candidate_metadata, f, indent=2)

    now = datetime.now(timezone.utc).isoformat()
    previous_active = registry["active_version"]
    if previous_active in registry["versions"]:
        registry["versions"][previous_active]["status"] = "archived"

    registry["versions"][new_version] = {
        "model_file": new_model_file,
        "metadata_file": new_metadata_file,
        "status": "production",
        "activated_at": now,
        "metrics": {k: candidate_metadata.get(k) for k in _METRIC_KEYS},
        "source": "retraining_workflow",
    }
    registry["active_version"] = new_version
    registry["history"].append(new_version)
    _write_registry(registry)

    return {
        "active_version": new_version,
        "previous_version": previous_active,
        "version_info": registry["versions"][new_version],
    }


def _rollback_impl(to_version: str | None = None) -> dict[str, Any]:
    """
    Reverts the active version to `to_version` if given, otherwise to
    whichever version was active immediately before the current one (per
    registry["history"]). Never deletes a model file -- rollback and a
    later roll-forward both just repoint `active_version`.
    """
    registry = load_registry()
    current_active = registry["active_version"]

    if to_version is None:
        previous_candidates = [v for v in reversed(registry["history"]) if v != current_active]
        if not previous_candidates:
            raise VersioningError(
                f"No previous version to roll back to from '{current_active}' "
                "(it's the only version in history)."
            )
        to_version = previous_candidates[0]

    if to_version not in registry["versions"]:
        raise VersioningError(
            f"Unknown version '{to_version}'. Known versions: {list(registry['versions'])}"
        )
    if to_version == current_active:
        raise VersioningError(f"'{to_version}' is already the active version.")

    registry["versions"][current_active]["status"] = "rolled_back"
    registry["versions"][to_version]["status"] = "production"
    registry["active_version"] = to_version
    registry["history"].append(to_version)
    _write_registry(registry)

    return {
        "active_version": to_version,
        "previous_version": current_active,
        "version_info": registry["versions"][to_version],
    }


def promote_candidate() -> dict[str, Any]:
    """Public entrypoint -- holds _registry_lock around the full
    read-modify-write sequence so two concurrent promotions (within this
    process) can't race and silently lose one of them. See the module
    docstring's concurrency note for what this does and doesn't cover."""
    with _registry_lock:
        return _promote_candidate_impl()


def rollback(to_version: str | None = None) -> dict[str, Any]:
    """Public entrypoint -- same locking as promote_candidate() above."""
    with _registry_lock:
        return _rollback_impl(to_version=to_version)
