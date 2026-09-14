"""
tests/load_test.py
-------------------
A threaded load/latency REPORT for the /predict endpoint, run separately
from the main test suite:

    python tests/load_test.py
    python tests/load_test.py --requests 500 --workers 20

    # Against a real deployed server (e.g. Render), over actual HTTP:
    python tests/load_test.py --url https://telecom-churn-backend.onrender.com --requests 200 --workers 10

This is deliberately NOT a pytest test with hard pass/fail latency
assertions. Absolute latency thresholds are environment-dependent (CPU
speed, whether the model was just cold-loaded, what else is running on
the machine), so hard-asserting "must be under 100ms" here would be
flaky, not meaningful -- see tests/test_api.py's existing
test_predict_latency_is_fast for the same reasoning applied more gently.
This script instead prints p50/p95/p99 latency and throughput so a human
can judge them against the project spec's sub-100ms target, the same way
a real load-testing tool (locust, k6, JMeter) would report results for a
human to interpret rather than auto-pass/fail.

TWO MODES:

1. In-process (default, no --url given): uses FastAPI's TestClient, calling
   backend.main.app directly in the same process. Fast, dependency-free
   (no real network), but per the caveat below does NOT measure real
   network/multi-connection concurrency -- mainly measures how fast the
   model inference + preprocessing pipeline itself is.

2. Real HTTP (--url given): fires actual HTTP requests over the network at
   a running server -- your local `uvicorn backend.main:app`, or a real
   deployed URL like your Render service. This is the mode that can
   actually answer "is this sub-100ms in production" -- it includes real
   network time, FastAPI's request handling, and (on Render specifically)
   whatever the free tier's cold-start/sleep behavior does to the first
   request. Requires the `requests` package (pip install requests
   --break-system-packages if you don't already have it) but does NOT
   require the backend's ML dependencies to be installed locally, since
   it never imports backend.main in this mode.

IMPORTANT CAVEAT for mode 1's numbers specifically: TestClient is
in-process and synchronous, so Python's GIL means it does NOT measure true
multi-connection network concurrency the way mode 2 (or a real tool like
locust/k6/JMeter) does. Use --url whenever you want a number you can
actually defend as "measured against the real thing."
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

# Running this file directly (`python tests/load_test.py`) only puts the
# tests/ folder itself on sys.path, not the project root -- so `import
# backend.main` below would fail with "No module named 'backend'" even
# though the file structure is correct. Inserting the project root here
# makes `python tests/load_test.py` work exactly as documented above,
# without requiring `python -m tests.load_test` instead. Only needed for
# in-process mode (--url mode never imports backend.main at all).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


class _PostClient(Protocol):
    """Either FastAPI's TestClient or _RealHTTPClient below -- both expose
    a .post(path, json=...) returning something with .status_code/.json()."""

    def post(self, path: str, json: dict): ...


class _RealHTTPClient:
    """Thin wrapper so real HTTP mode has the exact same .post() shape as
    FastAPI's TestClient, letting _one_request() stay identical either way."""

    def __init__(self, base_url: str):
        try:
            import requests
        except ImportError as e:
            raise SystemExit(
                "The 'requests' package is required for --url mode. "
                "Install it with: pip install requests --break-system-packages"
            ) from e
        self._session = requests.Session()
        self._base_url = base_url.rstrip("/")

    def post(self, path: str, json: dict):
        return self._session.post(f"{self._base_url}{path}", json=json, timeout=30)


def _one_request(client: _PostClient) -> tuple[float, float, int]:
    """Returns (wall_clock_ms, reported_latency_ms, status_code)."""
    start = time.perf_counter()
    response = client.post("/predict", json=VALID_PAYLOAD)
    wall_clock_ms = (time.perf_counter() - start) * 1000
    reported_latency_ms = response.json().get("latency_ms", float("nan")) if response.status_code == 200 else float("nan")
    return wall_clock_ms, reported_latency_ms, response.status_code


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(int(len(values) * pct), len(values) - 1)
    return values[idx]


def run_load_test(n_requests: int, n_workers: int, url: str | None = None) -> None:
    if url:
        print(f"Mode: REAL HTTP against {url}")
        client: _PostClient = _RealHTTPClient(url)
    else:
        print("Mode: in-process (TestClient) -- see module docstring's caveat")
        # Imported here, not at module level, so --url mode never needs the
        # backend's ML dependencies (sklearn, pandas, etc.) installed locally.
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

    # One warm-up request -- excluded from the report. This matters even
    # more in --url mode against Render's free tier specifically: a
    # service that's been idle spins down and can take 30-60s to wake on
    # the very first request. Warming up first keeps the reported
    # percentiles representative of steady-state performance, not a
    # one-time cold-start outlier.
    print("Warming up (first request may be slow, especially on Render free tier)...")
    warmup_wall_ms, _, warmup_status = _one_request(client)
    print(f"Warm-up request: {warmup_wall_ms:.0f}ms, status={warmup_status}\n")

    wall_clock_times: list[float] = []
    reported_latencies: list[float] = []
    error_count = 0

    overall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_one_request, client) for _ in range(n_requests)]
        for future in as_completed(futures):
            wall_clock_ms, reported_latency_ms, status_code = future.result()
            if status_code != 200:
                error_count += 1
                continue
            wall_clock_times.append(wall_clock_ms)
            reported_latencies.append(reported_latency_ms)
    overall_elapsed_s = time.perf_counter() - overall_start

    print(f"\n{'=' * 60}")
    print(f"Load test: {n_requests} requests, {n_workers} worker threads")
    print(f"{'=' * 60}")
    print(f"Errors: {error_count}/{n_requests}")
    print(f"Wall-clock total: {overall_elapsed_s:.2f}s")
    if not wall_clock_times:
        print("No successful requests -- check the URL/server and try again.")
        return
    print(f"Throughput: {len(wall_clock_times) / overall_elapsed_s:.1f} requests/sec")
    print()
    print(f"{'Metric':<25} {'Wall-clock (ms)':>18} {'Reported latency_ms':>22}")
    for label, pct in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        print(
            f"{label:<25} {_percentile(wall_clock_times, pct):>18.2f} "
            f"{_percentile(reported_latencies, pct):>22.2f}"
        )
    print(f"{'mean':<25} {statistics.mean(wall_clock_times):>18.2f} {statistics.mean(reported_latencies):>22.2f}")
    print(f"{'max':<25} {max(wall_clock_times):>18.2f} {max(reported_latencies):>22.2f}")
    print()
    if url:
        print(
            "NOTE: reported latency_ms is model-inference-only (see predictor.py); "
            "wall-clock here is the FULL real round trip -- DNS, TLS, network, "
            "FastAPI request handling, Pydantic validation, and inference -- "
            "since this was real HTTP against a live server. This IS the number "
            "to judge against the project spec's sub-100ms target."
        )
    else:
        print(
            "NOTE: reported latency_ms is model-inference-only (see predictor.py); "
            "wall-clock includes Pydantic validation, TestClient overhead, and "
            "thread scheduling -- but NOT real network time, since this was "
            "in-process. Re-run with --url against a real running server for a "
            "number that reflects actual production latency."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=200, help="Total number of /predict calls to make")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent worker threads")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help=(
            "Base URL of a real running server to test over actual HTTP "
            "(e.g. https://telecom-churn-backend.onrender.com or "
            "http://localhost:8000). If omitted, tests in-process via "
            "FastAPI's TestClient instead (no real network involved)."
        ),
    )
    args = parser.parse_args()
    run_load_test(n_requests=args.requests, n_workers=args.workers, url=args.url)
