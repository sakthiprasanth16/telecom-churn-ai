"""
tests/load_test.py
-------------------
A threaded load/latency REPORT for the /predict endpoint, run separately
from the main test suite:

    python tests/load_test.py
    python tests/load_test.py --requests 500 --workers 20

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

IMPORTANT CAVEAT this script's own numbers can't rule out: it uses
FastAPI's TestClient (in-process, synchronous) rather than a real running
uvicorn server hit over the network from separate client processes/
connections. Python's GIL and TestClient's synchronous request handling
mean this does NOT measure true multi-connection network concurrency the
way a real production load test would -- it mainly measures how fast the
model inference + preprocessing pipeline itself is under repeated calls.
For a genuine "sub-100ms at scale, many simultaneous users" claim, run a
real tool (e.g. `locust -f locustfile.py --host http://localhost:8000`)
against an actually-running `uvicorn backend.main:app` process. This
script is a fast, dependency-free sanity check, not that benchmark.
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi.testclient import TestClient

from backend.main import app

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


def _one_request(client: TestClient) -> tuple[float, float, int]:
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


def run_load_test(n_requests: int, n_workers: int) -> None:
    client = TestClient(app)

    # One warm-up request -- the project's own notes flag the first
    # prediction after startup as slower (cold-start); excluding it keeps
    # the report representative of steady-state performance.
    _one_request(client)

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
    print(
        "NOTE: reported latency_ms is model-inference-only (see predictor.py); "
        "wall-clock includes Pydantic validation, TestClient overhead, and "
        "thread scheduling. Neither measures real network/multi-connection "
        "load -- see this file's module docstring for what a genuine "
        "production load test would need instead."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=200, help="Total number of /predict calls to make")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent worker threads")
    args = parser.parse_args()
    run_load_test(n_requests=args.requests, n_workers=args.workers)
