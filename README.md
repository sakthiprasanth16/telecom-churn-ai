# 📞 Telecom Customer Churn Prediction System

A complete production ML system built to answer a specific take-home interview question (reproduced below, word for word) about building a churn-prediction system for a telecom company with 10 million customers. Every challenge in that question is implemented and **measured**, not just described — this README shows the actual numbers from real testing, not just the theory.

## Live Demo

**Frontend:** https://telecom-churn-frontend.onrender.com
**Backend:** https://telecom-churn-backend-59qw.onrender.com

---

## 📋 The Assignment This Solves

> *"You are building a machine learning system to predict customer churn for a telecom company. The company has 10 million customers, and the system must make predictions for customer service representatives in real-time. The system must be reliable, fair, and maintainable in production."* — Xplore IT Corp

The assignment listed 10 challenges the system must handle. Here they are, word for word, each followed by exactly how this project answers it — and where something isn't fully solved yet, that's said plainly, with a concrete plan instead of a vague excuse.

### 1. Data Quality Issues
> *The data contains missing values, outliers, inconsistencies, and errors. Data quality varies across different regions and time periods. You need to handle these issues automatically.*

**Answer:** Cleaning and validation happen in `notebooks/01_data_cleaning_and_eda.ipynb` before anything reaches training: `TotalCharges` (stored as text in the raw CSV, with blank strings for brand-new customers) is coerced to numeric and imputed; categorical values are standardized; the feature pipeline in `ml/preprocess.py` is the *same* code used for both training and live inference, so a request can never be preprocessed differently than the data the model was trained on — a common, subtle source of production bugs this design avoids by construction.

**Honest gap:** cleaning today is notebook-driven, run once. A real production system ingesting 10M customers' data continuously would need this promoted to an automated validation layer (e.g. Great Expectations or Pandera) that runs on every new data batch and rejects/flags bad records automatically, rather than a human running a notebook.

### 2. Model Drift
> *The model's performance degrades over time. A model trained 6 months ago performs worse than when it was first deployed. You need to detect when performance drops and take action.*

**Answer, honestly:** partially solved, and here's exactly why. `GET /fairness` and `GET /drift` both work correctly today and prove the *mechanism*, but true model-drift detection means watching recall/precision/F1 *decay* over time — and that requires real, ground-truth churn outcomes to compare predictions against. Every prediction already has an `actual_outcome` field reserved for exactly this (see `backend/schemas.py`), but it's always `None` right now, since there's no live CRM feeding real outcomes back in yet.

**What would close this gap:** wire in a feedback loop — 30–90 days after a prediction, record whether that customer actually churned — then compute rolling recall/precision on that labeled subset weekly and alert if it drops more than, say, 5% from the training baseline. This is a data-integration problem, not a modeling one; the architecture already has the field waiting for it.

### 3. Data Drift
> *The characteristics of incoming data change over time. Customer behavior changes with seasons, economic conditions, and company policies. The distribution of features shifts, making old models less effective.*

**Answer:** Fully implemented in `ml/drift.py` using **PSI (Population Stability Index)** — the same metric used in real production credit-risk monitoring, not an invented threshold. `GET /drift` compares the last *N* logged predictions (default 500, configurable 30–5000) against the training baseline stored in `models/drift_reference_v1.json`:

| PSI value | Status | What happens |
|---|---|---|
| < 0.10 | `no_drift` | Nothing — logged for visibility only |
| 0.10 – 0.25 | `moderate_drift` | Flagged on the dashboard, doesn't trigger retraining automatically |
| ≥ 0.25 | `significant_drift` | Makes the system *eligible* to retrain (see #5) |
| (fewer than 30 samples) | `insufficient_data` | Explicitly refuses to guess on too small a sample, rather than reporting a false "no drift" |

### 4. Concept Drift
> *The relationship between features and churn changes. What predicted churn 6 months ago might not predict churn today. You need to detect and adapt to these changes.*

**Answer, honestly: not built.** This is the one genuine gap in the project, and it shares its root cause with #2 above — concept drift (the feature→label *relationship* changing) can only be detected against real labeled outcomes over time, exactly like model drift. Once the feedback loop from #2 exists, a real concept-drift detector (e.g. ADWIN or DDM running on the labeled prediction stream) is the standard next step, or simpler: comparing the retrained model's coefficients against the previous version's over successive retrains to see if the *relationships* themselves are shifting, not just the accuracy.

### 5. Automated Retraining
> *When drift is detected, the system should automatically retrain. However, retraining must be done carefully to avoid deploying worse models.*

**Answer:** `POST /retrain` runs a LangGraph-orchestrated workflow (`ml/retraining_graph.py`) with four explicit stages — `check_drift → load_training_data → train_candidate → evaluate_candidate`. It only proceeds past the first stage if PSI ≥ 0.25 *or* `force=True` is explicitly passed (useful for demoing the workflow without waiting for real drift to accumulate). Critically: **a trained candidate is never automatically promoted to production.** It's written to `models/candidate_model.pkl` and just sits there — a human must explicitly call `POST /promote-candidate` before it ever serves a real customer. That's the "carefully" the requirement asks for, implemented as a hard gate, not a policy someone has to remember to follow.

### 6. Versioning and Rollback
> *You need to track all model versions, data versions, and feature versions. If a new model performs poorly, you must be able to quickly rollback to the previous version.*

**Answer:** `backend/versioning.py` maintains `models/version_registry.json` as the single source of truth for which model is currently active, with full history of every version, its training metrics, and when it was promoted. `POST /rollback` reverts to any previous version and reloads the live predictor instantly — no restart needed. Concurrent-safety was a real bug found and fixed during development: promotions/rollbacks use a lock plus an atomic file write, specifically because two simultaneous admin actions on the naive first version could corrupt the registry.

### 7. Explainability
> *Customer service representatives need to understand why the system predicts a customer will churn. You need to provide explanations that are understandable to non-technical people.*

**Answer:** Since the production model is Logistic Regression, `backend/predictor.py` uses an exact, cheap, and honest method — no approximation library needed: **standardized coefficient × scaled feature value** gives each feature's real contribution to *this specific* prediction. The top contributing factors are converted into plain-English sentences like *"Month-to-month contract increases churn risk"* rather than raw numbers — built specifically for a non-technical customer-service rep, not a data scientist.

### 8. Fairness
> *The model must not discriminate against customers based on protected attributes like age, gender, or location. You need to audit for fairness and mitigate biases.*

**Answer:** `ml/fairness.py` audits the two protected-attribute-adjacent columns the dataset actually has: `gender`, and `SeniorCitizen` (the closest available proxy for age — there's no raw age or location field in this dataset, which is itself disclosed rather than silently assumed). It computes disparate-impact ratios between groups and can *suggest* per-group threshold adjustments to equalize outcomes. `GET /fairness` exposes both a training-time audit (against real labels) and a live proxy audit (selection-rate only, since live traffic has no ground truth yet).

**Honest gap:** the suggested threshold mitigation is a **recommendation surfaced on the dashboard, not automatically applied** to the live serving path — a human decision point was left in deliberately rather than silently changing who gets flagged as high-risk.

### 9. Real-time Predictions
> *The system must make predictions in real-time (sub-100ms latency) for customer service representatives. Batch predictions are not acceptable.*

**Answer, honestly:** not achieved end-to-end on this project's free-tier infrastructure — and that's explained in full, with real measured numbers, in [Measured Performance](#-measured-performance-the-honest-numbers) below. The short version: **the model's own inference is genuinely fast — 12.59ms measured in production** — the gap to a guaranteed sub-100ms comes from network distance and single-process concurrency, both infrastructure problems with known, specific fixes (see [Scaling to 10 Million Customers](#-scaling-this-prototype-to-10-million-customers)), not a flaw in the model or the code design.

### 10. Business Impact
> *The system must be evaluated on business metrics (customers retained, revenue saved) not just technical metrics (accuracy, AUC).*

**Answer:** `ml/business_impact.py` (`GET /business-impact`) simulates retained customers and revenue saved using the model's real confusion matrix from the last training run, combined with three clearly-labeled, adjustable business assumptions: a **30% retention success rate** for contacted at-risk customers, a **$15 cost per intervention contact**, and a **12-month customer lifetime** used for revenue-saved projections. The response is explicitly tagged as *"SIMULATED, not measured"* in its own output — never presented as a verified result, exactly because these are configurable assumptions layered on real model performance, not a measured outcome.

---

## 🏗️ Architecture

```
                        ┌─────────────────────────┐
                        │   Streamlit Frontend    │
                        │  (Prediction Form  +    │
                        │   Monitoring Page)      │
                        └────────────┬────────────┘
                                     │ HTTP (JSON)
                                     ▼
                        ┌────────────────────────┐
                        │    FastAPI Backend     │
                        │ (model loaded ONCE at  │
                        │  startup, not per      │
                        │  request)              │
                        └────────────┬───────────┘
                                     │
       ┌───────────────┬─────────────┼─────────────┬────────────────────┐
       ▼               ▼             ▼             ▼                    ▼
┌───────────┐   ┌─────────────┐ ┌──────────┐ ┌──────────────┐  ┌────────────────┐
│ Predictor │   │ Drift Check │ │ Fairness │ │  Business    │  │  Retraining    │
│ (loaded   │   │ (PSI vs.    │ │  Audit   │ │  Impact      │  │  Workflow      │
│  model +  │   │  baseline)  │ │          │ │  Simulation  │  │  (LangGraph,   │
│  explain) │   │             │ │          │ │              │  │  admin-only)   │
└─────┬─────┘   └──────┬──────┘ └────┬─────┘ └──────┬───────┘  └───────┬────────┘
      │                │             │              │                  │
      └────────────────┴─────────────┴──────┬───────┴──────────────────┘
                                            │ (background tasks --
                                            │  logging never blocks
                                            │  a live prediction)
                                            ▼
                              ┌───────────────────────────────┐
                              │      MongoDB Atlas            │
                              │  predictions / drift_reports/ │
                              │  fairness_snapshots /         │
                              │  retraining_runs /            │
                              │  version_events               │
                              │  (audit trail ONLY -- never a │
                              │   serving dependency)         │
                              └───────────────────────────────┘

                              ┌────────────────────────────────┐
                              │   Local Model Files            │
                              │  models/model_v*.pkl           │
                              │  models/version_registry.json  │
                              │  (source of truth for which    │
                              │   model version is active)     │
                              │  models/drift_reference_v1.json│
                              └────────────────────────────────┘
```

**Two design decisions worth calling out explicitly:**
- **MongoDB is never a serving dependency.** If Atlas is unreachable, every endpoint still returns a valid response using an in-memory fallback (`mongomock`) — logging can fail without ever blocking a prediction a customer-service rep is waiting on.
- **The candidate model from retraining never auto-promotes.** `POST /retrain` only ever writes `candidate_model.pkl`. A human must explicitly call `POST /promote-candidate` before it serves a single real customer — this is what requirement #5's "carefully" actually means in code, not just in prose.

---

## 🔄 How a Prediction Actually Flows

1. **A customer-service rep enters a customer's details** into the Streamlit form and submits.
2. **The frontend calls `POST /predict`** on the backend with that customer's data as JSON.
3. **Pydantic validates the request shape** (`backend/schemas.py`) — malformed input is rejected before it ever reaches the model.
4. **The same preprocessing used in training** (`ml/preprocess.py`) transforms the raw fields into the model's expected feature vector.
5. **The already-loaded model scores it** — no disk read, no reload, no external API call. This is what keeps inference itself fast (measured at 12.59ms median in production — see below).
6. **The top contributing factors are computed** (coefficient × scaled value) and turned into a plain-English explanation for the rep.
7. **The response returns immediately** with the churn probability, risk label, and explanation.
8. **Separately, in the background:** the prediction is logged to MongoDB for future drift/fairness analysis — this happens *after* the response is already on its way back, so a slow or failed database write can never add latency to what the rep sees.

Drift detection and retraining are **completely separate, admin-only paths** (`GET /drift`, `POST /retrain`) — nothing about them runs during, or can slow down, a live customer-facing prediction.

---

## 📈 Measured Performance: The Honest Numbers

Every number below is a real measurement taken with `tests/load_test.py` against the actual deployed Render service and the local machine used for development — not an estimate.

| Test | Reported latency (pure model inference) | Wall-clock (full request, incl. network) |
|---|---|---|
| Local machine, 1 request at a time | 84.06ms | 170.53ms |
| Local machine, 5 concurrent requests | 395.47ms | 591.13ms |
| **Render (production), 1 request at a time** | **12.59ms** | 307.29ms |
| Render (production), 5 concurrent requests | 101.71ms | 610.44ms |

**The core finding:** the model itself is genuinely fast — **12.59ms median inference time, measured in actual production**, well under the 100ms target. The gap to a guaranteed sub-100ms *end-to-end* comes from two separate, independently confirmed causes, isolated by testing 1-worker vs. 5-worker load specifically:

1. **Network distance.** 307.29ms wall-clock minus 12.59ms actual inference leaves **~295ms of pure network/TLS/routing overhead** — the physical distance between the test location and Render's free-tier server region. There is no code fix for this; only a geographically closer region (a paid-tier option) would reduce it.
2. **Single-process concurrency.** Going from 1 → 5 simultaneous requests pushed *inference time itself* from 12.59ms → 101.71ms on Render (and 84ms → 395ms locally) — proof that requests were queuing behind each other inside one process, not a database or network problem. This one **is** fixable in software: multiple `uvicorn` workers or multiple Render replicas behind a load balancer would let concurrent requests run in parallel instead of serializing.

**What this means, stated plainly:** the model design achieves sub-100ms. Guaranteeing it end-to-end for many simultaneous real users at production scale requires infrastructure most free tiers don't offer — the same infrastructure covered in the next section.

---

## 🏭 Scaling This Prototype to 10 Million Customers

The honest framing: **this prototype proves every required capability works correctly at small scale using free tools.** Below is exactly which pieces would be swapped for paid/managed equivalents to handle 10 million customers and sustained real-time load — and why each swap is the one that matters. The *design* doesn't change; the *infrastructure* does.

| Prototype piece today | Bottleneck at 10M scale | Production swap |
|---|---|---|
| Single Render free-tier FastAPI instance | One process, one CPU, spins down when idle | Multiple stateless replicas behind a load balancer (Render autoscaling, or AWS ECS/EKS) — model still loaded once per replica at startup, same pattern, horizontally repeated |
| MongoDB Atlas free M0 | Shared, tiny, no sharding | Dedicated Atlas cluster (M30+), sharded by customer ID for write-heavy prediction logging |
| Direct synchronous Mongo write per prediction (background task) | Fine at low volume; a write-storm at millions/day risks backpressure | Predictions published to a queue (Kafka / AWS Kinesis), consumed asynchronously into a warehouse (Snowflake/BigQuery) — fully decouples logging from prediction latency |
| `GET /drift`, `/fairness`, `/business-impact` compute live on each request | Expensive/slow if hit constantly at scale | Precompute on a schedule (Airflow/Prefect job); endpoints just read the latest precomputed report |
| Local JSON file version registry | Explicitly single-process-safe only (documented, not hidden) | A real model registry (MLflow or SageMaker Model Registry) supporting concurrent multi-instance reads, plus CI/CD-gated promotion |
| Manual `POST /retrain` trigger | Fine for demonstrating the workflow | Same LangGraph workflow, scheduled automatically once the precomputed drift job crosses threshold, running on real labeled outcomes fed back from a CRM integration |
| No labeled-feedback loop (`actual_outcome` always `None`) | The actual blocker behind #2 and #4 above | CRM/billing integration feeding real churn/retention outcomes back ~30-90 days after each prediction — enables real model-drift and concept-drift detection |
| Streamlit frontend | Great for an internal prototype; not built for thousands of concurrent enterprise users | A React/Next.js frontend behind a CDN, calling the same FastAPI backend unchanged |
| Manual notebook-based data cleaning | Not repeatable safely on continuously arriving data | Automated schema/quality validation (Great Expectations or Pandera) on every new data batch |
| Public-internet hop between frontend and backend | Adds real round-trip latency (confirmed above) | Both services on the same private network / same region, removing that hop entirely |

---

## 🖥️ How the App Works, Page by Page

| Page | What it shows |
|---|---|
| **Home (Prediction Form)** | Enter a customer's details, get a churn probability, a risk label, and a plain-English explanation of the top contributing factors |
| **Monitoring** | Live system health (API/model/database status), prediction volume, average latency, drift status, fairness audit results, business-impact simulation, and full model version history |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- A free [MongoDB Atlas](https://cloud.mongodb.com) account (or run with the built-in `mongomock` fallback for local dev)
- Docker Desktop (optional, only needed for the containerized paths)

### Option A — Run directly with Python

```bash
git clone https://github.com/sakthiprasanth16/telecom-churn-ai.git
cd telecom-churn-ai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

copy .env.example .env
# edit .env: set MONGODB_URI to your real Atlas connection string
# (or leave unset to use the built-in mongomock fallback for local dev)
```

**Start the backend:**
```bash
uvicorn backend.main:app --reload
```
Confirm it's healthy: `http://127.0.0.1:8000/docs`

**Start the frontend** (a separate terminal):
```bash
streamlit run frontend/app.py
```
Open `http://localhost:8501`.

### Option B — Run everything with Docker Compose

```bash
docker compose build backend
docker compose build frontend
docker compose up
```
Frontend: `http://localhost:8501` · Backend docs: `http://localhost:8000/docs`. Both containers are wired together automatically over Docker's internal network — `BACKEND_URL` resolves to the backend container by name, no manual configuration needed.

### Option C — Individual Docker containers (no Compose)

```bash
docker build -f Dockerfile.backend -t telecom-churn-backend .
docker build -f Dockerfile.frontend -t telecom-churn-frontend .

docker run --rm --name backend-test -p 8000:8000 telecom-churn-backend
docker run --rm --name frontend-test -p 8501:8501 -e BACKEND_URL=http://host.docker.internal:8000 telecom-churn-frontend
```

### Option D — Deploy to Render (what the live demo above actually runs on)

A `render.yaml` blueprint is included at the project root. Go to `dashboard.render.com/blueprints` → New Blueprint Instance → connect this repo. Render detects `render.yaml` automatically and creates both services. After the backend deploys, copy its live URL and set it as the `BACKEND_URL` environment variable on the frontend service (this one field must be set by hand — Render's blueprint spec can't auto-wire a full `https://` URL between services).

---

## 🧪 Testing

**67 pytest tests** (`tests/test_api.py`, `tests/test_streamlit_apps.py`) covering the API endpoints, prediction validation, drift/fairness/business-impact logic, and the Streamlit UI itself — run with:
```bash
pytest
```

**Load/latency testing** (`tests/load_test.py`) has two modes:
```bash
# In-process, no network -- measures pure code + inference speed
python tests/load_test.py --requests 100 --workers 5

# Real HTTP against any running server, local or deployed
python tests/load_test.py --url https://telecom-churn-backend-59qw.onrender.com --requests 100 --workers 5
```
The `--url` mode is what produced every number in [Measured Performance](#-measured-performance-the-honest-numbers) above — it's the only mode that measures real network time, not just code speed.

---

## 🔌 API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Model/DB liveness for monitoring |
| GET | `/model-info` | Active model's metadata and metrics |
| POST | `/predict` | Real-time churn prediction for one customer |
| GET | `/predictions` | Recent prediction history |
| GET | `/metrics` | Aggregate monitoring metrics |
| GET | `/drift` | PSI-based data drift report vs. training baseline |
| POST | `/retrain` | Trigger the LangGraph retraining workflow |
| GET | `/model-versions` | Full version registry (active + history) |
| POST | `/promote-candidate` | Promote the last-trained candidate to production |
| POST | `/rollback` | Revert to a previous model version |
| GET | `/fairness` | Training-time fairness audit + live selection-rate proxy |
| GET | `/business-impact` | ROI-aware business impact simulation |

Full interactive docs (try-it-out for every endpoint): `/docs`.

---

## 📁 Project Structure

```
telecom-churn-ai/
├── backend/            # FastAPI app, schemas, predictor, versioning, database
├── ml/                 # preprocessing, drift, fairness, training, business impact, retraining graph
├── frontend/           # Streamlit app + monitoring dashboard
├── models/             # model artifacts, drift baseline, version registry
├── data/                # raw and processed datasets
├── notebooks/          # data cleaning, feature engineering, model training
├── tests/              # pytest suite + load_test.py (in-process AND real-HTTP modes)
├── Dockerfile.backend / .frontend
├── docker-compose.yml
├── render.yaml          # Render blueprint (one-click deploy for both services)
├── .env.example
└── requirements.txt
```

---

## ⚠️ Known Limitations (stated honestly, not hidden)

- **No real labeled-feedback loop yet** — `actual_outcome` is logged on every prediction but always `None`. This is the single root cause behind Model Drift (#2) and Concept Drift (#4) both being incomplete.
- **Sub-100ms is not guaranteed end-to-end** — the model itself measures 12.59ms in production; the gap is network distance and single-process concurrency, both explained with real numbers above.
- **Fairness mitigation is a recommendation, not automatically applied** to the live serving path — a deliberate human-decision point, not an oversight.
- **Business impact numbers are a simulation** built on real confusion-matrix data plus configurable assumptions (30% retention success rate, $15/contact, 12-month customer lifetime) — explicitly labeled as such in the API response itself.
- **The version registry assumes a single backend worker process.** A genuinely multi-process/multi-replica deployment needs a shared, database-backed lock instead of the current file-based one.
- **Free-tier hosting (Render) has cold-start/spin-down behavior** — the first request after idle time is slow; this is platform behavior, not an application bug.

---

## 🛠️ Common Issues (real ones hit during this project's own deployment)

- **`ModuleNotFoundError: No module named 'mongomock'`** — this dependency is used by `backend/database.py` itself as the default fallback (not just in tests), but was missing from `backend/requirements.txt`. Fixed by adding it there.
- **`0.0.0.0:8000` doesn't open in a browser** — that address means "listening on all interfaces" inside the container/server logs, not a real browsable address. Use `localhost:8000` instead.
- **`docker compose build` crashing Docker Desktop's engine** — building both images in parallel can exhaust resources on some machines. Building sequentially (`docker compose build backend` then `docker compose build frontend`) avoids it.
- **`BACKEND_URL` via Render's blueprint `fromService` returns a bare hostname, no scheme** — since the frontend code does `f"{BACKEND_URL}/predict"`, a missing `https://` breaks every request. Set this one variable by hand after the backend's first deploy instead.
- **`python tests/load_test.py` failing with `No module named 'backend'`** — running a script directly only adds its own folder to Python's path, not the project root. Fixed by inserting the project root into `sys.path` at the top of the script.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn (Python) |
| ML | scikit-learn 1.8.0 (Logistic Regression), pandas, numpy, joblib |
| Orchestration | LangGraph (retraining workflow) |
| Database | MongoDB Atlas (pymongo; mongomock for local/tests) |
| Frontend | Streamlit |
| Testing | pytest, FastAPI TestClient, Streamlit AppTest |
| Containerization | Docker, Docker Compose |
| Deployment | Render (Docker-based Web Services) |
| Data source | [Telco Customer Churn (IBM sample)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), Kaggle |

---

## 👨‍💻 Author

**Sakthi Prasanth**
Production-oriented telecom customer churn prediction system
