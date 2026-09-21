# LoanLens - Loan Approval Prediction System

A comprehensive full-stack machine learning application that predicts whether
a loan application will be **Approved** or **Rejected**. The project integrates
an end-to-end ML pipeline (Pandas + scikit-learn) with a FastAPI backend,
an SQLite prediction log, and a premium animated web frontend.

---

## Project Overview

An **end-to-end machine learning project** covering the complete modeling lifecycle:
data cleaning, exploratory data analysis, feature engineering, multi-model training
and comparison, evaluation, serialization, REST API serving, database logging,
containerization, and CI/CD. The ML pipeline is the core of the project - the web
app and database exist to serve and monitor the model. Trained on **4,269 real
loan application records** with 11 features (demographic, financial, and credit-score data).

---

## Key Technical Features

- **End-to-End ML Pipeline** - Median/mode imputation, standardization, and one-hot
  encoding wrapped in a scikit-learn `Pipeline` + `ColumnTransformer` to prevent data leakage.
- **Model Comparison** - Logistic Regression, Decision Tree, and Random Forest trained
  with `class_weight="balanced"`; best model auto-selected on F1-score.

  | Model | Accuracy | Precision | Recall | F1 |
  |---|---|---|---|---|
  | Random Forest | **0.984** | **0.985** | **0.989** | **0.987** |
  | Decision Tree | 0.975 | 0.987 | 0.974 | 0.980 |
  | Logistic Regression | 0.924 | 0.955 | 0.921 | 0.938 |

- **Key Insight** - `cibil_score` dominates the outcome (correlation ~ 0.77), explaining
  why the non-linear Random Forest outperforms Logistic Regression.
- **REST API** - FastAPI service with `POST /predict`, `GET /health`, `GET /metadata`,
  and automatic OpenAPI documentation (`/docs`) with Pydantic input validation (HTTP 422 on bad input).
- **Prediction Logging** - Every request is stored in SQLite via SQLAlchemy
  (applicant name, all 11 inputs, prediction, probability, UTC timestamp).
- **Analytics Dashboard** - `/history` page with sortable table and live stats
  (total submissions, approvals, rejections, approval rate).
- **Containerized & CI-Ready** - Dockerfile plus a GitHub Actions workflow
  that retrains the model and runs the test suite on every push.

---

## Exploratory Data Analysis

8 generated visualizations in `plots/`: CIBIL score distribution by approval status,
income and loan-amount distributions, categorical feature breakdowns, class balance,
loan-term analysis, correlation heatmap, and an income-vs-loan-amount scatter plot.

---

## Database Architecture

SQLite (via SQLAlchemy ORM) with automated schema creation:

- **prediction_logs** - one row per prediction: applicant name, all 11 input features,
  prediction, probability, and timestamp.
- **models/loan_model.pkl** - serialized pipeline (preprocessing + classifier) with
  `model_metadata.json` and `metrics.json` for auditability.

> `DATABASE_URL` is environment-driven - tests automatically use a throwaway
> database so real history is never polluted.

---

## File Structure & Logic

```
+-- src/train.py               # Full pipeline: EDA, preprocessing, training, evaluation, export
+-- app/main.py                # FastAPI app: /predict, /health, /metadata, /api/history
+-- app/schemas.py             # Pydantic request/response validation (incl. UTC timestamp fix)
+-- app/database.py            # SQLAlchemy engine, sessions, init_db()
+-- app/models.py              # ORM model for the prediction_logs table
+-- static/index.html|css|js   # 3-step animated form UI (progress bar, orbs, result reveal)
+-- static/history.html|js     # Sortable history dashboard with stat cards
+-- static/config.js           # Frontend API base URL (same origin vs. Vercel + Render split)
+-- tests/test_api.py          # 13 integration tests (TestClient, temp DB, CORS)
+-- notebooks/loan_analysis.ipynb  # Executed end-to-end; metrics match train.py exactly
+-- Dockerfile / .dockerignore # Container deployment
+-- render.yaml                # One-click Render blueprint (service `loanlens-api`)
+-- .github/workflows/ci.yml   # Retrain + test on push
```

---

## Deployment: Vercel (frontend) + Render (API)

The app can run in **one** place or be split in two. Both modes are supported by
the same code, and which one is active is decided by `ALLOWED_ORIGINS` (API side)
and `window.LOANLENS_API_BASE` (frontend side).

### Where each piece lives

| Piece | Same-origin (Render or Docker) | Split (Vercel UI + Render API) |
|---|---|---|
| HTML/CSS/JS | FastAPI serves `/`, `/history`, `/static/*` | Vercel serves the page |
| `/predict`, `/api/history` | Same host | Render Web Service |
| `window.LOANLENS_API_BASE` | `""` | `https://loanlens-api.onrender.com` |

### Render (API)

Live service: **https://loan-approval-prediction-qsnu.onrender.com**
(`GET /health` -> `{"status":"ok","model_name":"Random Forest","model_loaded":true}`)

* `render.yaml` names the service `loan-approval-prediction-qsnu` because Render
  **uniquified** the original `loan-approval-prediction` (that plain name belongs
  to the older *Loan-Approval-Prediction* static site). Blueprints match services
  by name, so keep this value in sync with the dashboard - renaming it would
  create a second service with a different URL instead of updating this one.
* Set `ALLOWED_ORIGINS` to the frontend origin(s), comma separated. The built-in
  default already allows `https://loan-approval-prediction-self.vercel.app`.
* Free plan has **no persistent disk**: the SQLite history resets on every
  deploy/restart (this is why `app/database.py` falls back to a temp directory
  instead of crashing). Uncomment `DATABASE_URL` + `databases:` in `render.yaml`
  for a durable database.

### Vercel (frontend)

Live site: **https://loan-approval-prediction-self.vercel.app**

1. `static/config.js` -> `window.LOANLENS_API_BASE = "https://loan-approval-prediction-qsnu.onrender.com";`
   (no trailing slash). Leave it `""` when Vercel hosts the API too.
2. Make sure the Vercel origin is in the API's `ALLOWED_ORIGINS`.

### Verifying the connection after a deploy

```bash
# 1. API alive on Render
curl https://loan-approval-prediction-qsnu.onrender.com/health

# 2. CORS preflight must answer 200 *and* echo the Vercel origin.
#    405 with no access-control-allow-origin header = the CORS build is not
#    deployed yet (browser will block every call and the UI looks "not connected").
curl -i -X OPTIONS https://loan-approval-prediction-qsnu.onrender.com/predict \
  -H "Origin: https://loan-approval-prediction-self.vercel.app" \
  -H "Access-Control-Request-Method: POST"

# 3. The frontend must ship the runtime config
curl -i https://loan-approval-prediction-self.vercel.app/static/config.js
```

All three must succeed **and the changes must be committed + pushed** - both
hosts deploy from `main`, so local edits alone never reach either URL.

### Why `POST /predict` used to return HTTP 500 on Vercel

Vercel (like every serverless host) mounts the deployment **read-only**, so
`db.commit()` into `<bundle>/predictions.db` raised
`sqlite3.OperationalError: attempt to write a readonly database`. GET routes were
fine because they only read, which made the API look half-connected. The
prediction itself never failed. `app/database.py` now probes the project
directory and falls back to a writable temp directory, and `POST /predict`
returns a descriptive JSON error instead of an opaque 500 if storage ever fails
again.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Libraries: `pip install -r requirements.txt` (scikit-learn, FastAPI, uvicorn, SQLAlchemy, pydantic)

### Installation
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/LoanLens.git
   cd LoanLens
   ```
2. **Train the Model** (generates `models/loan_model.pkl` and all plots):
   ```bash
   python -m src.train
   ```
3. **Run Tests** (optional, uses a temp database):
   ```bash
   python -m tests.test_api
   ```
4. **Launch the Platform:**
   ```bash
   uvicorn app.main:app --reload
   ```
5. Open `http://127.0.0.1:8000/` for the prediction form, `/history` for the
   analytics dashboard, and `/docs` for the interactive API.

### Docker
```bash
docker build -t loanlens .
docker run -p 8000:8000 loanlens
```

---

## Machine Learning Deep-Dive

The core of this project is the modeling workflow in `src/train.py`:

1. **Data Cleaning** - Strips hidden whitespace from headers and categorical values
   (a subtle real-world data quirk that silently produced an all-NaN target before
   being caught and fixed), drops duplicates, enforces type consistency.
2. **Exploratory Analysis** - Correlation analysis revealed `cibil_score` as the
   dominant signal (|r| = 0.77 vs 0.11 for loan term and < 0.02 for everything else),
   which guided feature treatment and model choice.
3. **Leakage-Safe Preprocessing** - A `ColumnTransformer` performs median imputation +
   `StandardScaler` on numeric features and most-frequent imputation + `OneHotEncoder`
   on categorical features, fitted strictly on training data inside the pipeline.
4. **Model Selection** - Three algorithms trained with stratified cross-validation and
   `class_weight="balanced"` to counter class imbalance; the winner is picked
   automatically on F1-score, not accuracy.
5. **Evaluation & Export** - Per-class precision/recall/F1 reports, confusion matrices,
   and the full fitted pipeline serialized with `joblib` so the served model applies
   identical preprocessing at inference time.
6. **Reproducibility** - `notebooks/loan_analysis.ipynb` re-derives the exact same
   metrics, and CI re-runs training on every push to catch drift.

---

## Future Roadmap (AI / ML Extensions)

- **Neural Network Benchmark** - Train an MLP (Keras/TensorFlow or PyTorch) with
  dropout, batch normalization, and early stopping; compare against the Random Forest
  on F1 and calibration, and serve the better model.
- **SHAP Explainability** - Per-applicant feature-importance attributions so every
  rejection comes with human-readable reasons (critical for credit decisions).
- **Probability Calibration** - Isotonic/Platt calibration so the reported approval
  probability is a true risk estimate, not just a ranking score.
- **Hyperparameter Optimization** - Automated tuning (Optuna / RandomizedSearchCV)
  with nested cross-validation instead of default estimator parameters.
- **Churn-Style Analytics** - Predict early repayment default from historical payment patterns.
- **What-If Simulator** - Sliders showing how much the CIBIL score must improve to
  flip a rejection to an approval.
- **PostgreSQL Migration** - Swap SQLite for PostgreSQL via the existing `DATABASE_URL`
  for multi-user deployments.
