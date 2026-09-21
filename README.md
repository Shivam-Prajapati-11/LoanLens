# LoanLens

LoanLens predicts whether a loan application will be **Approved** or **Rejected**.
Enter the applicant details in a 3-step form, the trained model scores them, and
every submission is saved so you can review it later on a history dashboard.

- **Live app:** https://loan-approval-prediction-self.vercel.app
- **API health check:** https://loan-approval-prediction-qsnu.onrender.com/health
- **Model:** Random Forest - F1 **0.987**, accuracy **0.984** (4,269 real applications)

> Demo project - this is a machine-learning prediction, not a real lending decision.

---

## What the project does

| Part | Description |
|---|---|
| **Machine learning** | Trains and compares 3 classifiers on 11 applicant features and saves the best one (Random Forest) as a single reusable pipeline |
| **Web app** | 3-step form with live validation, animated result card, and a sortable history dashboard |
| **API** | FastAPI service that serves the model (`POST /predict`) with auto-generated docs at `/docs` |
| **Storage** | Every prediction is logged to SQLite through SQLAlchemy |
| **Delivery** | Dockerfile, Render blueprint, and GitHub Actions CI that retrains and tests on every push |

---

## Quick start

```bash
git clone https://github.com/Shivam-Prajapati-11/LoanLens.git
cd LoanLens
pip install -r requirements.txt

python -m src.train        # trains the model + writes plots/ and models/
python -m tests.test_api   # 13 API tests (uses a throwaway database)
uvicorn app.main:app --reload
```

Then open:

- http://127.0.0.1:8000/ - prediction form
- http://127.0.0.1:8000/history - saved predictions
- http://127.0.0.1:8000/docs - interactive API

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check (used by Docker / Render) |
| `GET` | `/metadata` | Model name, metrics, and valid feature values |
| `POST` | `/predict` | Predict Approved / Rejected for one application |
| `GET` | `/api/history` | All stored predictions, newest first |
| `GET` | `/history` | History dashboard page |
| `GET` | `/docs` | Swagger UI |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_name": "Ravi Kumar",
    "No_of_dependents": 2,
    "Education": "Graduate",
    "Self_employed": "No",
    "Income_annum": 5000000,
    "Loan_amount": 20000000,
    "Loan_term": 10,
    "Cibil_score": 750,
    "Residential_assets_value": 10000000,
    "Commercial_assets_value": 5000000,
    "Luxury_assets_value": 3000000,
    "Bank_asset_value": 5000000
  }'
```

```json
{"result": "Approved", "probability": 0.9233}
```

Invalid input is rejected with HTTP `422` and the reason.

---

## How the model works

1. **Clean** - strip whitespace from headers and values, drop duplicates, enforce types.
2. **Explore** - EDA shows `cibil_score` is by far the strongest signal (`|r| = 0.77`).
3. **Preprocess** - one `ColumnTransformer`: median impute + scale the numeric features,
   most-frequent impute + one-hot encode the categorical ones. It is fitted on the
   training split only, so no information leaks from the test set.
4. **Train and compare** - three algorithms with `class_weight="balanced"`; the best
   F1-score wins.
5. **Serve** - the whole pipeline (preprocessing + model) is saved with `joblib`, so the
   API applies exactly the same steps at prediction time.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **0.984** | **0.985** | **0.989** | **0.987** |
| Decision Tree | 0.975 | 0.987 | 0.974 | 0.980 |
| Logistic Regression | 0.924 | 0.955 | 0.921 | 0.938 |

Metrics come from a 20% stratified test split (`random_state=42`). The 8 EDA charts and
3 confusion matrices are in `plots/`, and `notebooks/loan_analysis.ipynb` reproduces the
same numbers.

---

## Project structure

```
app/main.py            FastAPI app: routes, CORS, static pages
app/schemas.py         Pydantic request/response models (input validation)
app/database.py        SQLAlchemy engine/session, init_db(), writable-path fallback
app/models.py          ORM table `prediction_logs`
src/train.py           Training: EDA -> preprocessing -> train -> evaluate -> export
static/index.html      Prediction form UI
static/history.html    History dashboard UI
static/script.js       Form logic + POST /predict call
static/history.js      Dashboard logic + GET /api/history call
static/config.js       Frontend API base URL
static/style.css       Styling
tests/test_api.py      API tests (TestClient + temp database)
data/loan_data.csv     4,269 loan applications
models/                loan_model.pkl, model_metadata.json, metrics.json
notebooks/             Notebook that reproduces the training metrics
Dockerfile             Container image
render.yaml            Render blueprint (health check: /health)
.github/workflows/     CI: retrain model + run tests on push
```

---

## Docker

```bash
docker build -t loanlens .
docker run -p 8000:8000 loanlens
```

---

## Deployment

The app runs on a single host (Render serves both the UI and the API), or split in two
(UI on Vercel, API on Render). Three settings control it:

| Setting | Where | Value |
|---|---|---|
| `ALLOWED_ORIGINS` | API (Render env var) | Frontend origin(s) allowed to call the API, comma separated |
| `window.LOANLENS_API_BASE` | Frontend (`static/config.js`) | API URL, or `""` when the same host serves both |
| `DATABASE_URL` | API (Render env var) | Optional SQL database (e.g. Render PostgreSQL); defaults to SQLite |

* **Render:** deploy `render.yaml` as a Blueprint. It is kept in sync with the live
  service name (`loan-approval-prediction-qsnu`) because Blueprints match services by
  name - renaming it would create a second service instead of updating this one.
* **Vercel:** redeploys automatically on every push to `main`.
* Free tiers have no persistent disk, so the SQLite history resets on each deploy or
  restart. Set `DATABASE_URL` if you want it to survive.
* On read-only hosts (Vercel serverless) the app falls back to a temp-directory
  database instead of failing writes with an HTTP 500.

---

## Notes

* 11 input features: demographics, income, loan details, asset values, and CIBIL score.
* `prediction_logs` stores the applicant name, every input, the prediction, the
  probability, and a UTC timestamp.
* Tests never touch real history - they point `DATABASE_URL` at a temp file.
