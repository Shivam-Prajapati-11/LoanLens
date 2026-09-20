# ---------------------------------------------------------------------------
# Loan Approval Prediction - production image (Phase 10)
#
# Build:  docker build -t loan-approval-prediction .
# Run:    docker run -p 8000:8000 loan-approval-prediction
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so Docker can cache this layer between builds.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code, trained model and frontend.
COPY app/ ./app/
COPY static/ ./static/
COPY models/ ./models/
COPY src/ ./src/

# Model training writes here / the API reads plots from here when retrained.
RUN mkdir -p plots

EXPOSE 8000

# Render injects $PORT at runtime; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
