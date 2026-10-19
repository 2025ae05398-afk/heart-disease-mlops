"""FastAPI inference service with request logs and Prometheus metrics."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "model.joblib"))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_disease_api")

REQUESTS = Counter("api_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "HTTP request latency", ["endpoint"])

FEATURES = {
    "age": (float, 1, 120), "sex": (int, 0, 1), "cp": (int, 1, 4),
    "trestbps": (float, 50, 300), "chol": (float, 50, 800), "fbs": (int, 0, 1),
    "restecg": (int, 0, 2), "thalach": (float, 40, 250), "exang": (int, 0, 1),
    "oldpeak": (float, 0, 10), "slope": (int, 1, 3), "ca": (float, 0, 4),
    "thal": (float, 3, 7),
}
class PatientFeatures(BaseModel):
    age: float = Field(ge=1, le=120, examples=[58])
    sex: Literal[0, 1] = Field(examples=[1])
    cp: int = Field(ge=1, le=4, examples=[4])
    trestbps: float = Field(ge=50, le=300, examples=[130])
    chol: float = Field(ge=50, le=800, examples=[250])
    fbs: Literal[0, 1] = Field(examples=[0])
    restecg: int = Field(ge=0, le=2, examples=[0])
    thalach: float = Field(ge=40, le=250, examples=[140])
    exang: Literal[0, 1] = Field(examples=[0])
    oldpeak: float = Field(ge=0, le=10, examples=[2.0])
    slope: int = Field(ge=1, le=3, examples=[2])
    ca: float = Field(ge=0, le=4, examples=[0])
    thal: float = Field(ge=3, le=7, examples=[3])

app = FastAPI(
    title="Heart Disease Risk Classification API",
    version="1.0.0",
    description="Educational demonstration only; not for diagnosis or treatment decisions.",
)
model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("model_loaded path=%s", MODEL_PATH)
    else:
        logger.warning("model_missing path=%s; run python -m src.data && python -m src.train", MODEL_PATH)

@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    status = "500"
    endpoint = request.url.path
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(request.method, endpoint, status).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        logger.info("request method=%s path=%s status=%s latency_seconds=%.5f",
                    request.method, endpoint, status, elapsed)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    return {"service": "heart-disease-mlops", "docs": "/docs", "health": "/health"}

@app.post("/predict")
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it before serving.")
    row = pd.DataFrame([patient.model_dump()])
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    confidence = float(max(probabilities))
    positive_probability = float(probabilities[1])
    logger.info("prediction_completed predicted_class=%s positive_probability=%.4f", prediction, positive_probability)
    return {
        "prediction": prediction,
        "label": "disease_present" if prediction == 1 else "disease_not_detected",
        "confidence": round(confidence, 4),
        "probability_disease_present": round(positive_probability, 4),
        "disclaimer": "Educational estimate only; not a diagnosis. Consult a qualified clinician.",
    }
