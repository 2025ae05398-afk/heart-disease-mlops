from pathlib import Path
import json
import logging
import time
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH, META_PATH = ROOT / "artifacts/model.joblib", ROOT / "artifacts/metadata.json"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")
REQUESTS = Counter("api_requests_total", "Total API requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_duration_seconds", "Request duration", ["path"])
app = FastAPI(title="Heart Disease Educational Prediction API", version="1.0.0",
              description="Educational only; not medical advice.")
model, metadata = None, None


def load_artifacts():
    global model, metadata
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Artifacts missing; run `python -m src.train` first.")
    model, metadata = joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


@app.on_event("startup")
def startup():
    if MODEL_PATH.exists() and META_PATH.exists():
        load_artifacts()
    else:
        log.warning("Model artifacts missing; /predict will return 503 until trained.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start, status = time.perf_counter(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        LATENCY.labels(request.url.path).observe(elapsed)
        log.info("%s %s status=%s duration_ms=%.2f", request.method, request.url.path,
                 status, elapsed * 1000)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: dict[str, Any]):
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    features = metadata["feature_names"]
    missing = [name for name in features if name not in payload]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    try:
        row = pd.DataFrame([{name: payload[name] for name in features}], columns=features)
        prediction = int(model.predict(row)[0])
        probs = model.predict_proba(row)[0].tolist()
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}") from exc
    return {"prediction": prediction, "label": "presence" if prediction else "absence",
            "confidence": float(max(probs)),
            "class_probabilities": {str(i): float(p) for i, p in enumerate(probs)},
            "disclaimer": "Educational model output only; not a diagnosis."}
