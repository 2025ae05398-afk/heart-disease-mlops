from fastapi.testclient import TestClient
import src.api as api

client = TestClient(api.app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_prediction_requires_model(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    payload = {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 140, "exang": 0,
        "oldpeak": 2.0, "slope": 2, "ca": 0, "thal": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
