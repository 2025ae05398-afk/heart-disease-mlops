from fastapi.testclient import TestClient
import src.api as api


def test_health_without_model():
    old = api.model, api.metadata
    api.model, api.metadata = None, None
    try:
        assert TestClient(api.app).get("/health").json()["model_loaded"] is False
    finally:
        api.model, api.metadata = old


def test_predict_without_model_returns_503():
    old = api.model, api.metadata
    api.model, api.metadata = None, None
    try:
        assert TestClient(api.app).post("/predict", json={"age": 50}).status_code == 503
    finally:
        api.model, api.metadata = old
