from fastapi.testclient import TestClient
import backend.service as svc

test_payloads = {
    "age": 25,
    "income": 50000,
    "home_ownership": "rent",
    "employment_length": 5,
    "loan_amount": 10000,
    "def_on_file": 0,
    "loan_intent": "personal",
}


def test_return_503_when_model_missing(client: TestClient, monkeypatch):
    def fail_load_model():
        raise Exception("Model file not found")

    monkeypatch.setattr(svc.mlflow.xgboost, "load_model", fail_load_model)
    response = client.get("/ready")
    assert response.status_code == 503


def test_ready_with_model(client: TestClient, monkeypatch, low_risk_model):
    monkeypatch.setattr(svc, "_loaded_model", low_risk_model)
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_denied_approved(client: TestClient, monkeypatch, low_risk_model, high_risk_model):
    # Low risk model should approve
    monkeypatch.setattr(svc, "_loaded_model", low_risk_model)
    response = client.post("/score", json=test_payloads).json()
    assert response["approved"] is True

    # High risk model should deny
    monkeypatch.setattr(svc, "_loaded_model", high_risk_model)
    response = client.post("/score", json=test_payloads).json()
    assert response["approved"] is False
