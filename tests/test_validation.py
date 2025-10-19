from fastapi.testclient import TestClient
import backend.app as svc

test_payloads = {
    "age": 25,
    "income": 50000,
    "home_ownership": "rent",
    "employment_length": 5,
    "loan_amount": 10000,
    "def_on_file": 0,
    "loan_intent": "personal",
}


def test_employment_less_than_age(client: TestClient, monkeypatch, low_risk_model):
    monkeypatch.setattr(svc, "_loaded_model", low_risk_model)
    payload = test_payloads.copy()
    payload["employment_length"] = 30  # Invalid: employment length greater than age
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_negative_income(client: TestClient, monkeypatch, low_risk_model):
    monkeypatch.setattr(svc, "_loaded_model", low_risk_model)
    payload = test_payloads.copy()
    payload["income"] = -5000  # Invalid: negative income
    response = client.post("/score", json=payload)
    assert response.status_code in (400, 422)


def test_invalid_home_ownership(client: TestClient, monkeypatch, low_risk_model):
    monkeypatch.setattr(svc, "_loaded_model", low_risk_model)
    payload = test_payloads.copy()
    payload["home_ownership"] = "castle"  # Invalid homeownership

    response = client.post("/score", json=payload)
    assert response.status_code in (400, 422)
