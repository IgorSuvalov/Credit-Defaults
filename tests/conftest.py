import types

import numpy as np
import pytest
from fastapi.testclient import TestClient
import backend.service as svc

@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.setattr(svc, "_loaded_model", None, raising=False)

@pytest.fixture
def client():
    return TestClient(svc.app)

@pytest.fixture
def low_risk_model():
    model = types.SimpleNamespace()
    model.predict_proba = lambda X: np.array([[0.9, 0.1]], dtype=float)  # Low risk
    return model

@pytest.fixture
def high_risk_model():
    model = types.SimpleNamespace()
    model.predict_proba = lambda X: np.array([[0.1, 0.9]], dtype=float)  # High risk
    return model
