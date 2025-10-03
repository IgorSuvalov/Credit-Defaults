import numpy as np
import mlflow
import os, time
from fastapi import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum


class Home(str, Enum):
    other = "other"
    rent = "rent"
    mortgage = "mortgage"
    own = "own"


class Intent(str, Enum):
    debtconsolidation = "debtconsolidation"
    personal = "personal"
    education = "education"
    medical = "medical"
    venture = "venture"
    homeimprovement = "homeimprovement"


class ClientData(BaseModel):
    age: int = Field(ge=18, le=120)
    income: int = Field(ge=0, le=100000000)
    home_ownership: str = Home
    employment_length: float = Field(ge=0, le=110)
    loan_amount: int = Field(ge=1, le=1000000000)
    def_on_file: float = Field(ge=0, le=1)
    loan_intent: str = Intent


app = FastAPI()

origins = ["http://localhost:5173", "http://localhost:8004", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

feature_cols = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_amnt',
                'cb_person_default_on_file', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
                'loan_intent_HOMEIMPROVEMENT',
                'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE']

model_name = "XGBoost with SMOTETomek"
model_uri = f"models:/{model_name}@champion"

_loaded_model = None


def hom_own(x):
    mapping = {"other": 0, "rent": 1, "mortgage": 2, "own": 3}
    key = str(x).strip().lower()
    if key not in mapping:
        raise ValueError("home_ownership must be one of: other, rent, mortgage, own")
    return float(mapping[key])


def get_model_or_503():
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model
    try:
        _loaded_model = mlflow.xgboost.load_model(model_uri)
        return _loaded_model
    except Exception as e:
        print(f"Model loading error: {e}")
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")


@app.get("/live")
def live():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    _ = get_model_or_503()
    return {"status": "ready", "model_uri": model_uri}


@app.post("/score")
def score(data: ClientData):
    model = get_model_or_503()
    row = {
        "person_age": data.age,
        "person_income": data.income,
        "person_home_ownership": hom_own(data.home_ownership),
        "person_emp_length": data.employment_length,
        "loan_amnt": data.loan_amount,
        "cb_person_default_on_file": data.def_on_file,
    }

    intent_key = str(data.loan_intent).strip().upper()

    X_row = []
    for col in feature_cols:
        if col in row:
            X_row.append(row[col])
        elif col.startswith("loan_intent_"):
            intent_value = col.replace("loan_intent_", "")
            X_row.append(1.0 if intent_value == intent_key else 0.0)
        else:
            X_row.append(0.0)

    X = np.asarray([X_row], dtype=float)

    proba_default = float(model.predict_proba(X)[0, 1])

    approved = bool(proba_default < 0.5)
    return {
        "approved": approved,
    }
