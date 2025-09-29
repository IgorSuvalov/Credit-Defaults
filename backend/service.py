from pathlib import Path
import numpy as np
import joblib
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

class ClientData(BaseModel):
    age: int
    income: int
    home_ownership: str
    employment_length: float
    loan_amount: int
    def_on_file: float
    loan_intent: str


app = FastAPI()

origins = ["http://localhost:5173", "http://localhost:8004", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and feature columns

MODEL_JSON = Path("backend/xgboost.json")
FEATURE_COLS_PKL = Path("backend/feature_cols.pkl")

# Stuff below is result of debugging paths and pkl/json models not being fitted...

if not MODEL_JSON.exists():
    raise RuntimeError(f"Missing model file: {MODEL_JSON}")
if not FEATURE_COLS_PKL.exists():
    raise RuntimeError(f"Missing feature columns file: {FEATURE_COLS_PKL}")

feature_cols = joblib.load(FEATURE_COLS_PKL)

print("feature_cols".upper())

# Try to load as sklearn wrapper first; if that fails, fall back to Booster
_MODEL_MODE = "sklearn"
_model_clf = None
_model_booster = None
try:
    _model_clf = xgb.XGBClassifier()
    _model_clf.load_model(MODEL_JSON)  # loads fitted weights into wrapper
except Exception:
    _MODEL_MODE = "booster"
    _model_clf = None
    _model_booster = xgb.Booster()
    _model_booster.load_model(MODEL_JSON)


def hom_own(x):
    mapping = {"other": 0, "rent": 1, "mortgage": 2, "own": 3}
    key = str(x).strip().lower()
    if key not in mapping:
        raise ValueError("home_ownership must be one of: other, rent, mortgage, own")
    return float(mapping[key])


@app.post("/score")
def score(data: ClientData):

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

    # predict probability of default
    if _MODEL_MODE == "sklearn":
        proba_default = float(_model_clf.predict_proba(X)[0, 1])
    else:
        dmat = xgb.DMatrix(X)
        proba_default = float(_model_booster.predict(dmat)[0])

    approved = bool(proba_default < 0.5)
    return {
        "approved": approved,
        #   "prob_default": proba_default,  # helpful for UI thresholds
        #   "model_mode": _MODEL_MODE,      # 'sklearn' or 'booster' for debugging
    }
