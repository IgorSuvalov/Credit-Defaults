import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from preprocessing import hom_own

# ---- Config ----
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # set via env if you like


bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["feature_cols"]  # authoritative order


class ClientData(BaseModel):
    age: int
    income: int
    home_ownership: str
    employment_length: float
    loan_amount: int
    def_on_file: float
    intent: str

    # normalize strings to uppercase with no surrounding spaces
    @field_validator("home_ownership", "intent")
    @classmethod
    def normalize_upper(cls, v: str) -> str:
        return v.strip().upper()


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(d: ClientData):
    # Manual, deterministic feature vector matching training order
    intent = d.intent.strip().upper().replace(" ", "")
    intent_EDU  = 1 if intent == "EDUCATION"       else 0
    intent_HI   = 1 if intent == "HOMEIMPROVEMENT" else 0
    intent_MED  = 1 if intent == "MEDICAL"         else 0
    intent_PER  = 1 if intent == "PERSONAL"        else 0
    intent_VEN  = 1 if intent == "VENTURE"         else 0
    # baseline: DEBTCONSOLIDATION (all zeros)

    row = pd.DataFrame([{
        "person_age": d.age,
        "person_income": d.income,
        "person_home_ownership_enc": hom_own(d.home_ownership),
        "person_emp_length": d.employment_length,
        "loan_amnt": d.loan_amount,
        "cb_person_default_on_file": float(d.def_on_file),
        "loan_intent_EDUCATION": intent_EDU,
        "loan_intent_HOMEIMPROVEMENT": intent_HI,
        "loan_intent_MEDICAL": intent_MED,
        "loan_intent_PERSONAL": intent_PER,
        "loan_intent_VENTURE": intent_VEN,
    }], columns=FEATURES)  # enforce same column order

    proba_default = float(model.predict_proba(row.values)[:, 1][0])
    approved = proba_default < THRESHOLD
    return {"prob_default": proba_default, "approved": approved, "threshold": THRESHOLD}