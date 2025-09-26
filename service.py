import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class ClientData(BaseModel):
    age: int
    income: int
    home_ownership: str
    employment_length: float
    loan_amount: int
    def_on_file: float


app = FastAPI()


# Load artifacts
model = joblib.load("model.pkl")
feature_cols = joblib.load("feature_cols.pkl")  # must be the 6 columns above


def hom_own(x):
    mapping = {"other": 0, "rent": 1, "mortgage": 2, "own": 3}
    key = str(x).strip().lower()
    if key not in mapping:
        raise ValueError("home_ownership must be one of: other, rent, mortgage, own")
    return float(mapping[key])


@app.post("/score")
def score(data: ClientData):
    # build a row aligned to training-time feature order
    row = {
        "person_age": data.age,
        "person_income": data.income,
        "person_home_ownership": hom_own(data.home_ownership),
        "person_emp_length": data.employment_length,
        "loan_amnt": data.loan_amount,
        "cb_person_default_on_file": data.def_on_file,
    }
    X = [[row[c] for c in feature_cols]]  # shape (1, 6)
    yhat = model.predict(X)[0].item()  # assuming 1 = “deny”
    approved = not yhat
    return {"approved": bool(approved)}


