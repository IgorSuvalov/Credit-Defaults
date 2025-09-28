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
    loan_intent: str


app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and feature columns
model = joblib.load("backend/model.pkl")
feature_cols = joblib.load("backend/feature_cols.pkl")


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

    intent_key = str(data.loan_intent).strip().lower()

    X_row = []
    for col in row:
        if col in row:
            X_row.append(row[col])
        elif col.startswith("loan_intent_"):
            intent_value = col.replace("loan_intent_", "")
            X_row.append(1.0 if intent_value == intent_key else 0.0)
        else:
            X_row.append(0.0)

    yhat = model.predict([X_row])[0].item()  # 1 is default predicted, 0 is approved
    approved = not yhat
    return {"approved": approved}
