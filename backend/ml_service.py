import numpy as np
import mlflow
from fastapi import HTTPException
from .schemas import ClientData

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
        raise HTTPException(status_code=422, detail=f"home_ownership must be one of: other, rent, mortgage, own")
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


def predict_proba(data: ClientData):
    model = get_model_or_503()
    row = {
        "person_age": data.age,
        "person_income": data.income,
        "person_home_ownership": hom_own(data.home_ownership.value),
        "person_emp_length": data.employment_length,
        "loan_amnt": data.loan_amount,
        "cb_person_default_on_file": data.def_on_file,
    }

    intent_key = data.loan_intent.value.strip().upper()

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
    return proba_default
