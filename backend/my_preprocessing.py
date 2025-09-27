# Import the necessary libraries
import numpy as np
import pandas as pd
import random

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)


# Homeowner encoding
def hom_own(x):
    mapping = {"other": 0, "rent": 1, "mortgage": 2, "own": 3}
    if isinstance(x, pd.Series):
        m = x.astype(str).str.strip().str.lower()
        return m.map(mapping).astype(float)
    else:
        return float(mapping.get(str(x).strip().lower(), float("nan")))


# Map yes/no to 0/1
def yn_to01(s):
    m = s.astype(str).str.strip().str.lower()
    return m.map({"y": 1, "n": 0}).astype(float)


FEATURE_COLS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_amnt",
    "cb_person_default_on_file",
]
TARGET = "loan_status"

def preprocess(df):
    keep = FEATURE_COLS + [TARGET]
    df = df[keep].copy()

    df["cb_person_default_on_file"] = yn_to01(df["cb_person_default_on_file"])
    df["person_home_ownership"] = hom_own(df["person_home_ownership"])

    df = df.dropna()

    return df
