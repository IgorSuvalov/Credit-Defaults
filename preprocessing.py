# Import the necessary libraries
import os
import numpy as np
import pandas as pd
import opendatasets as od
import random
import re
import matplotlib.pyplot as plt

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

csv_path = os.path.join("credit-risk-dataset", "credit_risk_dataset.csv")
df = pd.read_csv(csv_path)
df = df.drop(columns=["loan_grade", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"])


# Map yes/no to 0/1
def yn_to01(s):
    m = s.astype(str).str.strip().str.lower()
    return m.map({"y": 1, "n": 0}).astype(float)


df["cb_person_default_on_file"] = yn_to01(df["cb_person_default_on_file"])

def normalize_intent_one(v: str) -> str:
    v = str(v).strip().upper().replace(" ", "")
    return v

# Homeowner encoding
def hom_own(x):
    mapping = {"other": 0, "rent": 1, "mortgage": 2, "own": 3}
    if isinstance(x, pd.Series):
        m = x.astype(str).str.strip().str.lower()
        return m.map(mapping).astype(float)
    else:
        return float(mapping.get(str(x).strip().lower(), float("nan")))


df["person_home_ownership"] = hom_own(df["person_home_ownership"])

# one-hot encode loan_intent into new columns (drop_first avoids dummy trap)
df = pd.get_dummies(df, columns=["loan_intent"], drop_first=True, dtype=int)

df = df.dropna()
