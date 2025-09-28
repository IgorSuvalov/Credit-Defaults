import pandas as pd
import inspect
import xgboost as xgb
from backend.my_preprocessing import preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

project_root = Path(__file__).parent.parent
csv_path = project_root / "credit-risk-dataset" / "credit_risk_dataset.csv"
df = pd.read_csv(csv_path)
data = preprocess(df)

X_df = data.drop(columns=["loan_status"]).copy()
y = data["loan_status"].astype(int).values

bool_cols = X_df.select_dtypes(include=["bool"]).columns.tolist()
for c in bool_cols:
    X_df[c] = X_df[c].astype(int)
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X = X_df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0

model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="auc",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

early_cb = xgb.callback.EarlyStopping(rounds=50, save_best=True)
trained_with_classifier = False

try:
    sig = inspect.signature(model.fit)
    if "callbacks" in sig.parameters:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_cb],
            verbose=20
        )
        trained_with_classifier = True
    else:
        raise TypeError("fit() doesn't accept callbacks in this version")
except TypeError:

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 6,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
        "nthread": -1
    }

    evals = [(dval, "validation"), (dtrain, "train")]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=20
    )

if trained_with_classifier:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    booster = model.get_booster()
else:
    y_proba = bst.predict(dtest)
    y_pred = (y_proba >= 0.5).astype(int)
    booster = bst

print(f"Rejected: {y_pred.mean() * 100:.0f}%")
print("Accuracy:", f"{accuracy_score(y_test, y_pred) * 100:.0f}%")
print("Precision:", f"{precision_score(y_test, y_pred) * 100:.0f}%")
print("Recall:", f"{recall_score(y_test, y_pred) * 100:.0f}%")
print("f1 score:", f"{f1_score(y_test, y_pred) * 100:.0f}%")
