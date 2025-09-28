import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from backend.my_preprocessing import preprocess

project_root = Path(__file__).parent.parent
csv_path = project_root / "credit-risk-dataset" / "credit_risk_dataset.csv"
df = pd.read_csv(csv_path)
data = preprocess(df)


X_df = data.drop(columns=["loan_status"]).copy()
y = data["loan_status"].astype(int).values

feature_cols = X_df.columns.tolist()
print("Features used for training:", feature_cols)



X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="saga")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Rejected: {y_pred.mean() * 100:.0f}%")
print("Accuracy:", f"{accuracy_score(y_test, y_pred) * 100:.0f}%")
print(f"Precision: {precision * 100:.0f}%")
print(f"Recall: {recall * 100:.0f}%")
print(f"f1: {f1_score(y_test, y_pred) * 100:.0f}%")

joblib.dump(model, "./backend/baseline.pkl")
joblib.dump(feature_cols, "./backend/feature_cols.pkl")

