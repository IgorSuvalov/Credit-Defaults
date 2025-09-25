import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from preprocessing import df

data = df
X_df = data.drop(columns=["loan_status"])
feature_cols = X_df.columns.tolist()
X = X_df.values
y = data["loan_status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="saga")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Rejected: {y_pred.mean() * 100:.0f}%")
print(f"Accuracy: {precision * 100:.0f}%")
print(f"Recall: {recall * 100:.0f}%")

joblib.dump({"model": model, "feature_cols": feature_cols}, "model.pkl")