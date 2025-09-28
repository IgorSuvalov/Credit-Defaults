import os
import joblib
import pandas as pd
from backend.my_preprocessing import preprocess
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_root = Path(__file__).parent.parent
csv_path = project_root / "credit-risk-dataset" / "credit_risk_dataset.csv"
df = pd.read_csv(csv_path)
data = preprocess(df)

X_df = data.drop(columns=["loan_status"]).copy()
y = data["loan_status"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)


class CreditRiskNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


model = CreditRiskNN(input_dim=X_train.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = (model(X_test) > 0.5).float()
            acc = (test_preds.eq(y_test).sum().item()) / len(y_test)
#        print(f"Epoch {epoch + 1}/{epochs}, Loss={loss.item():.4f}, Test Acc={acc:.3f}")


def predict(model, X, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs = model(X)
        if probs.shape[1] == 1:
            return (probs > threshold).float()
        else:
            return torch.argmax(probs, dim=1)


y_pred = predict(model, X_test, threshold=0.5).cpu().numpy()
y_test = y_test.cpu().numpy()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Rejected: {y_pred.mean() * 100:.0f}%")
print("Accuracy:", f"{accuracy_score(y_test, y_pred) * 100:.0f}%")
print(f"Precision: {precision * 100:.0f}%")
print(f"Recall: {recall * 100:.0f}%")
print(f"f1 score: {f1_score(y_test, y_pred) * 100:.0f}%")

torch.save(model.state_dict(), "./backend/nn_model.pth")
