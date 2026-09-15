import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank_transactions.csv"
MODEL_PATH = BASE_DIR / "models" / "bank_transaction_classifier.pkl"
OUTPUT_PATH = BASE_DIR / "outputs" / "feature_importance.png"

df = pd.read_csv(DATA_PATH)

features = [
    "amount",
    "transaction_hour",
    "is_weekend",
    "account_balance",
    "merchant_type",
    "transaction_channel"
]
target = "category"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Bank Transaction Classification Results")
print("-" * 45)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

os.makedirs(MODEL_PATH.parent, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

importance = pd.Series(
    model.feature_importances_, index=features
).sort_values()

importance.plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
plt.savefig(OUTPUT_PATH)
plt.close()
print(f"Chart saved to: {OUTPUT_PATH}")
