import sys
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bank_transaction_classifier.pkl"

if not MODEL_PATH.exists():
    print("Model not found! Run 'python train_model.py' first.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

transaction = pd.DataFrame([{
    "amount": 1250,
    "transaction_hour": 18,
    "is_weekend": 0,
    "account_balance": 45000,
    "merchant_type": 1,
    "transaction_channel": 1
}])

prediction = model.predict(transaction)[0]
probability = model.predict_proba(transaction).max()

print("Predicted transaction category:", prediction)
print(f"Prediction confidence: {probability:.2%}")
