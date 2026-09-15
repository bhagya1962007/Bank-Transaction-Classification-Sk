# How to Run: Bank Transaction Classification using Scikit-learn

This project classifies bank transactions into categories such as groceries, bills, shopping, travel, salary, and entertainment.

---

## 1. Prerequisites

- **Python**: Python 3.9+ (Python 3.14 tested)
- **Terminal**: Windows PowerShell, Command Prompt, or VS Code integrated terminal
- **Dependencies**: `pandas`, `scikit-learn`, `numpy`, `joblib`, `matplotlib`

---

## 2. Open Terminal & Navigate to Project Directory

Open PowerShell or Command Prompt, then change directory to the project folder:

```powershell
cd "c:\Users\user\Desktop\jj college\Bank_Transaction_Classification_Sklearn"
```

---

## 3. Install Dependencies

Install required packages:

```powershell
pip install -r requirements.txt
```

> **Windows Note**: If `python` or `pip` opens the Microsoft Store, specify the direct Python path:
> ```powershell
> & "C:\Users\user\AppData\Local\Python\bin\python.exe" -m pip install -r requirements.txt
> ```

---

## 4. Step 1: Train the Machine Learning Model

Execute the training script to train and save the model:

```powershell
python train_model.py
```
*(or `py train_model.py` / `& "C:\Users\user\AppData\Local\Python\bin\python.exe" train_model.py`)*

### What this does:
1. Loads the dataset from `data/bank_transactions.csv`.
2. Trains a **RandomForestClassifier** model predicting `category`.
3. Evaluates model performance (Accuracy Score, Classification Report).
4. Saves the trained model (`.pkl` file).
5. Generates and saves visualization plots/charts (e.g., feature importance or segment visualizations).

---

## 5. Step 2: Run Predictions

Run the prediction script to test predictions:

```powershell
python predict.py
```

### Sample Prediction:
The script runs a sample test case directly from preconfigured values:

```python
transaction = pd.DataFrame([{
    "amount": 1250,
    "transaction_hour": 18,
    "is_weekend": 0,
    "account_balance": 45000,
    "merchant_type": 1,
    "transaction_channel": 1
}])
```

To customize input values, edit [predict.py](predict.py) directly and run `python predict.py` again.

---

## 6. Directory Structure

```text
Bank_Transaction_Classification_Sklearn/
├── data/
│   └── bank_transactions.csv
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md
```
