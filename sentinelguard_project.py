"""
SENTINELGUARD
Banking Fraud Intelligence & Explainable Risk Scoring Platform
Educational prototype using synthetic data only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score,
    roc_auc_score
)

# 1. Generate synthetic transaction records
np.random.seed(42)
n = 10000

data = pd.DataFrame({
    "amount": np.random.uniform(10, 50000, n),
    "hour": np.random.randint(0, 24, n),
    "transaction_type": np.random.randint(0, 4, n),
    "merchant_category": np.random.randint(0, 8, n),
    "country": np.random.randint(0, 5, n),
    "device_risk_score": np.random.uniform(0, 1, n),
    "ip_risk_score": np.random.uniform(0, 1, n)
})

fraud_probability = (
    0.02
    + 0.25 * (data["amount"] > 30000)
    + 0.25 * (data["device_risk_score"] > 0.8)
    + 0.25 * (data["ip_risk_score"] > 0.8)
    + 0.15 * (data["hour"].isin([0, 1, 2, 3]))
)
fraud_probability = np.clip(fraud_probability, 0, 0.9)
data["fraud"] = (
    np.random.random(n) < fraud_probability
).astype(int)

data.to_csv("banking_fraud_dataset.csv", index=False)

# 2. Prepare features and target
X = data.drop("fraud", axis=1)
y = data["fraud"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_probability))
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("SentinelGuard - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=180)
plt.show()

# 6. Explain model feature importance
importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=True)

importance.plot(kind="barh", figsize=(8, 5))
plt.title("SentinelGuard - Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=180)
plt.show()

# 7. Save trained model
joblib.dump(model, "sentinelguard_model.pkl")
print("Model saved successfully.")

# 8. Example transaction prediction
def predict_fraud(
    amount, hour, transaction_type, merchant_category,
    country, device_risk_score, ip_risk_score
):
    transaction = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "country": country,
        "device_risk_score": device_risk_score,
        "ip_risk_score": ip_risk_score
    }])

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    print("\\nSENTINELGUARD ANALYSIS")
    print("Estimated model fraud score:", round(probability * 100, 2), "%")
    print("Risk status:", "HIGH RISK" if prediction == 1 else "LOW RISK")

predict_fraud(45000, 2, 1, 3, 2, 0.95, 0.90)
