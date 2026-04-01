import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
import json
import sys
import numpy as np

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# ARGUMENTS
model_type = sys.argv[1] if len(sys.argv) > 1 else "lr"
use_feature_selection = sys.argv[2] == "fs" if len(sys.argv) > 2 else False

# Feature Selection
if use_feature_selection:
    selected_features = X.columns[:10]
    X = X[selected_features]
else:
    selected_features = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model selection
if model_type == "lr":
    model = LogisticRegression(max_iter=2000)
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

# MLflow
mlflow.set_experiment("2022BCS0023_experiment")

with mlflow.start_run():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Logging
    mlflow.log_param("model", model_type)
    mlflow.log_param("feature_selection", use_feature_selection)
    mlflow.log_param("num_features", len(selected_features))

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Save model + scaler + features
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    feature_info = {
        "features": list(selected_features)
    }

    with open("features.json", "w") as f:
        json.dump(feature_info, f)

    # Metrics JSON (MANDATORY)
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "name": "Sravan",
        "roll_no": "2022BCS0023"
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # Log artifacts
    mlflow.log_artifact("metrics.json")
    mlflow.log_artifact("features.json")

print("✅ Training complete")
print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")