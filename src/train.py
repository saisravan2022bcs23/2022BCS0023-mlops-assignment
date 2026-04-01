import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import joblib
import json
import sys

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# ARGUMENTS
model_type = sys.argv[1] if len(sys.argv) > 1 else "lr"
use_feature_selection = sys.argv[2] == "fs" if len(sys.argv) > 2 else False

# Feature Selection
if use_feature_selection:
    X = X.iloc[:, :10]  # first 10 features

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model choice
if model_type == "lr":
    model = LogisticRegression(max_iter=1000)
else:
    model = RandomForestClassifier()

# MLflow
mlflow.set_experiment("2022BCS0023_experiment")

with mlflow.start_run():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("model", model_type)
    mlflow.log_param("feature_selection", use_feature_selection)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    joblib.dump(model, "model.pkl")

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "name": "Sravan",
        "roll_no": "2022BCS0023"
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    mlflow.log_artifact("metrics.json")