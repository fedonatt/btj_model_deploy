import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

load_dotenv(dotenv_path=".dev.env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
REGISTERED_MODEL_PREFIX = os.getenv("REGISTERED_MODEL_PREFIX")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


iris = load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#coba 3 models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

best_model = None
best_score = -1
best_model_name = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name}_pipeline") as run:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"{model_name} Results")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "accuracy": acc,
            "recall": rec,
            "f1_score": f1
        })

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=f"{REGISTERED_MODEL_PREFIX}_{model_name}_Pipeline"
        )
