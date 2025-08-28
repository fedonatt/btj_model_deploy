import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv(dotenv_path=".dev.env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "Iris_LogisticRegression_Pipeline"  
MODEL_VERSION = 1        

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

X_new = [[5.0, 3.0, 1.6, 0.2]]
y_pred_new = model.predict(X_new)

print(f"Model: {MODEL_NAME} (version {MODEL_VERSION})")
print("Prediction:", y_pred_new)
