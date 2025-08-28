import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from pathlib import Path
from api.predict.schemas import PredictionParams

IRIS_MODELS_PATH = Path("api/predict/models/iris_classifier.pkl")
SCALER_PATH      = Path("api/predict/models/iris_scaler.pkl")

map_species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

class Predict:
    _scaler = None
    _model  = None
    
    def __init__(self, params: PredictionParams):
        self.params = params

        if type(self)._scaler is None and SCALER_PATH.exists():
            type(self)._scaler = joblib.load(SCALER_PATH)
        if type(self)._model is None and IRIS_MODELS_PATH.exists():
            type(self)._model = joblib.load(IRIS_MODELS_PATH)

    def predict(self):
        try:
            pred_data = [[
                float(self.params.sepal_length),
                float(self.params.sepal_width),
                float(self.params.petal_length),
                float(self.params.petal_width),
            ]]
            
            X = type(self)._scaler.transform(pred_data)
            y = type(self)._model.predict(X).tolist()
            spesies = map_species.get(y[0], f"unknown({y[0]})")
            proba = type(self)._model.predict_proba(X).tolist()

            return {
                "message": "Prediction success",
                "result": y,
                "species": spesies,
                "probabilities": {
                    "setosa": proba[0][0],
                    "versicolor": proba[0][1],
                    "virginica": proba[0][2],
                }
            }
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}") from e

    def predict_v2(self):
        try:
            load_dotenv(dotenv_path=".dev.env")
            MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
            MODEL_NAME = "Iris_RandomForest_Pipeline"
            MODEL_VERSION = 5

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            model = mlflow.sklearn.load_model(model_uri)

            pred_data = pd.DataFrame([[
                float(self.params.sepal_length),
                float(self.params.sepal_width),
                float(self.params.petal_length),
                float(self.params.petal_width),
            ]], columns=[
                "sepal length (cm)", 
                "sepal width (cm)", 
                "petal length (cm)", 
                "petal width (cm)"
            ])

            y = model.predict(pred_data).tolist()
            spesies = map_species.get(y[0], f"unknown({y[0]})")
            proba = model.predict_proba(pred_data).tolist()

            return {
                "message": "Prediction success",
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "result": y,
                "species": spesies,
                "probabilities": {
                    "setosa": proba[0][0],
                    "versicolor": proba[0][1],
                    "virginica": proba[0][2],
                }
            }
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}") from e
