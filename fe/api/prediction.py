import os
from typing import Dict, Tuple, List
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=".dev.env")
host = os.environ.get("BE_APP_HOST")
port = os.environ.get("BE_APP_PORT")

map_species = {0: "setosa", 1: "versicolor", 2: "virginica"}

def get_pred(data: Dict,) -> Tuple[str, List[int], str, Dict[str, float]]:
    try:
        resp = requests.post(url=f"http://{host}:{port}/predict_v2", json=data)
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}", []

    try:
        result = resp.json()
    except Exception:
        return f"Backend returned non-JSON (status {resp.status_code})", []

    if resp.status_code == 200:
        return (
            result.get("message", ""),
            result.get("result", []),
            result.get("species", ""),
            result.get("probabilities", {})
        )
    else:
        detail = result.get("detail", result)
        return f"Request failed ({resp.status_code}): {detail}", [], "", {}


def mapping_species(y: int | None) -> str:
    if y is None:
        return "-"
    return map_species.get(y, f"unknown({y})")
