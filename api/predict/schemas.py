from typing import List, Dict, Optional
from pydantic import BaseModel

class PredictionParams(BaseModel):
    sepal_length: float 
    sepal_width:  float 
    petal_length: float 
    petal_width:  float 

class PredictionResult(BaseModel):
    message: str
    result: List[int]
    species: str
    probabilities: Dict[str, float]

class ErrorResponse(BaseModel):
    message: str
    code: int
    hint: Optional[str] = None

