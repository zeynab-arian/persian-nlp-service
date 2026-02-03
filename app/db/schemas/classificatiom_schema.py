from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    model_name: str
    text: str

class LabelScore(BaseModel):
    label: str
    score: float

class PredictResponse(BaseModel):
    label: str

class PredictTopKResponse(BaseModel):
    top_k_labels: List[LabelScore]
