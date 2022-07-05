from pydantic import BaseModel
from typing import List, Optional


class AccuracyScore(BaseModel):
    score: float


class ConfusionMatrixValues(BaseModel):
    Predicted: float
    Actual: float
    Value: float


class ConfusionMatrix(BaseModel):
    __root__: List[ConfusionMatrixValues]


class ClassificationModelOutput(BaseModel):
    accuracy_score: AccuracyScore
    confusion_matrix: ConfusionMatrix


