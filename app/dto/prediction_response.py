from pydantic import BaseModel
from typing import Optional

class Prediction(BaseModel):
    nextHourAverageUtilization: float
    nextHourMaximumUtilization: float
    nextHourMinimumUtilization: float
    peakProbabilityPercentage: float
    confidenceScore: float
    riskLevel: str
    trend: str

class PredictionResponse(BaseModel):
    prediction: Prediction
  

