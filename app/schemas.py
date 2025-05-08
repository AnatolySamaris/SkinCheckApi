from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

class PredictionRequest(BaseModel):
    image: bytes = Field(..., description="Binary image data")
    gender: str = Field(..., description="User gender")
    birthdate: str = Field(..., description="Birth date in ISO format (YYYY-MM-DD)")
    localization: str = Field(..., description="Localization string")
    # user_id: Optional[int] = Field(None, description="Optional user ID")

class PredictionResponse(BaseModel):
    ok: bool = Field(..., description="Is response OK or not")
    status: int = Field(..., description="Status code of the response")
    data: str = Field(..., description="Predicted classes separated with \";\" if OK, error description if not OK")
    # prediction: str = Field(..., description="Model prediction result")
    # confidence: float = Field(..., description="Prediction confidence score")
    # age: Optional[int] = Field(None, description="Calculated user age")