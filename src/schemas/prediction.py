from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    label: int
    species: str
    confidence: float = Field(ge=0, le=1)
