from pydantic import BaseModel, Field
from typing import List, Optional

class GPUData(BaseModel):
    gpu_id: Optional[str] = Field(None, alias="gpuId")
    window_minutes: Optional[int] = Field(60, alias="windowMinutes")
    timestamps: List[int]
    gpu_util: List[float] = Field(..., alias="gpuUtil")
    mem_util: Optional[List[float]] = Field(None, alias="memUtil")
    power: Optional[List[float]] = Field(None, alias="power")
    temperature: Optional[List[float]] = Field(None, alias="temperature")

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True  
