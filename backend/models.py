from pydantic import BaseModel, Field
from typing import Optional


class Response(BaseModel):
    participant_id: str
    stim_id: str

    groove: int = Field(ge=1, le=7)
    complexity: int = Field(ge=1, le=7)

    rt: float = Field(ge=0)

    rt_type: Optional[str] = None
    trial_index: Optional[int] = None
    session_id: Optional[str] = None
    condition: Optional[str] = None

    timestamp_client: Optional[float] = None