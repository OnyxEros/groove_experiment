from pydantic import BaseModel, Field


class Response(BaseModel):
    participant_id: str
    stim_id: str

    groove: int = Field(ge=1, le=7)
    complexity: int = Field(ge=1, le=7)

    rt: float
    rt_type: str | None = None

    trial_index: int | None = None
    session_id: str | None = None
    condition: str | None = None

    timestamp_client: float | None = None