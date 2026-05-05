from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import re


class Response(BaseModel):
    participant_id: str = Field(min_length=1, max_length=64)
    stim_id: str = Field(min_length=1, max_length=128)

    groove: int = Field(ge=1, le=7)
    complexity: int = Field(ge=1, le=7)

    rt: float = Field(ge=0, le=3600)  # cap à 1h, sanity check

    rt_type: Optional[str] = None
    trial_index: Optional[int] = Field(default=None, ge=0, le=10_000)
    session_id: Optional[str] = Field(default=None, max_length=128)
    condition: Optional[str] = Field(default=None, max_length=64)
    timestamp_client: Optional[float] = None

    @field_validator("participant_id", "stim_id")
    @classmethod
    def no_injection(cls, v: str) -> str:
        """Rejette les caractères suspects (SQL / path traversal)."""
        if re.search(r"['\";\\/<>]", v):
            raise ValueError("Caractère invalide dans l'identifiant")
        return v.strip()

    @field_validator("rt_type", "condition")
    @classmethod
    def strip_strings(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v else v

    @model_validator(mode="after")
    def rt_consistent(self) -> "Response":
        """Un RT de 0 n'est acceptable que si rt_type l'indique explicitement."""
        if self.rt == 0 and self.rt_type not in ("timeout", "skip", "error"):
            # Avertissement soft — on ne bloque pas mais on normalise
            object.__setattr__(self, "rt_type", "zero")
        return self