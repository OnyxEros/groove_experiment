from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
import re


# =========================================================
# GROOVE SAMPLE — Dataclass interne
#
# Notation mise à jour :
#   Paramètres génératifs : S_mv, D_mv, E_mv, P_mv
#   Descripteurs émergents : D, I, V, S, E, P
#
# Ancienne notation supprimée : s_mv, d_mv, e (float), s_real, e_real
# =========================================================

@dataclass
class GrooveSample:
    id: int

    # Conditions expérimentales
    phase:  int
    repeat: int

    # Paramètres génératifs (manipulés)
    S_mv: int
    D_mv: int
    E_mv: float
    P_mv: int

    # Descripteurs émergents (réalisés)
    D: float
    I: float
    V: float
    S: float   # syncopation réalisée (ex S_real)
    E: float   # micro-timing réalisé (ex E_real)
    P: float   # push/pull réalisé   (ex P_real)

    bpm: float

    # Assets (optionnel)
    midi:     Optional[str] = None
    waveform: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# RESPONSE — Modèle Pydantic (inchangé)
# =========================================================

class Response(BaseModel):
    participant_id: str = Field(min_length=1, max_length=64)
    stim_id:        str = Field(min_length=1, max_length=128)

    groove:     int   = Field(ge=1, le=7)
    complexity: int   = Field(ge=1, le=7)

    rt:      float = Field(ge=0, le=3600)
    rt_type: Optional[str] = None

    trial_index:      Optional[int]   = Field(default=None, ge=0, le=10_000)
    session_id:       Optional[str]   = Field(default=None, max_length=128)
    condition:        Optional[str]   = Field(default=None, max_length=64)
    timestamp_client: Optional[float] = None

    listen_duration: Optional[float] = Field(default=None, ge=0, le=3600)

    musical_background: Optional[Literal[
        "non_musician",
        "amateur",
        "semi_pro",
        "pro",
    ]] = None

    @field_validator("participant_id", "stim_id")
    @classmethod
    def no_injection(cls, v: str) -> str:
        if re.search(r"['\";\\/<>]", v):
            raise ValueError("Caractère invalide dans l'identifiant")
        return v.strip()

    @field_validator("rt_type", "condition")
    @classmethod
    def strip_strings(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v else v

    @model_validator(mode="after")
    def rt_consistent(self) -> "Response":
        if self.rt == 0 and self.rt_type not in ("timeout", "skip", "error"):
            object.__setattr__(self, "rt_type", "zero")
        return self