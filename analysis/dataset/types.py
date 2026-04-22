from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class GrooveSample:
    id: int

    # conditions expérimentales
    phase: int
    repeat: int

    s_mv: int
    d_mv: int
    e: float

    # métriques dérivées
    D: float
    I: float
    V: float
    S_real: float
    E_real: float

    bpm: float

    # assets (optionnel)
    midi: Optional[str] = None
    waveform: Optional[str] = None

    metadata: Dict[str, Any] = None