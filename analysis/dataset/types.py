from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class GrooveSample:
    id: int

    # Conditions expérimentales
    phase:  int
    repeat: int

    # Paramètres génératifs (manipulés) — suffixe _mv
    S_mv: int
    D_mv: int
    E_mv: float
    P_mv: int

    # Descripteurs émergents (réalisés) — sans indice
    D: float
    I: float
    V: float
    S: float   # syncopation réalisée (ex S_real)
    E: float   # micro-timing réalisé  (ex E_real)
    P: float   # push/pull réalisé     (ex P_real)

    bpm: float

    # Assets (optionnel)
    midi:     Optional[str] = None
    waveform: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)