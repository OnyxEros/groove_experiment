from dataclasses import dataclass

@dataclass(frozen=True)
class GrooveSpace:
    S_levels: list[float]
    D_levels: list[float]
    E_levels: list[float]


DEFAULT_SPACE = GrooveSpace(
    S_levels=[0.1, 0.5, 0.9],
    D_levels=[0.1, 0.5, 0.9],
    E_levels=[0.1, 0.5, 0.9],
)
