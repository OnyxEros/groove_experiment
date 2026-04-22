from abc import ABC, abstractmethod
from .context import AnalysisContext


class AnalysisStep(ABC):
    name: str

    def __init__(self):
        if not hasattr(self, "name") or self.name is None:
            raise ValueError(f"{self.__class__.__name__} missing 'name'")

    @abstractmethod
    def run(self, context: AnalysisContext) -> AnalysisContext:
        pass