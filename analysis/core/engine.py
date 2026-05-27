from analysis.core.step import AnalysisStep
from analysis.core.context import AnalysisContext


class AnalysisEngine:
    """
    Exécute une séquence de steps dans l'ordre.

    Note : finalize() des diagnostics est délégué à ExportStep,
    qui est le seul endroit où il est appelé.
    """

    def __init__(self, steps: list[AnalysisStep]):
        self.steps = steps

    def run(self, context: AnalysisContext) -> AnalysisContext:
        for step in self.steps:
            print(f"[ANALYSIS] ▶ {step.name}")
            context = step.run(context)

        print("[ANALYSIS] ✔ pipeline terminé")
        return context