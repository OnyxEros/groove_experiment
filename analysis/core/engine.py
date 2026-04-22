from analysis.core.step import AnalysisStep
from analysis.core.context import AnalysisContext


class AnalysisEngine:
    """
    Executes a sequence of steps in order.
    """

    def __init__(self, steps: list[AnalysisStep]):
        self.steps = steps

    def run(self, context: AnalysisContext) -> AnalysisContext:

        for step in self.steps:
            print(f"[ANALYSIS] ▶ {step.name}")

            context = step.run(context)

        print("[ANALYSIS] ✔ pipeline finished")
        return context