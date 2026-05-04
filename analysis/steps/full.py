from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("full")
class FullAnalysisStep(AnalysisStep):

    name = "full"

    def run(self, context):

        # =====================================================
        # FULL PIPELINE ASSERTION
        # =====================================================

        required_steps = [
            "embeddings",
            "clustering",
            "projection",
            "temporal",
            "interpretation",
            "export",
        ]

        # optionnel : log debug
        print(f"[FULL PIPELINE] running {len(required_steps)} steps")

        return context