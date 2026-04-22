import pandas as pd
from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("conditional")
class ConditionalAnalysisStep(AnalysisStep):

    name = "conditional"

    def run(self, context):

        df = context.dataset

        grouped = df.groupby(
            ["S_mv", "D_mv", "E"]
        ).mean(numeric_only=True)

        context.cache["conditional_stats"] = grouped

        return context