import numpy as np
from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("metrics_view")
class MetricsViewStep(AnalysisStep):

    name = "metrics_view"

    def run(self, context):

        df = context.dataset

        required_cols = ["D", "I", "V", "S_real", "E_real"]

        # =====================================================
        # VALIDATION
        # =====================================================

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(
                f"MetricsViewStep missing columns: {missing}"
            )

        # =====================================================
        # MATRIX BUILD
        # =====================================================

        metrics = np.stack(
            [df[c].values for c in required_cols],
            axis=1
        )

        context.cache["metrics_matrix"] = metrics

        return context