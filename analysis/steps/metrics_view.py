import numpy as np
from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("metrics_view")
class MetricsViewStep(AnalysisStep):

    name = "metrics_view"

    def run(self, context):

        df = context.dataset

        # Descripteurs émergents complets (D, I, V, S, E, P)
        # P (push/pull inter-voix) était absent — ajouté pour aligner avec
        # RealizedEmbedding.COLS et generator.py:Metrics.inter_voice_push()
        required_cols = ["D", "I", "V", "S", "E", "P"]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(
                f"MetricsViewStep missing columns: {missing}\n"
                "Vérifie que generator.py calcule bien tous les descripteurs émergents."
            )

        metrics = np.stack(
            [df[c].values for c in required_cols],
            axis=1
        )

        context.cache["metrics_matrix"] = metrics

        return context