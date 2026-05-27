import numpy as np
from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("metrics_view")
class MetricsViewStep(AnalysisStep):
    """
    Construit la matrice des descripteurs émergents indépendants.
    Aligné sur RealizedEmbedding.COLS — I et V exclus (redondants).
    """

    name = "metrics_view"

    # Doit rester synchronisé avec RealizedEmbedding.COLS
    COLS = ["D", "S", "E", "P"]

    def run(self, context):
        df = context.dataset

        missing = [c for c in self.COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"MetricsViewStep: colonnes manquantes {missing}\n"
                "Vérifie que generator.py calcule bien tous les descripteurs émergents."
            )

        metrics = np.stack([df[c].values for c in self.COLS], axis=1)
        context.cache["metrics_matrix"] = metrics

        return context