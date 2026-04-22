import numpy as np
from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("temporal")
class TemporalStep(AnalysisStep):

    name = "temporal"

    def run(self, context):

        cache = context.cache

        # =====================================================
        # CHECK DEPENDENCIES (OPTIONNEL MAIS PROPRE)
        # =====================================================

        required = ["emb_realized"]
        for r in required:
            if r not in cache:
                raise ValueError(f"TemporalStep missing dependency: {r}")

        emb = np.asarray(cache["emb_realized"])

        # =====================================================
        # MINIMAL TEMPORAL FEATURES (REAL BASELINE)
        # =====================================================

        # centroid trajectory proxy (simple but meaningful)
        temporal_mean = np.mean(emb, axis=0)
        temporal_std = np.std(emb, axis=0)

        # pseudo temporal structure (baseline, extensible)
        temporal_features = {
            "mean": temporal_mean,
            "std": temporal_std,
            "shape": emb.shape,
        }

        # =====================================================
        # STORE
        # =====================================================

        cache["temporal"] = temporal_features

        return context