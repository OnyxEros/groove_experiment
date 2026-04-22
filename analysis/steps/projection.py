import umap
import numpy as np
from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("projection")
class ProjectionStep(AnalysisStep):

    name = "projection"

    def run(self, context):

        if "emb_realized" not in context.cache:
            raise ValueError("ProjectionStep: missing 'emb_realized'")

        emb = context.cache["emb_realized"]

        emb = np.asarray(emb)

        if emb.ndim != 2:
            raise ValueError(f"ProjectionStep: invalid embedding shape {emb.shape}")

        # =====================================================
        # UMAP CONFIG
        # =====================================================

        reducer = umap.UMAP(
            n_components=3,
            metric="cosine",
            random_state=context.seed
        )

        proj_3d = reducer.fit_transform(emb)

        # derive 2D from same manifold (stable)
        proj_2d = proj_3d[:, :2]

        # =====================================================
        # STORE
        # =====================================================

        context.cache["umap_2d"] = proj_2d
        context.cache["umap_3d"] = proj_3d

        return context