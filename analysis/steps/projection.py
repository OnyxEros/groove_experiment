import numpy as np
import umap
from sklearn.preprocessing import normalize

from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("projection")
class ProjectionStep(AnalysisStep):

    name = "projection"

    def run(self, context):

        if "emb_realized" not in context.cache:
            raise ValueError("ProjectionStep: missing 'emb_realized'")

        emb = np.asarray(context.cache["emb_realized"])

        if emb.ndim != 2:
            raise ValueError(f"ProjectionStep: invalid shape {emb.shape}")

        # =====================================================
        # AUDIO SPACE (UMAP 3D → 2D slice)
        # =====================================================

        emb = normalize(emb)

        reducer = umap.UMAP(
            n_components=3,
            metric="cosine",
            random_state=context.seed
        )

        proj_3d = reducer.fit_transform(emb)
        proj_2d = proj_3d[:, :2]

        context.cache["umap_audio_3d"] = proj_3d
        context.cache["umap_audio_2d"] = proj_2d
        context.cache["umap_model"] = reducer

        # =====================================================
        # GENERATIVE SPACE (paramétrique)
        # =====================================================

        df = context.dataset

        if all(col in df.columns for col in ["D", "I", "V", "S_mv"]):
            X_em = df[["D", "I", "V", "S_mv"]].values

            reducer_em = umap.UMAP(
                n_components=2,
                metric="euclidean",
                random_state=context.seed
            )

            context.cache["umap_emergent"] = reducer_em.fit_transform(X_em)

        else:
            context.cache["umap_emergent"] = None

        return context