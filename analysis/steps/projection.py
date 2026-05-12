import numpy as np
import umap
from sklearn.preprocessing import normalize
from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("projection")
class ProjectionStep(AnalysisStep):
    name = "projection"

    def run(self, context):
        df = context.dataset

        # =====================================================
        # ESPACE DES DESCRIPTEURS ÉMERGENTS
        # Input: (D, I, V, S, E) depuis emb_realized
        # =====================================================
        if "emb_realized" not in context.cache:
            raise ValueError("ProjectionStep: missing 'emb_realized'")

        emb_realized = np.asarray(context.cache["emb_realized"])
        if emb_realized.ndim != 2:
            raise ValueError(f"ProjectionStep: invalid emb_realized shape {emb_realized.shape}")

        emb_realized_norm = normalize(emb_realized)

        reducer_realized = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=context.seed,
            n_neighbors=15,
            min_dist=0.1
        )
        umap_realized = reducer_realized.fit_transform(emb_realized_norm)

        context.cache["umap_realized"]       = umap_realized
        context.cache["umap_realized_model"] = reducer_realized

        reducer_3d = umap.UMAP(
            n_components=3,
            metric="cosine",
            random_state=context.seed,
            n_neighbors=15,
            min_dist=0.1
        )
        umap_realized_3d = reducer_3d.fit_transform(emb_realized_norm)
        context.cache["umap_realized_3d"] = umap_realized_3d

        # =====================================================
        # ESPACE ÉMERGENT
        # Input: (D, I, V, S_mv) — théorique depuis paramètres génératifs
        # =====================================================
        if all(col in df.columns for col in ["D", "I", "V", "S_mv"]):
            X_emergent = df[["D", "I", "V", "S_mv"]].values

            reducer_emergent = umap.UMAP(
                n_components=2,
                metric="euclidean",
                random_state=context.seed,
                n_neighbors=15,
                min_dist=0.1
            )
            umap_emergent = reducer_emergent.fit_transform(X_emergent)

            context.cache["umap_emergent"]       = umap_emergent
            context.cache["umap_emergent_model"] = reducer_emergent
        else:
            context.cache["umap_emergent"] = None

        # =====================================================
        # ESPACE PARAMÉTRIQUE
        # Input: (S_mv, D_mv, E_mv) — paramètres génératifs bruts
        # =====================================================
        if all(col in df.columns for col in ["S_mv", "D_mv", "E_mv"]):
            context.cache["parametric_coords"] = df[["S_mv", "D_mv", "E_mv"]].values
        else:
            context.cache["parametric_coords"] = None

        return context