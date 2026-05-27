"""
analysis/steps/projection.py
=============================
Projections UMAP des espaces de représentation.

- umap_realized (2D + 3D) : depuis emb_realized (RealizedEmbedding).
- umap_emergent supprimé : était un doublon de umap_realized
  (même colonnes D/S/E/P, même scaler, même UMAP params).
- metric="cosine", min_dist=0.4, n_neighbors=15.
"""

import numpy as np
import umap

from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("projection")
class ProjectionStep(AnalysisStep):

    name = "projection"

    UMAP_PARAMS = dict(
        metric="cosine",
        n_neighbors=15,
        min_dist=0.4,
    )

    def run(self, context):
        if "emb_realized" not in context.cache:
            raise ValueError("ProjectionStep: 'emb_realized' manquant dans le cache")

        emb = np.nan_to_num(np.asarray(context.cache["emb_realized"]))

        if emb.ndim != 2:
            raise ValueError(
                f"ProjectionStep: shape invalide pour emb_realized : {emb.shape}"
            )

        seed = context.seed

        # ── 2D ───────────────────────────────────────────────────────────────
        reducer_2d = umap.UMAP(n_components=2, random_state=seed, **self.UMAP_PARAMS)
        context.cache["umap_realized"] = reducer_2d.fit_transform(emb)
        # FIX #3 : umap_realized_model retiré du cache — jamais consommé ni exporté,
        # conserve une référence volumineuse inutilement en mémoire.

        # ── 3D ───────────────────────────────────────────────────────────────
        reducer_3d = umap.UMAP(n_components=3, random_state=seed, **self.UMAP_PARAMS)
        context.cache["umap_realized_3d"] = reducer_3d.fit_transform(emb)

        # ── Coordonnées paramétriques (référence) ────────────────────────────
        param_cols = ["S_mv", "D_mv", "E_mv"]
        df = context.dataset
        context.cache["parametric_coords"] = (
            df[param_cols].values
            if all(c in df.columns for c in param_cols)
            else None
        )

        return context