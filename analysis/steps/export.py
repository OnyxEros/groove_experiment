import json
import numpy as np

from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


def sanitize(obj):
    """Convertit les types numpy en types JSON-sérialisables."""
    if isinstance(obj, dict):
        return {
            int(k) if isinstance(k, np.integer) else k: sanitize(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize(x) for x in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


@register_step("export")
class ExportStep(AnalysisStep):

    name = "export"

    # Embeddings requis (produits par EmbeddingsStep)
    REQUIRED_EMBEDDINGS = ["emb_structural", "emb_realized", "emb_pattern"]

    # FIX #1 : "umap_emergent" supprimé — projection.py ne produit plus cette clé
    # (doublon de umap_realized, retiré dans projection.py avec commentaire explicite).
    OPTIONAL_EMBEDDINGS = {
        "emb_structural":   ("embeddings", "structural"),
        "emb_realized":     ("embeddings", "realized"),
        "emb_pattern":      ("embeddings", "pattern"),
        "umap_realized":    ("embeddings", "umap_2d"),
        "umap_realized_3d": ("embeddings", "umap_3d"),
    }

    def run(self, context):
        rm = getattr(context, "run_manager", None)
        if rm is None:
            raise ValueError("ExportStep: run_manager manquant dans le contexte")

        cache = context.cache
        df    = context.dataset

        # ── Vérification des artefacts requis ────────────────────────────────
        missing = [k for k in self.REQUIRED_EMBEDDINGS if not self._has(cache, k)]
        if missing:
            raise ValueError(f"ExportStep: artefacts requis manquants : {missing}")

        # ── Export des embeddings (requis + optionnels) ───────────────────────
        for key, (folder, filename) in self.OPTIONAL_EMBEDDINGS.items():
            if self._has(cache, key):
                rm.save_npy(folder, filename, cache[key])

        # ── Clusters ─────────────────────────────────────────────────────────
        has_clusters  = self._has(cache, "clusters")
        has_semantics = self._has(cache, "cluster_semantics")

        if has_clusters:
            rm.save_npy("clustering", "labels", cache["clusters"])

        # ── Carte stimulus id ─────────────────────────────────────────────────
        if "stim_id" in df.columns:
            stim_id_map = df["stim_id"].tolist()
        elif "id" in df.columns:
            stim_id_map = [f"stim_{int(i):04d}" for i in df["id"]]
        else:
            stim_id_map = list(map(str, range(len(df))))

        rm.save_json("stim_id_map", sanitize(stim_id_map))

        # ── Interprétation clusters ───────────────────────────────────────────
        if has_semantics:
            rm.save_json("interpretation", sanitize(cache["cluster_semantics"]))

        # ── Debug snapshot ────────────────────────────────────────────────────
        debug_snapshot = {
            "n_samples":          int(len(df)),
            "cache_keys":         sorted(cache.keys()),
            "has_embeddings":     True,
            "has_clusters":       has_clusters,
            "has_interpretation": has_semantics,
            "umap_keys":          sorted(k for k in cache if "umap" in k),
        }

        if has_clusters:
            clusters = np.asarray(cache["clusters"])
            debug_snapshot.update({
                "n_clusters":    int(len(np.unique(clusters))),
                "cluster_ids":   np.unique(clusters).tolist(),
                "cluster_sizes": np.bincount(clusters).tolist(),
            })

        context.cache["debug_snapshot_export"] = debug_snapshot

        # ── Diagnostics (une seule fois, ici) ────────────────────────────────
        # engine.py n'appelle PAS finalize() — c'est ExportStep qui s'en charge.
        if hasattr(context, "diagnostics_collector"):
            diagnostics = context.diagnostics_collector.finalize(context)
            rm.save_json("diagnostics", sanitize(diagnostics))

        rm.save_json("debug_snapshot", sanitize(debug_snapshot))
        rm.save_json("summary",        sanitize(debug_snapshot))

        print(
            f"[EXPORT] terminé "
            f"(clusters={has_clusters}, semantics={has_semantics})"
        )
        return context

    @staticmethod
    def _has(cache, key):
        return key in cache and cache[key] is not None