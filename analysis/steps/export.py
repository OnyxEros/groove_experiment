import numpy as np
from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


def sanitize(obj):
    if isinstance(obj, dict):
        return {
            int(k) if isinstance(k, (np.integer,)) else k: sanitize(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [sanitize(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


@register_step("export")
class ExportStep(AnalysisStep):

    name = "export"

    def run(self, context):

        rm = getattr(context, "run_manager", None)
        if rm is None:
            raise ValueError("ExportStep: missing run_manager in context")

        cache = context.cache
        df = context.dataset

        required = ["emb_structural", "emb_realized", "clusters"]
        for key in required:
            if key not in cache:
                raise ValueError(f"ExportStep: missing '{key}' in cache")

        rm.save_npy("embeddings", "structural", cache["emb_structural"])
        rm.save_npy("embeddings", "realized",   cache["emb_realized"])
        rm.save_npy("clustering", "labels",      cache["clusters"])

        # ── CRITIQUE : mapping stim_id → row index ──────────────
        # realized.npy[i] correspond au stimulus dont le stim_id est
        # stim_id_map[i]. Sans ce mapping, l'alignement
        # perception_space est corrompu silencieusement.
        if "stim_id" in df.columns:
            stim_id_map = df["stim_id"].tolist()
        elif "id" in df.columns:
            stim_id_map = [f"stim_{int(i):04d}" for i in df["id"]]
        else:
            stim_id_map = [str(i) for i in range(len(df))]

        rm.save_json("stim_id_map", stim_id_map)

        if "cluster_semantics" in cache:
            clean = sanitize(cache["cluster_semantics"])
            rm.save_json("interpretation", clean)

        rm.save_json("summary", {
            "n_samples":          len(context.dataset),
            "has_embeddings":     True,
            "has_clusters":       True,
            "has_interpretation": "cluster_semantics" in cache,
            "has_stim_id_map":    True,
        })

        return context