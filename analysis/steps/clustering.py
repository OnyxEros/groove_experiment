import numpy as np
from sklearn.cluster import KMeans

from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step


@register_step("clustering")
class ClusteringStep(AnalysisStep):

    name = "clustering"

    REQUIRED_KEYS = ["emb_realized"]

    def run(self, context):

        # =====================================================
        # DEPENDENCIES
        # =====================================================

        missing = [k for k in self.REQUIRED_KEYS if k not in context.cache]
        if missing:
            raise ValueError(f"ClusteringStep missing: {missing}")

        emb = context.cache["emb_realized"]

        # =====================================================
        # VALIDATION
        # =====================================================

        if not isinstance(emb, np.ndarray):
            raise TypeError("emb_realized must be numpy.ndarray")

        if len(emb) == 0:
            raise ValueError("Empty embedding matrix")

        if np.isnan(emb).any():
            raise ValueError("NaN detected in embeddings")

        # =====================================================
        # CONFIG
        # =====================================================

        n_clusters = context.config.get("n_clusters", 6)
        random_state = context.config.get("random_state", 42)

        context.log(f"[CLUSTERING] KMeans k={n_clusters} on shape={emb.shape}")

        # =====================================================
        # MODEL
        # =====================================================

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto"
        )

        labels = kmeans.fit_predict(emb)

        # =====================================================
        # STORE
        # =====================================================

        context.cache["clusters"] = labels
        context.cache["cluster_model"] = kmeans

        context.log(f"[CLUSTERING] done → {len(np.unique(labels))} clusters")

        return context