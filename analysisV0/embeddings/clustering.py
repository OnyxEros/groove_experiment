import numpy as np
import hdbscan


# =========================================================
# CLUSTERING LATENT SPACE
# =========================================================

def cluster_latent_space(
    Z: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int | None = None
):
    """
    Robust clustering of embedding space.

    Returns:
        labels: cluster id (-1 = noise)
        clusterer: fitted model
    """

    if Z.ndim != 2:
        raise ValueError("Z must be (N, 3)")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(Z)

    return labels, clusterer