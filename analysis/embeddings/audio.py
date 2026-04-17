import umap
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_umap(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    Z = reducer.fit_transform(X_scaled)

    return Z, reducer, scaler


def umap_audio_3d(vectors: np.ndarray):
    return compute_umap(vectors, n_components=3)