import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

from config import UMAP_CONFIG


# =========================================================
# JOINT EMBEDDING (AUDIO + GROOVE)
# =========================================================

def compute_joint_embedding(
    X_audio: np.ndarray,
    X_groove: np.ndarray,
    return_labels: bool = True
):
    """
    Build a shared latent space for audio + groove features.

    Args:
        X_audio: (N1, D1)
        X_groove: (N2, D2)
    """

    if X_audio.ndim != 2 or X_groove.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")

    # -----------------------------------------------------
    # MERGE FEATURES
    # -----------------------------------------------------
    X = np.vstack([X_audio, X_groove])

    labels = (
        ["audio"] * len(X_audio)
        + ["groove"] * len(X_groove)
    )

    # -----------------------------------------------------
    # NORMALIZATION (global alignment)
    # -----------------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # -----------------------------------------------------
    # SHARED UMAP SPACE
    # -----------------------------------------------------
    reducer = umap.UMAP(**UMAP_CONFIG)
    Z = reducer.fit_transform(Xs)

    if return_labels:
        return Z, labels, reducer, scaler

    return Z, reducer, scaler
