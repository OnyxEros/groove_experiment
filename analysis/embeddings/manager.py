import umap
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import UMAP_CONFIG


class EmbeddingManager:
    """
    Centralized embedding system (paper-safe version)

    Guarantees:
    - deterministic embeddings
    - modality separation
    - reproducibility
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        self.scalers = {}
        self.reducers = {}

    # =====================================================
    # FIT (TRAIN EMBEDDING SPACE)
    # =====================================================

    def fit(self, name: str, X: np.ndarray):

        # ---------------------------
        # VALIDATION
        # ---------------------------
        if hasattr(X, "select_dtypes"):
            raise ValueError("Expected numpy array, not DataFrame")

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be numpy array")

        if X.ndim != 2:
            raise ValueError("X must be 2D array (N, D)")

        # ---------------------------
        # NORMALIZATION
        # ---------------------------
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # ---------------------------
        # UMAP (DETERMINISTIC FIX)
        # ---------------------------
        reducer = umap.UMAP(
            **UMAP_CONFIG,
            random_state=self.random_state,   # 🔥 CRITICAL FIX
        )

        Z = reducer.fit_transform(Xs)

        # ---------------------------
        # STORE MODELS
        # ---------------------------
        self.scalers[name] = scaler
        self.reducers[name] = reducer

        return Z

    # =====================================================
    # TRANSFORM (NEW DATA → SAME SPACE)
    # =====================================================

    def transform(self, name: str, X: np.ndarray):

        if name not in self.scalers or name not in self.reducers:
            raise ValueError(f"Model '{name}' not fitted yet")

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be numpy array")

        Xs = self.scalers[name].transform(X)
        return self.reducers[name].transform(Xs)

    # =====================================================
    # OPTIONAL: REBUILD SAFE (IMPORTANT FOR PAPER)
    # =====================================================

    def fit_from_cache(self, name: str, X: np.ndarray):
        """
        Rebuild embedding deterministically (for analysis reproducibility)
        """

        return self.fit(name, X)