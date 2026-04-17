import umap
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import UMAP_CONFIG


class EmbeddingManager:
    """
    Centralized embedding system for all modalities (audio, groove, joint).
    """

    def __init__(self):
        self.scalers = {}
        self.reducers = {}

    # =====================================================
    # SINGLE MODALITY FIT
    # =====================================================

    def fit(self, name: str, X: np.ndarray):
        """
        Fit UMAP embedding for a single modality.
        """
        if hasattr(X, "select_dtypes"):
            raise ValueError(
                "EmbeddingManager expects numpy array, not DataFrame. "
                 "Select features BEFORE calling fit()."
            )

        if X.ndim != 2:
            raise ValueError("X must be a 2D array (N, D)")

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        reducer = umap.UMAP(**UMAP_CONFIG)
        Z = reducer.fit_transform(Xs)

        self.scalers[name] = scaler
        self.reducers[name] = reducer

        return Z

    # =====================================================
    # TRANSFORM NEW DATA
    # =====================================================

    def transform(self, name: str, X: np.ndarray):
        """
        Project new samples into an existing embedding space.
        """

        if name not in self.scalers or name not in self.reducers:
            raise ValueError(f"Model '{name}' not fitted yet")

        scaler = self.scalers[name]
        reducer = self.reducers[name]

        Xs = scaler.transform(X)
        return reducer.transform(Xs)

    # =====================================================
    # JOINT EMBEDDING (AUDIO + GROOVE)
    # =====================================================

    def fit_joint(self, X_audio: np.ndarray, X_groove: np.ndarray):
        """
        Build a shared latent space for audio + groove.

        This uses modality encoding to help UMAP distinguish sources.
        """

        if X_audio.ndim != 2 or X_groove.ndim != 2:
            raise ValueError("Inputs must be 2D arrays (N, D)")

        # =====================================================
        # CONCAT FEATURES
        # =====================================================

        X = np.vstack([X_audio, X_groove])

        # modality encoding (important for structure separation)
        modality = np.array(
            [0] * len(X_audio) + [1] * len(X_groove)
        ).reshape(-1, 1)

        X = np.hstack([X, modality])

        # =====================================================
        # SCALE
        # =====================================================

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # =====================================================
        # UMAP
        # =====================================================

        reducer = umap.UMAP(**UMAP_CONFIG)
        Z = reducer.fit_transform(Xs)

        # =====================================================
        # STORE MODELS
        # =====================================================

        self.scalers["joint"] = scaler
        self.reducers["joint"] = reducer

        return Z