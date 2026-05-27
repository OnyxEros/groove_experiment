from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler


def normalize(
    X: np.ndarray,
    *,
    return_scaler: bool = False,
) -> np.ndarray | tuple[np.ndarray, RobustScaler]:
    """
    Normalisation robuste des embeddings.

    Utilise median/IQR plutôt que mean/std afin de :
        - réduire l'impact des outliers,
        - stabiliser les espaces perceptifs,
        - améliorer la cohérence des métriques locales.

    Args:
        X : np.ndarray shape (n_samples, n_features)
        return_scaler : retourne aussi le scaler fitted

    Returns:
        X_norm
        ou (X_norm, scaler)
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(
            f"normalize attend un array 2D, got shape={X.shape}"
        )

    if X.shape[0] < 2:
        raise ValueError(
            "normalize nécessite au moins 2 samples"
        )

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )

    X_norm = scaler.fit_transform(X)

    # sécurité numérique
    X_norm = np.nan_to_num(
        X_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if return_scaler:
        return X_norm, scaler

    return X_norm