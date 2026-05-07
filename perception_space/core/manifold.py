"""
perception_space/core/manifold.py
==================================
Géométrie locale dans l'espace latent.

Corrections v2 :
    - k adaptatif : min(k_target, sqrt(n), n-1)
      → évite que "local" devienne "global" avec peu de stimuli
    - Calcul paires SHAP-style sans matrice O(n²) complète
    - Warning explicite si k effectif < 5
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_local_geometry(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 15,
) -> dict:
    """
    Calcule la géométrie locale de y dans l'espace X.

    Args:
        X : embeddings normalisés, shape (n, d)
        y : ratings (groove ou complexity), shape (n,)
        k : nombre de voisins cible — adapté automatiquement à n

    Returns:
        dict {
            local_mean      : moyenne de y dans le voisinage
            local_std       : écart-type de y dans le voisinage
            local_slope     : gradient local (régression linéaire sur les voisins)
            local_coherence : corrélation distance-rating dans le voisinage
            k_effective     : k réellement utilisé
        }
    """
    n = len(X)

    # ── k adaptatif ───────────────────────────────────────
    # Règle : k ≤ sqrt(n) pour rester "local", et k ≤ n-1
    k_sqrt   = max(2, int(np.sqrt(n)))
    k_eff    = min(k, k_sqrt, n - 1)

    if k_eff < k:
        warnings.warn(
            f"compute_local_geometry : k réduit de {k} à {k_eff} "
            f"(n={n}, règle k≤sqrt(n)={k_sqrt})",
            UserWarning,
            stacklevel=2,
        )

    if k_eff < 3:
        raise ValueError(
            f"Pas assez de samples ({n}) pour calculer la géométrie locale "
            f"(k_eff={k_eff} < 3)."
        )

    if k_eff < 5:
        warnings.warn(
            f"k_eff={k_eff} — géométrie locale peu fiable avec si peu de voisins.",
            UserWarning,
            stacklevel=2,
        )

    # ── KNN ───────────────────────────────────────────────
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    y = np.asarray(y, dtype=np.float64)

    # ── Métriques locales ────────────────────────────────
    local_mean  = np.array([y[idx].mean()                           for idx in indices])
    local_std   = np.array([y[idx].std()                            for idx in indices])
    local_slope = np.array([np.polyfit(np.arange(k_eff), y[idx], 1)[0] for idx in indices])

    # Cohérence locale : corrélation distance-centroïde / rating
    local_coherence = np.zeros(n)
    for i, idx in enumerate(indices):
        centroid  = X[idx].mean(axis=0)
        distances = np.linalg.norm(X[idx] - centroid, axis=1)
        y_local   = y[idx]

        if distances.std() < 1e-10 or y_local.std() < 1e-10:
            local_coherence[i] = 0.0
        else:
            local_coherence[i] = float(np.corrcoef(distances, y_local)[0, 1])

    return {
        "local_mean":      local_mean,
        "local_std":       local_std,
        "local_slope":     local_slope,
        "local_coherence": local_coherence,
        "k_effective":     k_eff,
    }