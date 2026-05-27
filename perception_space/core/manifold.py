"""
perception_space/core/manifold.py
==================================
Géométrie locale dans l'espace latent.

Métriques calculées pour chaque point i :
    local_mean      — moyenne des ratings dans le voisinage k-NN
    local_std       — écart-type des ratings dans le voisinage
    local_slope     — corrélation(distance, rating) dans le voisinage
                      → positif : les voisins lointains ont un groove plus élevé
                      → négatif : les voisins proches ont un groove plus élevé
    local_agreement — 1 - CV_local normalisé ∈ [0, 1]
                      → 1.0 : tous les voisins ont le même rating (accord parfait)
                      → 0.0 : désaccord maximal dans le voisinage
                      Remplace local_coherence (corrélation dist-rating → peu interprétable).

Correction v4 :
    local_coherence supprimée — corrélation(dist_to_centroid, rating) n'a pas
    d'interprétation psychoacoustique directe et produisait des valeurs proches
    de zéro (μ=−0.054) indistinguables du bruit.
    local_agreement mesure directement le consensus perceptif local.
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
            local_mean      : (n,) — moyenne locale des ratings
            local_std       : (n,) — écart-type local
            local_slope     : (n,) — corrélation distance-rating locale
            local_agreement : (n,) — consensus perceptif local ∈ [0, 1]
            k_effective     : int
        }
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(X)

    if n != len(y):
        raise ValueError(f"Incohérence X/y : {n} vs {len(y)}")
    if n < 3:
        raise ValueError(f"Pas assez de samples (n={n})")

    # ── k adaptatif ───────────────────────────────────────
    k_sqrt = max(2, int(np.sqrt(n)))
    k_eff  = min(k, k_sqrt, n - 1)

    if k_eff < k:
        warnings.warn(
            f"k réduit de {k} à {k_eff} (n={n}, sqrt(n)={k_sqrt})",
            UserWarning,
            stacklevel=2,
        )
    if k_eff < 3:
        raise ValueError(f"k_eff trop petit ({k_eff}) pour géométrie locale fiable")
    if k_eff < 5:
        warnings.warn(f"k_eff={k_eff} : géométrie locale peu stable", UserWarning, stacklevel=2)

    # ── KNN ───────────────────────────────────────────────
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Rating scale max pour normaliser local_agreement
    y_range = float(np.ptp(y))  # max - min
    if y_range < 1e-10:
        y_range = 1.0  # fallback si tous identiques

    local_mean      = np.empty(n, dtype=np.float64)
    local_std       = np.empty(n, dtype=np.float64)
    local_slope     = np.empty(n, dtype=np.float64)
    local_agreement = np.empty(n, dtype=np.float64)

    for i in range(n):
        idx     = indices[i]
        d       = distances[i]
        y_local = y[idx]

        # ── mean / std ────────────────────────────────────
        local_mean[i] = np.mean(y_local)
        std_i         = np.std(y_local)
        local_std[i]  = std_i

        # ── local_slope : corrélation(distance, rating) ──
        # Interprétation : est-ce que s'éloigner dans l'espace latent
        # change systématiquement le groove ?
        if d.std() < 1e-12 or std_i < 1e-12:
            local_slope[i] = 0.0
        else:
            local_slope[i] = float(np.corrcoef(d, y_local)[0, 1])

        # ── local_agreement : consensus perceptif local ──
        # CV normalisé par la plage globale des ratings.
        # local_agreement = 1 - (std / y_range)
        # → 1.0 : voisins unanimes
        # → 0.0 : voisins couvrent toute la plage
        local_agreement[i] = float(np.clip(1.0 - std_i / y_range, 0.0, 1.0))

    return {
        "local_mean":      local_mean,
        "local_std":       local_std,
        "local_slope":     local_slope,
        "local_agreement": local_agreement,
        "k_effective":     k_eff,
    }