"""
regression/model.py
===================
Définition et entraînement des modèles de régression.

Modèles :
    Ridge         — linéaire régularisé, interprétable via coefficients
    RandomForest  — non-linéaire, interprétable via feature importances

Les deux sont entraînés sur les mêmes données pour comparer
l'apport de la non-linéarité sur la prédiction du groove.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# FIT
# =========================================================

def fit_models(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    seed: int = 42,
) -> dict:
    """
    Entraîne Ridge et RandomForest sur (X, y).

    Args:
        X:        features (n_stimuli, n_features), déjà normalisées si normalize=True
        y:        target groove_mean (n_stimuli,)
        features: noms des colonnes (pour les rapports)
        seed:     graine pour RandomForest

    Returns:
        dict { "Ridge": model, "RandomForest": model }
    """
    models = {}

    # ── Ridge ────────────────────────────────────────────
    # RidgeCV sélectionne automatiquement alpha par CV interne
    # StandardScaler en pipeline au cas où normalize=False dans data_loader
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)),
    ])
    ridge.fit(X, y)
    models["Ridge"] = ridge

    # ── RandomForest ─────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    models["RandomForest"] = rf

    return models