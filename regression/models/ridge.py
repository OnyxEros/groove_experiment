"""
regression/models/ridge.py
===========================
Ridge regression (L2) pour la prédiction du groove.

Choix justifié :
    - Linéaire et interprétable : les β sont directement comparables
      entre features normalisées → lecture directe de l'importance relative.
    - Régularisation L2 : stabilise les estimateurs face à la légère
      multicolinéarité résiduelle entre D et E (VIF ≈ 5).
    - CV interne (RidgeCV, 5-fold) : sélection automatique de α.

Note : X doit être déjà normalisé (z-score) avant fit().
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import RidgeCV
from regression.models.base import GrooveModel


class RidgeModel(GrooveModel):

    name = "Ridge"
    supports_raw_data = False

    def __init__(self, seed: int = 42):
        self.seed     = seed
        self._model   = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            cv=5,
            scoring="r2",
        )
        self._features: list[str] = []
        self._fitted = False

    def fit(self, X, y, features, df_raw=None) -> "RidgeModel":
        self._features = features
        self._model.fit(X, y)
        self._fitted = True
        print(f"  [Ridge] α={self._model.alpha_:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_results(self) -> dict:
        coefs = dict(
            sorted(
                zip(self._features, self._model.coef_.tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )
        return {
            "name":    self.name,
            "coefs":   coefs,
            "alpha":   float(self._model.alpha_),
        }

    @property
    def estimator(self):
        return self._model