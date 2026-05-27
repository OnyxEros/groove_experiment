"""
regression/models/random_forest.py
====================================
Modèle RandomForest pour la régression groove.

Capture les non-linéarités et interactions que Ridge ne peut pas modéliser.
Les feature importances (MDI) sont complémentaires aux coefficients Ridge.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from regression.models.base import GrooveModel


class RandomForestModel(GrooveModel):

    name = "RandomForest"
    supports_raw_data = False

    def __init__(self, seed: int = 42):
        self.seed   = seed
        self._model = RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
        self._features: list[str] = []
        self._fitted  = False

    def fit(self, X, y, features, df_raw=None) -> "RandomForestModel":
        self._features = features
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_results(self) -> dict:
        importances = dict(
            sorted(
                zip(self._features, self._model.feature_importances_.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        return {
            "name":        self.name,
            "coefs":       None,
            "importances": importances,
        }

    @property
    def estimator(self) -> RandomForestRegressor:
        """Accès direct à l'estimateur sklearn (pour SHAP, etc.)."""
        return self._model
