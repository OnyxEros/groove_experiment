"""
regression/models/elasticnet.py
================================
ElasticNet (L1 + L2) pour la sélection de features.

Rôle dans le mémoire :
    Complément interprétatif de Ridge : la pénalité L1 met certains β = 0,
    révélant quelles features sont vraiment informatives vs redondantes.
    Les features retenues par ElasticNet ET significatives au LMM constituent
    le signal le plus robuste.

Note : X doit être déjà normalisé avant fit().
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import ElasticNetCV
from regression.models.base import GrooveModel


class ElasticNetModel(GrooveModel):

    name = "ElasticNet"
    supports_raw_data = False

    def __init__(self, seed: int = 42):
        self.seed    = seed
        self._model  = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
            alphas=[0.001, 0.01, 0.1, 0.5, 1.0],
            cv=5,
            random_state=seed,
            max_iter=5000,
        )
        self._features: list[str] = []
        self._fitted = False

    def fit(self, X, y, features, df_raw=None) -> "ElasticNetModel":
        self._features = features
        self._model.fit(X, y)
        self._fitted = True
        n_zero = int(np.sum(np.abs(self._model.coef_) < 1e-6))
        print(
            f"  [ElasticNet] α={self._model.alpha_:.4f}, "
            f"l1_ratio={self._model.l1_ratio_:.2f}, "
            f"features nulles : {n_zero}/{len(features)}"
        )
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
        selected  = [f for f, c in coefs.items() if abs(c) > 1e-6]
        eliminated = [f for f, c in coefs.items() if abs(c) <= 1e-6]
        return {
            "name":              self.name,
            "coefs":             coefs,
            "alpha":             float(self._model.alpha_),
            "l1_ratio":          float(self._model.l1_ratio_),
            "selected_features": selected,
            "eliminated_features": eliminated,
            "n_selected":        len(selected),
        }

    @property
    def estimator(self):
        return self._model