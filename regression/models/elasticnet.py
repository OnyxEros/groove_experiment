"""
regression/models/elasticnet.py
================================
ElasticNet pour la régression groove.

Pourquoi ElasticNet ?
    - Combine pénalité L1 (LASSO) + L2 (Ridge)
    - L1 → sélection automatique des features redondantes (met certains β = 0)
    - L2 → stabilité en présence de multicolinéarité (D ↔ D_mv corrélés)
    - Utile pour confirmer quelles features sont réellement informatives
      au-delà de ce que Ridge seul révèle (S, E quasi-nuls → candidats à l'exclusion)

Interprétation mémoire :
    Les features avec coef = 0.0 après ElasticNet sont redondantes ou non-informatives.
    À comparer avec les features significatives du LMM (D★, P★).

Hyperparamètres :
    alpha   : [0.001, 0.01, 0.1, 1.0]   — force de régularisation totale
    l1_ratio: [0.1, 0.5, 0.9]           — mix L1/L2 (0=Ridge, 1=LASSO)
    CV      : 5-fold, scoring R²
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import ElasticNetCV
from regression.models.base import GrooveModel


class ElasticNetModel(GrooveModel):

    name = "ElasticNet"
    supports_raw_data = False

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._features: list[str] = []
        self._fitted = False

        self._model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
            alphas=[0.001, 0.01, 0.1, 0.5, 1.0],
            cv=5,
            random_state=seed,
            max_iter=5000,
        )

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
        selected = [f for f, c in coefs.items() if abs(c) > 1e-6]

        return {
            "name":        self.name,
            "coefs":       coefs,
            "importances": None,
            "alpha":       float(self._model.alpha_),
            "l1_ratio":    float(self._model.l1_ratio_),
            "selected_features": selected,
            "n_selected":  len(selected),
        }

    @property
    def estimator(self):
        return self._model
