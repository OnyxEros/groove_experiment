"""
regression/models/svr.py
=========================
Support Vector Regression (kernel RBF) pour la régression groove.

Pourquoi SVR plutôt que RandomForest ?
    - n=118 stimuli : RF surapprend systématiquement (R²CV < 0 observé)
    - SVR RBF = bon biais inductif pour les petits datasets non-linéaires
    - Margin maximisation → meilleure généralisation que les arbres sur peu de données
    - Les hyperparamètres C et gamma sont optimisés par GridSearchCV interne

Hyperparamètres :
    C     : [0.1, 1, 10, 100]   — trade-off marge/erreur
    gamma : ['scale', 0.1, 1]   — largeur du kernel RBF
    CV    : 5-fold, scoring R²
"""

from __future__ import annotations
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from regression.models.base import GrooveModel


class SVRModel(GrooveModel):

    name = "SVR"
    supports_raw_data = False

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._features: list[str] = []
        self._fitted = False
        self._best_params: dict = {}

        param_grid = {
            "C":     [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", 0.1, 1.0],
        }

        self._model = GridSearchCV(
            SVR(kernel="rbf", epsilon=0.1),
            param_grid,
            cv=5,
            scoring="r2",
            refit=True,
            n_jobs=-1,
        )

    def fit(self, X, y, features, df_raw=None) -> "SVRModel":
        self._features = features
        self._model.fit(X, y)
        self._best_params = self._model.best_params_
        self._fitted = True
        print(
            f"  [SVR] best params : C={self._best_params.get('C')}, "
            f"gamma={self._best_params.get('gamma')}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_results(self) -> dict:
        # SVR n'a pas de coefs interprétables comme Ridge
        # On retourne les paramètres optimaux pour la transparence
        return {
            "name":        self.name,
            "coefs":       None,
            "importances": None,
            "best_params": self._best_params,
            "best_cv_r2":  float(self._model.best_score_),
        }

    @property
    def estimator(self):
        """Estimateur sklearn refitté sur toutes les données."""
        return self._model.best_estimator_
