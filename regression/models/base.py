"""
regression/models/base.py
==========================
Classe abstraite GrooveModel.

Chaque modèle expose la même interface, ce qui permet à evaluation.py
de les traiter uniformément sans isinstance() partout.

Contrat :
    fit(X, y, df_raw)   → self
    predict(X)          → np.ndarray
    get_results()       → dict   (métriques, coefs, etc.)
    name                → str
    supports_raw_data   → bool   (True pour LMM uniquement)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class GrooveModel(ABC):

    name: str
    supports_raw_data: bool = False  # True uniquement pour LMM

    @abstractmethod
    def fit(
        self,
        X:      np.ndarray,
        y:      np.ndarray,
        features: list[str],
        df_raw: pd.DataFrame | None = None,
    ) -> "GrooveModel":
        """
        Entraîne le modèle.
        X et y sont déjà normalisés (normalize=True dans load_aggregated).
        df_raw est utilisé uniquement par LMMModel.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit le groove pour X (normalisé)."""
        ...

    @abstractmethod
    def get_results(self) -> dict:
        """
        Retourne un dict serialisable avec les métriques et paramètres du modèle.
        Structure minimale :
            {
                "name":        str,
                "coefs":       dict | None,
                "importances": dict | None,
            }
        """
        ...

    def is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)
