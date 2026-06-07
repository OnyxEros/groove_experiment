"""
regression/models/base.py
==========================
Classe abstraite GrooveModel.

Contrat :
    fit(X, y, features, df_raw)  → self
    predict(X)                   → np.ndarray
    get_results()                → dict
    name                         → str
    supports_raw_data            → bool   (True pour LMM uniquement)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class GrooveModel(ABC):

    name: str
    supports_raw_data: bool = False

    @abstractmethod
    def fit(
        self,
        X:        np.ndarray,
        y:        np.ndarray,
        features: list[str],
        df_raw:   pd.DataFrame | None = None,
    ) -> "GrooveModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_results(self) -> dict: ...

    def is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)