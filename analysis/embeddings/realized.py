import numpy as np
from analysis.embeddings.base import BaseEmbedding


class RealizedEmbedding(BaseEmbedding):
    """
    Embedding sur les descripteurs réalisés.

    Vecteur : [D, I, V, S_real, E_real, P_real]
    P_real optionnel (rétro-compatible si absent du DataFrame).
    """

    name = "realized"

    COLS = ["D", "I", "V", "S_real", "E_real", "P_real"]

    def compute(self, df, cache=None):
        cols = [c for c in self.COLS if c in df.columns]
        X    = np.stack([df[c].values for c in cols], axis=1)
        return X.astype(float)