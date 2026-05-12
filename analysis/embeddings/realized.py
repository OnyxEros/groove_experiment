import numpy as np
from analysis.embeddings.base import BaseEmbedding


class RealizedEmbedding(BaseEmbedding):
    """
    Embedding sur les descripteurs émergents (réalisés).

    Vecteur : [D, I, V, S, E, P]
    P optionnel (rétro-compatible si absent du DataFrame).
    """

    name = "realized"

    COLS = ["D", "I", "V", "S", "E", "P"]

    def compute(self, df, cache=None):
        cols = [c for c in self.COLS if c in df.columns]
        X    = np.stack([df[c].values for c in cols], axis=1)
        return X.astype(float)