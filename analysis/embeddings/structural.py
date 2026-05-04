import numpy as np
from analysis.embeddings.base import BaseEmbedding


class StructuralEmbedding(BaseEmbedding):
    """
    Embedding sur les paramètres génératifs.

    Vecteur : [S_mv, D_mv, E, P, BPM, phase]
    P optionnel (rétro-compatible si absent du DataFrame).
    """

    name = "structural"

    COLS = ["S_mv", "D_mv", "E", "P", "BPM", "phase"]

    def compute(self, df, cache=None):
        cols = [c for c in self.COLS if c in df.columns]
        X    = np.stack([df[c].values for c in cols], axis=1)
        return X.astype(float)