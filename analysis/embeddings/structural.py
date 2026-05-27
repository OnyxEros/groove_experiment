import numpy as np
from sklearn.preprocessing import StandardScaler
from analysis.embeddings.base import BaseEmbedding


class StructuralEmbedding(BaseEmbedding):
    """
    ESPACE GÉNÉRATIF (CAUSES)
    Colonnes : paramètres manipulés + phase + BPM.
    """

    name = "structural"
    COLS = ["S_mv", "D_mv", "E_mv", "P_mv", "BPM", "phase"]

    def compute(self, df, cache=None):
        if df is None or len(df) == 0:
            return np.zeros((0, 1))

        cols = [c for c in self.COLS if c in df.columns]

        if len(cols) < 2:
            raise ValueError(f"StructuralEmbedding: missing features, found only {cols}")

        X = np.stack([df[c].values.astype(float) for c in cols], axis=1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        return scaler.fit_transform(X)