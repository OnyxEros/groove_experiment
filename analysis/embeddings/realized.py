import numpy as np
from sklearn.preprocessing import StandardScaler
from analysis.embeddings.base import BaseEmbedding


class RealizedEmbedding(BaseEmbedding):
    """
    ESPACE PERCEPTIF (EFFETS)

    Colonnes retenues : D, S, E, P.
      - I retiré : r=0.914 avec D (quasi-redondant).
      - V retiré : r=0.860 avec E (même phénomène mesuré deux fois,
        amplitude 1e-4 → StandardScaler amplifie du bruit).
    """

    name = "realized"
    COLS = ["D", "S", "E", "P"]

    def compute(self, df, cache=None):
        if df is None or len(df) == 0:
            return np.zeros((0, 1))

        cols = [c for c in self.COLS if c in df.columns]

        if len(cols) < 2:
            raise ValueError(f"RealizedEmbedding: missing features, found only {cols}")

        X = np.stack([df[c].values.astype(float) for c in cols], axis=1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        return scaler.fit_transform(X)