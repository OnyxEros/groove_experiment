import numpy as np
from analysis.embeddings.base import BaseEmbedding


class RealizedEmbedding(BaseEmbedding):

    name = "realized"

    def compute(self, df, cache=None):

        X = np.stack([
            df["D"].values,
            df["I"].values,
            df["V"].values,
            df["S_real"].values,
            df["E_real"].values,
        ], axis=1)

        return X.astype(float)
