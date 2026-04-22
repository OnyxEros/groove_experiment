import numpy as np
from analysis.embeddings.base import BaseEmbedding


class StructuralEmbedding(BaseEmbedding):

    name = "structural"

    def compute(self, df, cache=None):

        X = np.stack([
            df["S_mv"].values,
            df["D_mv"].values,
            df["E"].values,
            df["BPM"].values,
            df["phase"].values,
        ], axis=1)

        return X.astype(float)
