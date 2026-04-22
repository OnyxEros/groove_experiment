import numpy as np
from analysis.embeddings.base import BaseEmbedding

class PatternEmbedding(BaseEmbedding):

    name = "pattern"

    def compute(self, df, cache=None):

        stim_cache = cache.get("stim_cache")

        if stim_cache is None:
            raise ValueError("PatternEmbedding requires 'stim_cache'")

        X = []

        for _, row in df.iterrows():

            stim = stim_cache[row["id"]]

            vec = np.concatenate([
                stim["kick"],
                stim["snare"],
                stim["hihat"],
            ])

            X.append(vec)

        return np.array(X)