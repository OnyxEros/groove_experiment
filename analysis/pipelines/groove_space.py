from analysis.embeddings.manager import EmbeddingManager
from analysis.embeddings.clustering import cluster_latent_space
import numpy as np


def run_groove_space(df):
    manager = EmbeddingManager()

    FEATURES = ["D", "V", "S_real"]

    X = df[FEATURES].values.astype(np.float32)

    Z = manager.fit("groove", X)

    labels, _ = cluster_latent_space(Z)

    return Z, labels