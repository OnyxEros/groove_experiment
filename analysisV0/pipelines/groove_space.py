from analysis.embeddings.manager import EmbeddingManager
from analysis.embeddings.clustering import cluster_latent_space
import numpy as np


def run_groove_space(X_groove: np.ndarray):

    manager = EmbeddingManager()

    Z = manager.fit("groove", X_groove)

    labels, _ = cluster_latent_space(Z)

    return Z, labels