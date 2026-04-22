from analysis.embeddings.manager import EmbeddingManager
from analysis.embeddings.clustering import cluster_latent_space
import numpy as np


def run_audio_space(X_audio: np.ndarray):

    manager = EmbeddingManager()

    Z = manager.fit("audio", X_audio)

    labels, _ = cluster_latent_space(Z)

    return Z, labels