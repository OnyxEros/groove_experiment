from analysis.embeddings.manager import EmbeddingManager
from analysis.embeddings.clustering import cluster_latent_space


def run_audio_space(X):
    manager = EmbeddingManager()

    Z = manager.fit("audio", X)

    labels, _ = cluster_latent_space(Z)

    return Z, labels