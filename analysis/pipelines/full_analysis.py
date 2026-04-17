from analysis.embeddings.manager import EmbeddingManager
from analysis.embeddings.audio import compute_audio_umap
from analysis.embeddings.groove import compute_groove_umap
from analysis.embeddings.clustering import cluster_latent_space


def run_full_analysis(X_audio, X_groove, mode="joint"):

    manager = EmbeddingManager()

    # -----------------------------------------------------
    # AUDIO
    # -----------------------------------------------------
    if mode == "audio":
        Z = manager.fit("audio", X_audio)
        labels, _ = cluster_latent_space(Z)
        return Z, labels, manager

    # -----------------------------------------------------
    # GROOVE
    # -----------------------------------------------------
    if mode == "groove":
        Z = manager.fit("groove", X_groove)
        labels, _ = cluster_latent_space(Z)
        return Z, labels, manager

    # -----------------------------------------------------
    # JOINT MODE (STEP 2)
    # -----------------------------------------------------
    if mode == "joint":

        Z = manager.fit_joint(X_audio, X_groove)

        labels, _ = cluster_latent_space(Z)

        return Z, labels, manager

    raise ValueError("Only joint mode implemented in step 2")