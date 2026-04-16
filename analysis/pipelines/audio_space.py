from analysis.embeddings.audio_dataset import build_audio_embeddings
from analysis.embeddings.umap_audio import umap_audio_3d
from analysis.visualization.plot_audio import plot_audio_space


def run_audio_pipeline(mp3_dir, save=True):

    vectors, paths = build_audio_embeddings(mp3_dir)

    emb, reducer = umap_audio_3d(vectors)

    if save:
        plot_audio_space(emb, paths)

    return emb, paths, reducer