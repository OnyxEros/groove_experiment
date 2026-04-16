import pandas as pd
from config import METADATA_PATH, MP3_DIR
from analysis.embeddings.audio_dataset import build_audio_embeddings
from analysis.embeddings.umap_audio import umap_audio_3d


def build_explorer_dataset():
    df = pd.read_csv(METADATA_PATH)

    vectors, paths = build_audio_embeddings(str(MP3_DIR))
    emb, _ = umap_audio_3d(vectors)

    df = df.copy()
    df["path"] = paths
    df["x"] = emb[:, 0]
    df["y"] = emb[:, 1]
    df["z"] = emb[:, 2]

    return df
