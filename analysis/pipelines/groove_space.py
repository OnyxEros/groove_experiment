import pandas as pd
from analysis.embeddings.groove import compute_umap_groove


def run_groove_pipeline(df, save=True):

    if df.empty:
        raise ValueError("Empty dataset")

    emb, reducer = compute_umap_groove(df)

    df = df.copy()
    df["u1"] = emb[:, 0]
    df["u2"] = emb[:, 1]
    df["u3"] = emb[:, 2]

    return df, reducer