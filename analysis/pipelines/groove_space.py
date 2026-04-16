import pandas as pd
from analysis.embeddings.umap_groove import umap_3d
from config import METADATA_PATH


def run_groove_pipeline(save=True):

    df = pd.read_csv(METADATA_PATH)

    emb, reducer = umap_3d(df)

    return df, emb, reducer
