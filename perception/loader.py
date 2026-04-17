import numpy as np
import pandas as pd
from perception.supabase_io import fetch_ratings


def load_perceptual_dataset(embedding_df):
    """
    Merge embeddings with human ratings.
    """

    ratings = fetch_ratings()

    ratings_df = pd.DataFrame(ratings)

    if "stimulus_id" not in embedding_df.columns:
        raise ValueError("embedding_df must contain stimulus_id")

    df = embedding_df.merge(ratings_df, on="stimulus_id")

    return df
