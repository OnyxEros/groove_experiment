import pandas as pd
import numpy as np

from perception_space.core.validation import validate_perception_df

def align_embeddings_with_perception(embeddings, df):
    """
    Aligne les embeddings avec les ratings perceptifs via une jointure sur stimulus_id.

    embeddings : np.ndarray shape (n, d), avec embeddings[i] correspondant au stimulus i
    df         : DataFrame avec colonnes stimulus_id, groove, complexity

    Hypothèse : stimulus_id dans df est un entier ou castable en int,
                et correspond à l'index de ligne dans embeddings.
    """
    df = df.copy()
    validate_perception_df(df)

    if "stimulus_id" not in df.columns:
        raise ValueError("Missing stimulus_id")

    # Cast stimulus_id en int pour l'indexation
    try:
        df["stimulus_id"] = df["stimulus_id"].astype(int)
    except (ValueError, TypeError) as e:
        raise ValueError(f"stimulus_id must be castable to int: {e}")

    n_embeddings = embeddings.shape[0]

    # Filtre les stimulus_id valides
    valid_mask = (df["stimulus_id"] >= 0) & (df["stimulus_id"] < n_embeddings)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"[align] Warning: {n_invalid} stimulus_id hors plage ignorés")
    df = df[valid_mask].copy()

    if df.empty:
        raise ValueError("No valid stimulus_id found after filtering")

    # Jointure explicite par index
    idx        = df["stimulus_id"].values
    X_aligned  = embeddings[idx]
    y_groove   = df["groove"].values
    y_complexity = df["complexity"].values

    return X_aligned, y_groove, y_complexity