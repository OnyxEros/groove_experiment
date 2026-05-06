"""
perception_space/core/validation.py
=====================================
complexity est optionnelle dans tout le pipeline.
Elle est requise dans REQUIRED_COLS uniquement si elle est présente dans le df.
"""

import pandas as pd

REQUIRED_COLS_ALWAYS    = {"stimulus_id", "groove"}
REQUIRED_COLS_IF_PRESENT = {"complexity"}


def validate_perception_df(df: pd.DataFrame) -> bool:
    # Colonnes toujours requises
    missing = REQUIRED_COLS_ALWAYS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # groove : pas de NaN
    if df["groove"].isna().any():
        raise ValueError("NaN values detected in 'groove' column")

    # stimulus_id : pas de NaN
    if df["stimulus_id"].isna().any():
        raise ValueError("NaN values detected in 'stimulus_id' column")

    # complexity : optionnelle — si présente, on accepte les NaN
    # (ils seront imputés en aval dans cmd_perception_space)

    return True