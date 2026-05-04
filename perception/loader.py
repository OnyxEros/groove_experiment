"""
perception/loader.py
====================
Charge et agrège les ratings perceptifs depuis Supabase (ou cache local),
puis les joint avec les métadonnées des stimuli.
"""

import pandas as pd
from pathlib import Path

from perception.supabase_io import fetch_ratings
from config import METADATA_PATH


# =========================================================
# RATINGS AGRÉGÉS PAR STIMULUS
# =========================================================

def load_ratings_df(refresh: bool = False) -> pd.DataFrame:
    """
    Retourne les ratings agrégés par stim_id (moyenne inter-participants).

    Colonnes retournées :
        stim_id, groove_mean, groove_std, complexity_mean, n_participants

    Args:
        refresh: si True, re-fetch depuis Supabase même si le cache existe.
    """
    df = fetch_ratings(refresh=refresh)

    # Colonnes d'agrégation — complexity peut être absente
    agg_dict: dict = {
        "groove_mean":     ("groove", "mean"),
        "groove_std":      ("groove", "std"),
        "n_participants":  ("participant_id", "nunique"),
    }

    # CORRECTION : le ternaire inline dans .agg() ne fonctionne pas en pandas.
    # On ajoute complexity_mean seulement si la colonne existe.
    if "complexity" in df.columns:
        agg_dict["complexity_mean"] = ("complexity", "mean")

    agg = (
        df.groupby("stim_id")
        .agg(**agg_dict)
        .reset_index()
    )

    return agg


# =========================================================
# DATASET JOINT (stimuli × ratings)
# =========================================================

def load_perceptual_dataset(
    embedding_df: pd.DataFrame | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Joint les métriques des stimuli avec les ratings perceptifs.

    Si embedding_df est fourni, il doit contenir 'stim_id'.
    Sinon, charge metadata.csv automatiquement.

    Args:
        embedding_df: DataFrame optionnel (features/embeddings des stimuli).
        refresh:      forcer un re-fetch Supabase.

    Returns:
        DataFrame joint : features stimuli + groove_mean + groove_std + n_participants
    """
    if embedding_df is None:
        meta_path = Path(METADATA_PATH)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.csv introuvable : {meta_path}\n"
                "Lance d'abord : python cli.py --generate"
            )
        embedding_df = pd.read_csv(meta_path)

    if "stim_id" not in embedding_df.columns:
        raise ValueError(
            "embedding_df doit contenir une colonne 'stim_id'. "
            f"Colonnes disponibles : {list(embedding_df.columns)}"
        )

    ratings = load_ratings_df(refresh=refresh)

    if ratings.empty:
        raise ValueError("Aucune donnée perceptive disponible après agrégation.")

    # Alignement des types pour la jointure
    embedding_df = embedding_df.copy()
    ratings      = ratings.copy()
    embedding_df["stim_id"] = embedding_df["stim_id"].astype(str)
    ratings["stim_id"]      = ratings["stim_id"].astype(str)

    df = embedding_df.merge(ratings, on="stim_id", how="inner")

    if df.empty:
        raise ValueError(
            "La jointure metadata × ratings est vide.\n"
            "Vérifie que les stim_id dans metadata.csv correspondent "
            "à ceux stockés dans Supabase.\n"
            f"  metadata stim_id sample : {embedding_df['stim_id'].head(3).tolist()}\n"
            f"  ratings  stim_id sample : {ratings['stim_id'].head(3).tolist()}"
        )

    return df