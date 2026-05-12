"""
perception/loader.py
====================
Charge et agrège les ratings perceptifs depuis Supabase (ou cache local),
puis les joint avec les métadonnées des stimuli.

v2 :
    - load_ratings_df : agrège musical_background par stim_id
      → musical_background_mode (valeur la plus fréquente parmi les répondants)
      → pct_musicians (proportion de semi_pro + pro, entre 0 et 1)
      Ces deux colonnes sont optionnelles — absentes si la colonne
      musical_background est absente des réponses brutes.
"""

import pandas as pd
from pathlib import Path

from perception.supabase_io import fetch_ratings
from config import METADATA_PATH

# Ordre de niveau musical (pour tri et interprétation)
BACKGROUND_ORDER = ["non_musician", "amateur", "semi_pro", "pro"]
MUSICIAN_LEVELS  = {"semi_pro", "pro"}   # seuil "musicien formé"


# =========================================================
# RATINGS AGRÉGÉS PAR STIMULUS
# =========================================================

def load_ratings_df(refresh: bool = False) -> pd.DataFrame:
    """
    Retourne les ratings agrégés par stim_id (moyenne inter-participants).

    Colonnes toujours présentes :
        stim_id, groove_mean, groove_std, n_participants

    Colonnes optionnelles (si complexity présent dans les réponses) :
        complexity_mean

    Colonnes optionnelles (si musical_background présent dans les réponses) :
        musical_background_mode  — profil le plus fréquent parmi les répondants
        pct_musicians            — proportion de répondants semi_pro ou pro [0–1]

    Args:
        refresh: si True, re-fetch depuis Supabase même si le cache existe.
    """
    df = fetch_ratings(refresh=refresh)

    # ── Agrégation de base ────────────────────────────────
    agg_dict: dict = {
        "groove_mean":    ("groove", "mean"),
        "groove_std":     ("groove", "std"),
        "n_participants": ("participant_id", "nunique"),
    }

    if "complexity" in df.columns:
        agg_dict["complexity_mean"] = ("complexity", "mean")

    agg = (
        df.groupby("stim_id")
        .agg(**agg_dict)
        .reset_index()
    )

    # ── Agrégation musical_background ────────────────────
    if "musical_background" in df.columns and df["musical_background"].notna().any():

        def _mode(s):
            """Mode (valeur la plus fréquente) — NaN si aucune valeur."""
            vals = s.dropna()
            return vals.mode().iloc[0] if len(vals) > 0 else None

        def _pct_musicians(s):
            """Proportion semi_pro + pro parmi les réponses non-nulles."""
            vals = s.dropna()
            if len(vals) == 0:
                return float("nan")
            return float((vals.isin(MUSICIAN_LEVELS)).sum() / len(vals))

        bg_agg = (
            df.groupby("stim_id")["musical_background"]
            .agg(
                musical_background_mode=_mode,
                pct_musicians=_pct_musicians,
            )
            .reset_index()
        )

        agg = agg.merge(bg_agg, on="stim_id", how="left")

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
        DataFrame joint : features stimuli + groove_mean + groove_std
                          + n_participants (+ musical_background_* si disponibles)
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

    # Log des colonnes musical_background si présentes
    bg_cols = [c for c in df.columns if c.startswith("musical_background") or c == "pct_musicians"]
    if bg_cols:
        print(f"[loader] musical_background colonnes disponibles : {bg_cols}")

    return df