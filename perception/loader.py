"""
perception/loader.py
====================
Charge et agrège les ratings perceptifs depuis Supabase (ou cache local),
puis les joint avec les métadonnées des stimuli.

Corrections :
    I1 — groove_std NaN quand n_participants == 1.
         pandas groupby.agg("std") retourne NaN pour n=1 (ddof=1).
         Ce NaN se propage silencieusement dans la jointure et peut
         casser les downstream qui font df[["groove_mean","groove_std"]].dropna().
         Correction : fillna(0.0) après agrégation — variance nulle pour n=1
         est mathématiquement correct.
         Un flag booléen single_response est ajouté pour identifier ces stimuli
         dans les analyses (à déclarer dans le mémoire comme limitation).

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
        stim_id, groove_mean, groove_std, n_participants, single_response

    Colonnes optionnelles (si complexity présent dans les réponses) :
        complexity_mean

    Colonnes optionnelles (si musical_background présent dans les réponses) :
        musical_background_mode  — profil le plus fréquent parmi les répondants
        pct_musicians            — proportion de répondants semi_pro ou pro [0–1]

    Note sur groove_std :
        Pour les stimuli avec n_participants == 1, groove_std est fixé à 0.0
        (variance nulle, mathématiquement correct) plutôt que NaN.
        Le flag single_response=True identifie ces cas.
        À déclarer dans le mémoire : les stimuli single_response ont une
        mesure de groove_mean moins fiable (pas de variabilité inter-juges).

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

    # ── I1 : Correction NaN pour n_participants == 1 ──────
    # pandas std(ddof=1) retourne NaN pour un seul point.
    # On remplace par 0.0 (variance nulle = correct pour n=1)
    # et on ajoute un flag pour traçabilité.
    n_nan_std = int(agg["groove_std"].isna().sum())
    if n_nan_std > 0:
        print(
            f"[loader] {n_nan_std} stimulus/stimuli avec n=1 réponse — "
            f"groove_std=NaN → 0.0 (flag single_response=True)"
        )
    agg["groove_std"]      = agg["groove_std"].fillna(0.0)
    agg["single_response"] = agg["n_participants"] == 1

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
                          + n_participants + single_response
                          + musical_background_* si disponibles.

    Note :
        Les stimuli avec single_response=True ont groove_std=0.0.
        Selon l'analyse, il peut être pertinent de les exclure :
            df = df[~df["single_response"]]
        ou de les pondérer différemment dans la régression.
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

    # Rapport single_response
    n_single = int(df["single_response"].sum()) if "single_response" in df.columns else 0
    if n_single > 0:
        print(
            f"[loader] {n_single}/{len(df)} stimuli avec une seule réponse "
            f"(single_response=True) — groove_std=0.0 pour ces stimuli."
        )

    # Log des colonnes musical_background si présentes
    bg_cols = [c for c in df.columns if c.startswith("musical_background") or c == "pct_musicians"]
    if bg_cols:
        print(f"[loader] musical_background colonnes disponibles : {bg_cols}")

    return df