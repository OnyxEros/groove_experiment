"""
regression/data_loader.py
=========================
Charge et prépare les données pour la régression groove.

Pipeline (agrégé) :
    metadata.csv  ──┐
                    ├──► join sur stim_id ──► features + groove_mean ──► X, y
    responses.csv ──┘

Pipeline (brut, pour LMM) :
    responses.csv ──► join sur stim_id ──► features + groove + participant_id

Features disponibles :
    Design (manipulés) : S_mv, D_mv, E, P
    Acoustic (réalisés): D, I, V, S_real, E_real, P_real

Target :
    groove_mean  — moyenne des ratings groove par stim_id (agrégé)
    groove       — rating brut par réponse (pour LMM)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from config import METADATA_PATH
from perception.loader import load_perceptual_dataset
from perception.supabase_io import fetch_ratings

# =========================================================
# FEATURE SETS
# =========================================================

DESIGN_FEATURES   = ["S_mv", "D_mv", "E", "P"]
ACOUSTIC_FEATURES = ["D", "I", "V", "S_real", "E_real", "P_real"]
ALL_FEATURES      = DESIGN_FEATURES + ACOUSTIC_FEATURES
TARGET            = "groove_mean"


# =========================================================
# STIM_ID
# =========================================================

def _resolve_stim_id(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    if "stim_id" in meta.columns:
        meta["stim_id"] = meta["stim_id"].astype(str)
        return meta
    if "mp3_path" in meta.columns:
        meta["stim_id"] = meta["mp3_path"].apply(lambda p: Path(p).stem)
        return meta
    if "id" in meta.columns:
        meta["stim_id"] = meta["id"].apply(lambda i: f"stim_{int(i):04d}")
        return meta
    raise ValueError(
        "metadata.csv ne contient ni 'stim_id', ni 'mp3_path', ni 'id'.\n"
        f"Colonnes disponibles : {list(meta.columns)}"
    )


# =========================================================
# LOADER AGRÉGÉ (Ridge + RF)
# =========================================================

def load_regression_data(
    feature_set:      str  = "all",
    refresh:          bool = False,
    min_participants: int  = 1,
    normalize:        bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Charge les données agrégées (groove_mean par stim_id) pour Ridge et RF.

    Returns:
        df, X, y, features
    """
    meta = pd.read_csv(METADATA_PATH)
    meta = _resolve_stim_id(meta)

    df = load_perceptual_dataset(embedding_df=meta, refresh=refresh)

    if "n_participants" in df.columns:
        before  = len(df)
        df      = df[df["n_participants"] >= min_participants].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"[data_loader] {dropped} stimuli filtrés (< {min_participants} participant(s))")

    if df.empty:
        raise ValueError("Aucun stimulus disponible après jointure et filtrage.")

    features = _select_features(df, feature_set)

    missing_counts = df[features].isnull().sum()
    if missing_counts.any():
        bad = missing_counts[missing_counts > 0].to_dict()
        print(f"[data_loader] NaN → imputation médiane : {bad}")
        for col in bad:
            df[col] = df[col].fillna(df[col].median())

    X = df[features].values.astype(np.float64)
    y = df[TARGET].values.astype(np.float64)

    if normalize:
        X, _, _ = _normalize(X)
        df      = df.copy()
        df[features] = X

    return df, X, y, features


# =========================================================
# LOADER BRUT (LMM)
# =========================================================

def load_raw_responses(
    feature_set: str  = "all",
    refresh:     bool = False,
) -> pd.DataFrame | None:
    """
    Charge les réponses brutes (une ligne par réponse) jointes aux features
    des stimuli. Nécessaire pour le LMM (effet aléatoire participant).

    Colonnes retournées :
        groove, participant_id, stim_id + features du feature_set

    Returns:
        DataFrame brut ou None si données indisponibles.
    """
    try:
        raw = fetch_ratings(refresh=refresh)
    except Exception as e:
        print(f"  [LMM data] Impossible de charger les réponses brutes : {e}")
        return None

    if "participant_id" not in raw.columns:
        print("  [LMM data] colonne participant_id absente — LMM ignoré")
        return None

    # ── Join avec les features des stimuli ───────────────
    try:
        meta = pd.read_csv(METADATA_PATH)
        meta = _resolve_stim_id(meta)
    except Exception as e:
        print(f"  [LMM data] Impossible de charger metadata.csv : {e}")
        return None

    raw["stim_id"] = raw["stim_id"].astype(str)
    meta["stim_id"] = meta["stim_id"].astype(str)

    df = raw.merge(meta, on="stim_id", how="inner")

    if df.empty:
        print("  [LMM data] jointure raw × metadata vide")
        return None

    # ── Sélection des features disponibles ───────────────
    all_candidates = ALL_FEATURES if feature_set == "all" else (
        DESIGN_FEATURES if feature_set == "design" else ACOUSTIC_FEATURES
    )
    features_available = [f for f in all_candidates if f in df.columns]

    if not features_available:
        print("  [LMM data] aucune feature disponible")
        return None

    # ── Filtre RT ─────────────────────────────────────────
    if "rt" in df.columns:
        df = df[
            df["rt"].between(4.0, 600.0, inclusive="both") | df["rt"].isna()
        ].copy()

    keep_cols = ["groove", "participant_id", "stim_id"] + features_available
    df = df[[c for c in keep_cols if c in df.columns]].dropna(subset=["groove"])

    n = len(df)
    n_p = df["participant_id"].nunique()
    n_s = df["stim_id"].nunique()
    print(f"  [LMM data] {n} réponses brutes · {n_p} participants · {n_s} stimuli")

    return df


# =========================================================
# HELPERS
# =========================================================

def _select_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    candidates = {
        "design":   DESIGN_FEATURES,
        "acoustic": ACOUSTIC_FEATURES,
        "all":      ALL_FEATURES,
    }.get(feature_set, ALL_FEATURES)

    available = [f for f in candidates if f in df.columns]
    absent    = set(candidates) - set(available)

    if absent:
        print(f"[data_loader] Features absentes ignorées : {sorted(absent)}")
    if not available:
        raise ValueError(f"Aucune feature disponible pour feature_set='{feature_set}'.")

    return available


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds, means, stds


def describe_dataset(df: pd.DataFrame, features: list[str]) -> None:
    w = 50
    print(f"\n{'─'*w}")
    print(f"  Dataset régression")
    print(f"{'─'*w}")
    print(f"  Stimuli          : {len(df)}")
    if "n_participants" in df.columns:
        print(f"  Réponses totales : {int(df['n_participants'].sum())}")
        print(f"  Médiane / stim   : {df['n_participants'].median():.1f}")
    print(f"  Features ({len(features):2d})     : {features}")
    print(f"  Target           : {TARGET}")
    print(f"  groove_mean      : {df[TARGET].mean():.3f} ± {df[TARGET].std():.3f}")
    print(f"  groove range     : [{df[TARGET].min():.1f} – {df[TARGET].max():.1f}]")
    print(f"{'─'*w}\n")


# ── Constante RT ────────────────────────────────────────────
RT_MIN_S  = 4.0
RT_MAX_S  = 600.0


def filter_valid_responses(df):
    before = len(df)
    if "rt" in df.columns:
        df = df[
            df["rt"].between(RT_MIN_S, RT_MAX_S, inclusive="both") |
            df["rt"].isna()
        ].copy()
    n_dropped = before - len(df)
    if n_dropped > 0:
        print(f"[data_loader] {n_dropped} réponses filtrées (RT hors [{RT_MIN_S}s–{RT_MAX_S}s])")
    return df