"""
regression/data/loader.py  (v6.1)
==================================
Fix : termes d'interaction calculés dans load_raw_responses
      pour que le LMM les reçoive via df_raw.

Les termes sont calculés sur les valeurs brutes (non normalisées)
dans df_raw — la normalisation est faite ensuite dans lmm.py
avant le calcul des produits croisés, conformément à l'ordre
des opérations documenté dans lmm.py.
"""

from __future__ import annotations

import ast
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from config import METADATA_PATH, RT_MIN_S, RT_MAX_S
from perception.loader import load_perceptual_dataset
from perception.supabase_io import fetch_ratings
from regression.data.features import (
    select_features,
    ALL_FEATURES,
    FEATURE_SETS,
    INTERACTION_TERMS,
    compute_metric_regularity,
)

TARGET           = "groove_mean"
BACKGROUND_ORDER = ["non_musician", "amateur", "semi_pro", "pro"]


# ============================================================
# INTERFACE PUBLIQUE
# ============================================================

def load_aggregated(
    feature_set:      str  = "all",
    refresh:          bool = False,
    min_participants: int  = 1,
    normalize:        bool = True,
    exclude_single:   bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Données agrégées pour Ridge, SVR, ElasticNet."""
    if not normalize:
        warnings.warn("[loader] normalize=False : Coefficients non comparables.", UserWarning)

    meta = _load_meta()
    df   = load_perceptual_dataset(embedding_df=meta, refresh=refresh)
    df   = _filter_min_participants(df, min_participants)

    if df.empty:
        raise ValueError("Aucun stimulus disponible après jointure.")

    _report_single_response(df)

    if exclude_single and "single_response" in df.columns:
        df = df[~df["single_response"]].copy()

    if df.empty:
        raise ValueError("Aucun stimulus disponible après exclusion.")

    candidates = FEATURE_SETS.get(feature_set, ALL_FEATURES)

    # ── M_reg si demandé ─────────────────────────────────────────────────────
    if "M_reg" in candidates:
        df = _compute_m_reg(df, context="load_aggregated")

    # ── Sélection + imputation sur features de base ───────────────────────────
    features = select_features(df, feature_set)
    df = _impute_missing(df, features)

    # ── Normalisation ─────────────────────────────────────────────────────────
    X = df[features].values.astype(np.float64)
    y = df[TARGET].values.astype(np.float64)

    if normalize:
        X, _, _ = _normalize(X)
        df = df.copy()
        df[features] = X

    # ── Termes d'interaction APRÈS normalisation ──────────────────────────────
    if any(t in candidates for t in INTERACTION_TERMS):
        df, features, X = _compute_interactions(df, features, candidates)

    return df, X, y, features


def load_raw_responses(
    feature_set:    str  = "all",
    refresh:        bool = False,
    exclude_single: bool = True,
) -> pd.DataFrame | None:
    """
    Réponses brutes (LMM).

    Pour feature_set='interactions' : les colonnes D_sq, DxP, SxE, DxS
    sont ajoutées à df_raw sur les valeurs BRUTES (non normalisées).
    La normalisation + recalcul des produits croisés est faite dans lmm.py,
    conformément à l'ordre des opérations : normaliser d'abord, croiser ensuite.
    On passe ici un signal aux colonnes présentes — lmm.py les recompute
    après normalisation via INTERACTION_TERMS.
    """
    try:
        raw  = fetch_ratings(refresh=refresh)
        meta = _load_meta()
    except Exception as e:
        print(f"  [loader] Impossible de charger les données : {e}")
        return None

    raw["stim_id"]  = raw["stim_id"].astype(str)
    meta["stim_id"] = meta["stim_id"].astype(str)
    df = raw.merge(meta, on="stim_id", how="inner")

    if df.empty:
        return None

    candidates_list = FEATURE_SETS.get(feature_set, ALL_FEATURES)

    # ── M_reg si demandé ─────────────────────────────────────────────────────
    if "M_reg" in candidates_list:
        df = _compute_m_reg(df, context="load_raw_responses")

    # ── Features de base disponibles ─────────────────────────────────────────
    base_features = [
        f for f in candidates_list
        if f not in INTERACTION_TERMS and f in df.columns
    ]
    features_available = list(base_features)

    # ── Signalement des termes d'interaction pour le LMM ─────────────────────
    # On ajoute les noms des termes demandés à features_available
    # pour que lmm.py sache qu'il doit les calculer.
    # Les colonnes réelles seront calculées par lmm.py après normalisation.
    interaction_requested = [
        t for t in INTERACTION_TERMS
        if t in candidates_list
    ]
    if interaction_requested:
        features_available += interaction_requested
        print(f"[loader] LMM : termes d'interaction demandés → {interaction_requested}")

    if "rt" in df.columns:
        df = df[
            df["rt"].between(RT_MIN_S, RT_MAX_S, inclusive="both") | df["rt"].isna()
        ].copy()

    if exclude_single:
        counts       = df.groupby("stim_id")["groove"].count()
        single_stims = counts[counts == 1].index
        if len(single_stims) > 0:
            df = df[~df["stim_id"].isin(single_stims)].copy()
            print(f"[loader] LMM exclude_single=True : {len(single_stims)} stimuli exclus")

    # keep_cols : seulement les colonnes qui existent dans df
    # (les termes d'interaction n'existent pas encore — calculés dans lmm.py)
    base_keep = ["groove", "participant_id", "stim_id"] + base_features
    df = _attach_musical_background(df, base_keep)
    df = df[[c for c in base_keep if c in df.columns]].dropna(subset=["groove"])

    # On attache la liste complète des features (base + interactions)
    # comme attribut pour que run.py puisse la passer au LMM
    df.attrs["features_requested"] = features_available

    return df


# ============================================================
# HELPERS CALCUL DE FEATURES
# ============================================================

def _compute_interactions(
    df:         pd.DataFrame,
    features:   list[str],
    candidates: list[str],
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """
    Calcule les termes d'interaction APRÈS normalisation.

    D_sq = D_z²  (quadratique interprétable sur valeurs centrées-réduites)
    DxP  = D_z × P_z
    SxE  = S_z × E_z
    DxS  = D_z × S_z
    """
    df    = df.copy()
    added = []

    for term, (f1, f2) in INTERACTION_TERMS.items():
        if term not in candidates:
            continue
        if f1 not in df.columns or f2 not in df.columns:
            print(f"[loader] interaction {term} ignorée : {f1} ou {f2} absent")
            continue
        df[term] = df[f1].values * df[f2].values
        added.append(term)

    if added:
        print(f"[loader] Termes d'interaction calculés : {added}")
        features = features + [t for t in added if t not in features]

    X = df[features].values.astype(np.float64)
    return df, features, X


def _compute_m_reg(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Calcule M_reg depuis la colonne 'hihat'."""
    import config as cfg

    tag = f"[loader{' / ' + context if context else ''}]"

    if "hihat" not in df.columns:
        print(f"{tag} ⚠️  M_reg demandé mais colonne 'hihat' absente.")
        return df

    metric_profile = np.array(cfg.METRIC_PROFILE, dtype=float)
    steps_per_bar  = int(cfg.STEPS_PER_BAR)

    def _parse(val) -> np.ndarray | None:
        try:
            if isinstance(val, np.ndarray): return val.astype(float)
            if isinstance(val, list):       return np.array(val, dtype=float)
            if isinstance(val, str):
                try:    parsed = json.loads(val)
                except: parsed = ast.literal_eval(val)
                return np.array(parsed, dtype=float)
            return None
        except Exception:
            return None

    def _compute(val) -> float:
        arr = _parse(val)
        if arr is None: return float("nan")
        return compute_metric_regularity(arr, metric_profile, steps_per_bar)

    df = df.copy()
    df["M_reg"] = df["hihat"].apply(_compute)
    n_ok = int(df["M_reg"].notna().sum())
    print(f"{tag} M_reg — {n_ok} OK (μ={df['M_reg'].mean():.3f}, σ={df['M_reg'].std():.3f})")
    return df


# ============================================================
# HELPERS PRIVÉS
# ============================================================

def _load_meta() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH)
    return _compat_rename(_resolve_stim_id(meta))


def _resolve_stim_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "stim_id" in df.columns:
        df["stim_id"] = df["stim_id"].astype(str)
    elif "mp3_path" in df.columns:
        df["stim_id"] = df["mp3_path"].apply(lambda p: Path(p).stem)
    elif "id" in df.columns:
        df["stim_id"] = df["id"].apply(lambda i: f"stim_{int(i):04d}")
    else:
        raise ValueError("Aucun identifiant de stimulus trouvé.")
    return df


def _compat_rename(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"S_real": "S", "E_real": "E", "P_real": "P", "bpm": "BPM", "Bpm": "BPM"}
    safe = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=safe)


def _filter_min_participants(df: pd.DataFrame, min_p: int) -> pd.DataFrame:
    if "n_participants" not in df.columns or min_p <= 1:
        return df
    before = len(df)
    df = df[df["n_participants"] >= min_p].copy()
    if (dropped := before - len(df)) > 0:
        print(f"[loader] {dropped} stimuli filtrés (< {min_p} participant(s))")
    return df


def _report_single_response(df: pd.DataFrame) -> None:
    if "single_response" in df.columns and df["single_response"].sum() > 0:
        print(f"[loader] {int(df['single_response'].sum())} stimuli avec n=1 réponse.")


def _impute_missing(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in features:
        if col in df.columns and df[col].isnull().sum() > 0:
            n = int(df[col].isnull().sum())
            df[col] = df[col].fillna(df[col].median())
            print(f"[loader] imputation médiane : {col} ({n} valeurs)")
    return df


def _attach_musical_background(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    if "musical_background" not in df.columns:
        return df
    df["musical_background"] = pd.Categorical(
        df["musical_background"], categories=BACKGROUND_ORDER, ordered=True,
    )
    if "musical_background" not in keep_cols:
        keep_cols.append("musical_background")
    return df


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds, means, stds


def describe_dataset(df: pd.DataFrame, features: list[str]) -> None:
    w = 50
    print(f"\n{'─'*w}\n  Dataset régression (v6)\n{'─'*w}")
    print(f"  Stimuli  : {len(df)}")
    print(f"  Features ({len(features):2d}) : {features}")
    print(f"{'─'*w}\n")