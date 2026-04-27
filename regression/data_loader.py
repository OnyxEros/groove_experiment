"""
regression/data_loader.py
=========================
Charge et prépare les données pour la régression groove.

Pipeline :
    metadata.csv  ──┐
                    ├──► join sur stim_id ──► features + target ──► X, y
    responses.csv ──┘         (Supabase cache)

Features extraites :
    Paramètres manipulés  : S_mv, D_mv, E
    Métriques réalisées   : D, I, V, S_real, E_real
    (optionnel)            : BPM

Target :
    groove_mean  (moyenne des ratings groove par stim_id)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import METADATA_PATH
from perception.loader import load_perceptual_dataset


# =========================================================
# FEATURES
# =========================================================

# Paramètres du design expérimental (manipulés)
DESIGN_FEATURES = ["S_mv", "D_mv", "E"]

# Métriques acoustiques calculées sur les stimuli
ACOUSTIC_FEATURES = ["D", "I", "V", "S_real", "E_real"]

# Toutes les features disponibles
ALL_FEATURES = DESIGN_FEATURES + ACOUSTIC_FEATURES

# Target principale
TARGET = "groove_mean"


# =========================================================
# LOADER
# =========================================================

def load_regression_data(
    feature_set: str = "all",
    refresh: bool = False,
    min_participants: int = 1,
    normalize: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Charge les données prêtes pour la régression.

    Args:
        feature_set:      "design" | "acoustic" | "all"
        refresh:          re-fetch Supabase même si cache local existe
        min_participants: filtre les stimuli avec trop peu de réponses
        normalize:        centre-réduit les features (recommandé pour Ridge)

    Returns:
        df       : DataFrame complet (pour inspection / SHAP)
        X        : np.ndarray (n_stimuli, n_features)
        y        : np.ndarray (n_stimuli,)  → groove_mean
        features : liste des noms de colonnes dans X
    """
    # --- chargement et jointure ---
    meta = pd.read_csv(METADATA_PATH)

    # stim_id : utilise la colonne 'id' de metadata si stim_id absent
    if "stim_id" not in meta.columns:
        if "id" in meta.columns:
            from pathlib import Path
            meta["stim_id"] = meta["mp3_path"].apply(lambda p: Path(p).name)
        elif "id" in meta.columns:
            meta["stim_id"] = meta["id"].astype(str)
        else:
            raise ValueError(
                "metadata.csv doit contenir une colonne 'id' ou 'stim_id'.\n"
                f"Colonnes disponibles : {list(meta.columns)}"
            )
    
    meta["stim_id"] = meta["stim_id"].astype(str)
    df = load_perceptual_dataset(embedding_df=meta, refresh=refresh)

    # --- filtre qualité ---
    if "n_participants" in df.columns:
        before = len(df)
        df = df[df["n_participants"] >= min_participants].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"[data_loader] {dropped} stimuli filtrés (< {min_participants} participants)")

    if df.empty:
        raise ValueError(
            f"Aucun stimulus avec ≥ {min_participants} participant(s) après jointure."
        )

    # --- sélection des features ---
    features = _select_features(df, feature_set)

    # vérification des NaN
    missing = df[features].isnull().sum()
    if missing.any():
        bad = missing[missing > 0].to_dict()
        print(f"[data_loader] NaN détectés dans les features : {bad} — imputation par médiane")
        for col in bad:
            df[col] = df[col].fillna(df[col].median())

    X = df[features].values.astype(np.float64)
    y = df[TARGET].values.astype(np.float64)

    # --- normalisation ---
    if normalize:
        X, means, stds = _normalize(X)
        df = df.copy()
        df[features] = X

    return df, X, y, features


def _select_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Retourne la liste des features disponibles selon le feature_set."""
    candidates = {
        "design":   DESIGN_FEATURES,
        "acoustic": ACOUSTIC_FEATURES,
        "all":      ALL_FEATURES,
    }.get(feature_set, ALL_FEATURES)

    available = [f for f in candidates if f in df.columns]

    missing = set(candidates) - set(available)
    if missing:
        print(f"[data_loader] Features absentes ignorées : {missing}")

    if not available:
        raise ValueError(
            f"Aucune feature disponible pour feature_set='{feature_set}'.\n"
            f"Colonnes dans df : {list(df.columns)}"
        )

    return available


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre-réduit X. Retourne (X_norm, means, stds)."""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # évite division par zéro pour features constantes
    return (X - means) / stds, means, stds


# =========================================================
# HELPER : résumé rapide
# =========================================================

def describe_dataset(df: pd.DataFrame, features: list[str]) -> None:
    """Affiche un résumé lisible du dataset de régression."""
    print(f"\n{'─'*50}")
    print(f"  Dataset régression")
    print(f"{'─'*50}")
    print(f"  Stimuli        : {len(df)}")
    if "n_participants" in df.columns:
        print(f"  Participants   : {df['n_participants'].sum():.0f} réponses au total")
        print(f"  Médiane réponses/stim : {df['n_participants'].median():.1f}")
    print(f"  Features ({len(features)})  : {features}")
    print(f"  Target         : {TARGET}")
    print(f"  groove_mean    : {df[TARGET].mean():.3f} ± {df[TARGET].std():.3f}")
    print(f"{'─'*50}\n")