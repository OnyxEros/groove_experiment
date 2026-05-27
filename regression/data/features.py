"""
regression/data/features.py
============================
Définition des feature sets et sélection des colonnes disponibles.

Feature sets :
    DESIGN         : paramètres manipulés (S_mv, D_mv, E_mv, P_mv)
    ACOUSTIC       : descripteurs réalisés (D, I, V, S, E, P)
    ALL            : union DESIGN + ACOUSTIC
    PREDICTABILITY : exploratoire — ALL + régularité métrique M_reg
    INTERACTIONS   : espace orthogonalisé (D, S, E, P) + termes croisés
                     (D_sq, DxP, SxE, DxS) + paramètres génératifs

    Pourquoi INTERACTIONS exclut I et V ?
        D et I sont colinéaires (r=0.91, ACP Axe1 alignés).
        E et V sont colinéaires (r=0.86, ACP Axe2 alignés).
        Inclure I avec D_sq provoque une multicolinéarité catastrophique
        dans le LMM (β~10¹³, AIC/BIC=nan). On travaille sur l'espace
        orthogonalisé (D, S, E, P) identifié à l'ACP — cohérent avec
        la section 6.3 du mémoire.

    Termes d'interaction :
        D_sq  — non-linéarité densité/groove (trop peu → pas de groove, trop → chaos)
        D×P   — l'effet du désalignement dépend-il de la densité ? (Keil 1995)
        S×E   — syncopation × micro-timing (Witek 2017)
        D×S   — densité et syncopation sont-elles synergiques ou substituables ?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy


# ────────────────────────────────────────────────────────────
# Feature sets
# ────────────────────────────────────────────────────────────

DESIGN_FEATURES         = ["S_mv", "D_mv", "E_mv", "P_mv"]
ACOUSTIC_FEATURES       = ["D", "I", "V", "S", "E", "P"]
ALL_FEATURES            = DESIGN_FEATURES + ACOUSTIC_FEATURES
PREDICTABILITY_FEATURES = ALL_FEATURES + ["M_reg"]

# Espace orthogonalisé : I et V exclus (colinéaires avec D et E)
# + paramètres génératifs + termes d'interaction
INTERACTION_FEATURES = [
    "S_mv", "D_mv", "E_mv", "P_mv",   # génératifs
    "D", "S", "E", "P",                # acoustiques orthogonaux (sans I, V)
    "D_sq", "DxP", "SxE", "DxS",      # termes croisés
]

FEATURE_SETS = {
    "design":         DESIGN_FEATURES,
    "acoustic":       ACOUSTIC_FEATURES,
    "all":            ALL_FEATURES,
    "predictability": PREDICTABILITY_FEATURES,
    "interactions":   INTERACTION_FEATURES,
}

# Labels affichables pour les figures
FEATURE_LABELS = {
    "D":     "Densité (D)",
    "I":     "Irrégularité (I)",
    "V":     "Variabilité (V)",
    "S":     "Syncopation (S)",
    "E":     "Micro-timing (E)",
    "P":     "Désalignement (P)",
    "D_mv":  "Densité gén. (D_mv)",
    "S_mv":  "Syncopation gén. (S_mv)",
    "E_mv":  "Micro-timing gén. (E_mv)",
    "P_mv":  "Décalage gén. (P_mv)",
    "BPM":   "BPM",
    "S_sq":  "Syncopation² (S²)",
    "M_reg": "Régularité métrique (M_reg)",
    "D_sq":  "Densité² (D²)",
    "DxP":   "Densité × Désalignement (D×P)",
    "SxE":   "Syncopation × Micro-timing (S×E)",
    "DxS":   "Densité × Syncopation (D×S)",
}

# Termes d'interaction — calculés dans loader._compute_interactions()
INTERACTION_TERMS = {
    "D_sq": ("D", "D"),
    "DxP":  ("D", "P"),
    "SxE":  ("S", "E"),
    "DxS":  ("D", "S"),
}


# ────────────────────────────────────────────────────────────
# M_reg — régularité métrique
# ────────────────────────────────────────────────────────────

def compute_metric_regularity(
    hihat:          np.ndarray,
    metric_profile: np.ndarray,
    steps_per_bar:  int,
) -> float:
    """
    Régularité métrique du hi-hat (M_reg).
    M_reg = moyenne des poids métriques aux positions frappées.
    M_reg → 1 : frappes sur les temps forts (prévisible)
    M_reg → 0 : frappes évitent les temps forts (imprévisible)
    """
    try:
        hihat = np.asarray(hihat, dtype=float)
        hits  = np.where(hihat == 1.0)[0]
        if len(hits) == 0:
            return float("nan")
        n = len(hihat)
        metric_full = np.tile(
            metric_profile / metric_profile.max(),
            int(np.ceil(n / steps_per_bar))
        )[:n]
        return float(np.mean(metric_full[hits]))
    except Exception:
        return float("nan")


# ────────────────────────────────────────────────────────────
# H_pred (conservée pour référence — NON utilisée)
# ────────────────────────────────────────────────────────────

def compute_predictability_feature(pattern) -> float:
    """Entropie de Shannon inverse — NON utilisée. Corrélée avec D, p=0.974."""
    try:
        if not isinstance(pattern, np.ndarray):
            pattern = np.array(pattern, dtype=float)
        else:
            pattern = pattern.astype(float)
        if pattern.sum() < 1e-10:
            return 0.0
        vals, counts = np.unique(pattern, return_counts=True)
        probs  = counts / counts.sum()
        h      = float(entropy(probs, base=2))
        n_cls  = len(vals)
        h_max  = np.log2(n_cls) if n_cls > 1 else 1.0
        h_norm = h / h_max if h_max > 0 else 0.0
        return float(1.0 / (1.0 + h_norm))
    except Exception:
        return float("nan")


# ────────────────────────────────────────────────────────────
# select_features
# ────────────────────────────────────────────────────────────

def select_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    """
    Retourne les features disponibles dans df pour le feature_set demandé.
    Exclut les features à variance nulle et logue les absentes.
    """
    candidates = FEATURE_SETS.get(feature_set, ALL_FEATURES)
    available  = [f for f in candidates if f in df.columns]
    absent     = set(candidates) - set(available)

    zero_var = [f for f in available if df[f].std() < 1e-10]
    if zero_var:
        print(f"[features] Variance nulle exclue : {zero_var}")
        available = [f for f in available if f not in zero_var]

    if absent:
        print(f"[features] Absentes ignorées : {sorted(absent)}")

    if not available:
        raise ValueError(
            f"Aucune feature disponible pour feature_set='{feature_set}'.\n"
            f"Colonnes du df : {list(df.columns)}"
        )

    return available