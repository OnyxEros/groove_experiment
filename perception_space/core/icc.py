"""
perception_space/core/icc.py
============================
Intraclass Correlation Coefficient (ICC) inter-participants.

Implémente ICC(2,1) — two-way random, single measures, absolute agreement.
C'est la mesure standard pour la fiabilité inter-juges en psychologie
de la perception (Shrout & Fleiss 1979, Koo & Mae 2016).

Interprétation (Koo & Mae 2016) :
    ICC < 0.50  : fiabilité faible
    0.50–0.75   : fiabilité modérée
    0.75–0.90   : fiabilité bonne
    > 0.90      : fiabilité excellente

Usage :
    from perception_space.core.icc import compute_icc, icc_summary
    result = compute_icc(ratings_wide)   # shape (n_stimuli, n_participants)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# =========================================================
# ICC(2,1) — Two-way random, absolute agreement
# =========================================================

def compute_icc(
    ratings: np.ndarray,
    model: str = "ICC2",
) -> dict:
    """
    Calcule l'ICC inter-participants sur une matrice de ratings.

    Args:
        ratings : np.ndarray shape (n_stimuli, n_participants)
                  Les NaN sont ignorés (participants absents sur certains stimuli).
        model   : "ICC1" | "ICC2" | "ICC3"
                  ICC2 recommandé pour des participants échantillonnés aléatoirement.

    Returns:
        dict {
            icc       : float — valeur ICC
            ci95_low  : float — borne basse IC 95%
            ci95_high : float — borne haute IC 95%
            F         : float — statistique F
            df1, df2  : degrés de liberté
            p_value   : float
            n_stimuli : int
            n_raters  : int
            interpretation : str
        }
    """
    ratings = np.asarray(ratings, dtype=np.float64)

    if ratings.ndim != 2:
        raise ValueError(f"ratings doit être 2D (n_stimuli, n_participants), got {ratings.ndim}D")

    n, k = ratings.shape  # n stimuli, k participants

    if n < 3:
        raise ValueError(f"Minimum 3 stimuli requis, got {n}")
    if k < 2:
        raise ValueError(f"Minimum 2 participants requis, got {k}")

    # ── Gestion NaN : remplace par la moyenne du stimulus ─
    row_means = np.nanmean(ratings, axis=1, keepdims=True)
    ratings_filled = np.where(np.isnan(ratings), row_means, ratings)

    # ── Sommes des carrés (ANOVA à deux facteurs) ─────────
    grand_mean = ratings_filled.mean()

    # SS between rows (stimuli)
    row_means_vec = ratings_filled.mean(axis=1)
    SS_r = k * np.sum((row_means_vec - grand_mean) ** 2)
    df_r = n - 1

    # SS between columns (participants/raters)
    col_means_vec = ratings_filled.mean(axis=0)
    SS_c = n * np.sum((col_means_vec - grand_mean) ** 2)
    df_c = k - 1

    # SS total
    SS_t = np.sum((ratings_filled - grand_mean) ** 2)
    df_t = n * k - 1

    # SS error (résiduel)
    SS_e = SS_t - SS_r - SS_c
    df_e = (n - 1) * (k - 1)

    # Mean squares
    MS_r = SS_r / df_r
    MS_c = SS_c / df_c
    MS_e = SS_e / df_e if df_e > 0 else 1e-10

    # ── ICC selon le modèle ───────────────────────────────
    if model == "ICC1":
        # One-way random
        icc_val = (MS_r - MS_e) / (MS_r + (k - 1) * MS_e)
        F_val   = MS_r / MS_e
        df1, df2 = df_r, df_t - df_r

    elif model == "ICC2":
        # Two-way random, absolute agreement
        icc_val = (MS_r - MS_e) / (MS_r + (k - 1) * MS_e + k * (MS_c - MS_e) / n)
        F_val   = MS_r / MS_e
        df1, df2 = df_r, df_e

    elif model == "ICC3":
        # Two-way mixed, consistency
        icc_val = (MS_r - MS_e) / (MS_r + (k - 1) * MS_e)
        F_val   = MS_r / MS_e
        df1, df2 = df_r, df_e

    else:
        raise ValueError(f"model doit être ICC1, ICC2 ou ICC3, got '{model}'")

    icc_val = float(np.clip(icc_val, -1.0, 1.0))

    # ── p-value ───────────────────────────────────────────
    p_value = float(1 - stats.f.cdf(F_val, df1, df2))

    # ── Intervalle de confiance 95% (Shrout & Fleiss 1979) ─
    alpha = 0.05
    F_lower = F_val / stats.f.ppf(1 - alpha / 2, df1, df2)
    F_upper = F_val * stats.f.ppf(1 - alpha / 2, df2, df1)

    if model == "ICC1":
        ci_low  = (F_lower - 1) / (F_lower + k - 1)
        ci_high = (F_upper - 1) / (F_upper + k - 1)
    else:
        ci_low  = (F_lower - 1) / (F_lower + k - 1)
        ci_high = (F_upper - 1) / (F_upper + k - 1)

    ci_low  = float(np.clip(ci_low,  -1.0, 1.0))
    ci_high = float(np.clip(ci_high, -1.0, 1.0))

    return {
        "icc":            icc_val,
        "ci95_low":       ci_low,
        "ci95_high":      ci_high,
        "F":              float(F_val),
        "df1":            int(df1),
        "df2":            int(df2),
        "p_value":        p_value,
        "n_stimuli":      int(n),
        "n_raters":       int(k),
        "model":          model,
        "MS_r":           float(MS_r),
        "MS_e":           float(MS_e),
        "interpretation": _interpret_icc(icc_val),
    }


# =========================================================
# ICC PAR STIMULUS (variabilité locale)
# =========================================================

def compute_per_stimulus_variance(
    ratings_long: pd.DataFrame,
    stim_col: str = "stimulus_id",
    rating_col: str = "groove",
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Calcule la variance inter-participants par stimulus.
    Utile pour identifier les stimuli ambigus.

    Returns:
        DataFrame avec colonnes : stimulus_id, mean, std, cv, n_raters, iqr
    """
    rows = []
    for stim_id, group in ratings_long.groupby(stim_col):
        vals = group[rating_col].dropna().values
        if len(vals) < 2:
            continue
        rows.append({
            stim_col:    stim_id,
            "mean":      float(np.mean(vals)),
            "std":       float(np.std(vals, ddof=1)),
            "cv":        float(np.std(vals, ddof=1) / (np.mean(vals) + 1e-9)),
            "iqr":       float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "n_raters":  int(len(vals)),
        })
    return pd.DataFrame(rows).sort_values("std", ascending=False)


# =========================================================
# WIDE FORMAT HELPER
# =========================================================

def ratings_to_wide(
    ratings_long: pd.DataFrame,
    stim_col: str = "stimulus_id",
    participant_col: str = "participant_id",
    rating_col: str = "groove",
) -> np.ndarray:
    """
    Convertit un DataFrame long en matrice wide (n_stimuli × n_participants).
    Les cellules manquantes sont NaN.
    """
    pivot = ratings_long.pivot_table(
        index=stim_col,
        columns=participant_col,
        values=rating_col,
        aggfunc="mean",
    )
    return pivot.values.astype(np.float64)


# =========================================================
# INTERPRÉTATION
# =========================================================

def _interpret_icc(icc: float) -> str:
    if icc < 0:
        return "négatif (variance résiduelle > variance inter-stimuli)"
    elif icc < 0.50:
        return "faible"
    elif icc < 0.75:
        return "modérée"
    elif icc < 0.90:
        return "bonne"
    else:
        return "excellente"


def icc_summary(result: dict) -> None:
    """Affiche un rapport ICC dans le terminal."""
    w = 52
    print(f"\n{'─'*w}")
    print(f"  ICC inter-participants  [{result['model']}]")
    print(f"{'─'*w}")
    print(f"  Stimuli     : {result['n_stimuli']}")
    print(f"  Participants: {result['n_raters']}")
    print(
        f"  ICC         : {result['icc']:.3f}  "
        f"[{result['ci95_low']:.3f} – {result['ci95_high']:.3f}]  95% CI"
    )
    print(
        f"  F({result['df1']}, {result['df2']})    : {result['F']:.2f}   p = {result['p_value']:.4f}"
    )
    print(f"  Fiabilité   : {result['interpretation']}")
    print(f"{'─'*w}\n")
