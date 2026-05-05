"""
perception_space/core/stats.py
==============================
Tests statistiques pour l'analyse perceptive du groove.

Fonctions :
    anova_by_condition      — ANOVA one-way groove ~ condition (S_mv, D_mv, E, P)
    permutation_test        — test de permutation non-paramétrique
    eta_squared             — effect size η²
    partial_eta_squared     — η²p pour ANOVA multi-facteurs
    compute_condition_stats — stats descriptives par cellule du design
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


# =========================================================
# ANOVA ONE-WAY PAR CONDITION
# =========================================================

def anova_by_condition(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    ANOVA one-way groove ~ chaque variable de design.

    Args:
        df             : DataFrame avec groove_mean + colonnes conditions
        groove_col     : nom de la colonne target
        condition_cols : liste des variables à tester
                         (défaut : S_mv, D_mv, E, P si présentes)

    Returns:
        DataFrame avec colonnes : condition, F, p_value, eta2, interpretation
    """
    if condition_cols is None:
        condition_cols = [c for c in ["S_mv", "D_mv", "E", "P"] if c in df.columns]

    if groove_col not in df.columns:
        raise ValueError(f"Colonne '{groove_col}' absente du DataFrame")

    y = df[groove_col].dropna().values
    rows = []

    for cond in condition_cols:
        if cond not in df.columns:
            continue

        groups = [
            df.loc[df[cond] == level, groove_col].dropna().values
            for level in sorted(df[cond].unique())
        ]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            continue

        try:
            F, p = stats.f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan

        eta2 = _eta_squared_groups(groups)

        rows.append({
            "condition":      cond,
            "n_levels":       len(groups),
            "F":              float(F),
            "p_value":        float(p),
            "eta2":           float(eta2),
            "significant":    p < 0.05,
            "interpretation": _interpret_eta2(eta2),
        })

    return pd.DataFrame(rows).sort_values("eta2", ascending=False)


# =========================================================
# KRUSKAL-WALLIS (non-paramétrique)
# =========================================================

def kruskal_by_condition(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Kruskal-Wallis H-test — alternative non-paramétrique à l'ANOVA.
    Plus robuste avec de petits effectifs (< 30 stimuli).
    """
    if condition_cols is None:
        condition_cols = [c for c in ["S_mv", "D_mv", "E", "P"] if c in df.columns]

    rows = []
    for cond in condition_cols:
        if cond not in df.columns:
            continue

        groups = [
            df.loc[df[cond] == level, groove_col].dropna().values
            for level in sorted(df[cond].unique())
        ]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            continue

        try:
            H, p = stats.kruskal(*groups)
        except Exception:
            H, p = np.nan, np.nan

        eta2 = _eta_squared_groups(groups)

        rows.append({
            "condition":      cond,
            "H":              float(H),
            "p_value":        float(p),
            "eta2":           float(eta2),
            "significant":    p < 0.05,
            "interpretation": _interpret_eta2(eta2),
        })

    return pd.DataFrame(rows).sort_values("eta2", ascending=False)


# =========================================================
# TEST DE PERMUTATION
# =========================================================

def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Test de permutation : les ratings groove sont-ils structurés
    dans l'espace latent au-delà du hasard ?

    Statistique de test : corrélation entre distance euclidienne
    dans l'espace X et différence de rating |y_i - y_j|.

    Args:
        X               : embeddings normalisés (n, d)
        y               : groove ratings (n,)
        n_permutations  : nombre de permutations

    Returns:
        dict { observed_r, p_value, permutation_dist }
    """
    rng = np.random.default_rng(seed)
    n   = len(y)

    # Matrice de distances euclidiennes
    diff_X = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    diff_y = np.abs(y[:, None] - y[None, :])

    # Indices triangulaire supérieur (évite doublons)
    triu = np.triu_indices(n, k=1)
    dist_X = diff_X[triu]
    dist_y = diff_y[triu]

    # Corrélation observée
    if dist_X.std() < 1e-10 or dist_y.std() < 1e-10:
        return {
            "observed_r":       0.0,
            "p_value":          1.0,
            "permutation_dist": [],
            "n_permutations":   n_permutations,
            "warning":          "Variance nulle — test non calculable",
        }

    observed_r, _ = stats.pearsonr(dist_X, dist_y)

    # Distribution nulle
    null_dist = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm    = rng.permutation(y)
        diff_perm = np.abs(y_perm[:, None] - y_perm[None, :])
        r_perm, _ = stats.pearsonr(dist_X, diff_perm[triu])
        null_dist[i] = r_perm

    p_value = float(np.mean(null_dist >= observed_r))

    return {
        "observed_r":       float(observed_r),
        "p_value":          p_value,
        "permutation_dist": null_dist.tolist(),
        "n_permutations":   n_permutations,
        "significant":      p_value < 0.05,
    }


# =========================================================
# STATS DESCRIPTIVES PAR CELLULE DU DESIGN
# =========================================================

def compute_condition_stats(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Statistiques descriptives de groove_mean par cellule du design.
    Utile pour les figures d'interaction.

    Returns:
        DataFrame groupé avec mean, std, sem, ci95, n
    """
    if condition_cols is None:
        condition_cols = [c for c in ["S_mv", "D_mv", "E", "P"] if c in df.columns]

    if not condition_cols:
        raise ValueError("Aucune colonne de condition trouvée")

    agg = (
        df.groupby(condition_cols)[groove_col]
        .agg(
            mean="mean",
            std="std",
            n="count",
        )
        .reset_index()
    )

    agg["sem"]  = agg["std"] / np.sqrt(agg["n"])
    agg["ci95"] = 1.96 * agg["sem"]

    return agg


# =========================================================
# POST-HOC PAIRWISE (Tukey HSD simplifié)
# =========================================================

def pairwise_comparisons(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_col: str = "S_mv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Comparaisons par paires entre niveaux d'une condition.
    Correction de Bonferroni appliquée.

    Returns:
        DataFrame avec level_a, level_b, mean_diff, t, p_bonferroni, significant
    """
    levels = sorted(df[condition_col].unique())
    groups = {
        lv: df.loc[df[condition_col] == lv, groove_col].dropna().values
        for lv in levels
    }

    n_comparisons = len(list(combinations(levels, 2)))
    rows = []

    for lv_a, lv_b in combinations(levels, 2):
        g_a, g_b = groups[lv_a], groups[lv_b]

        if len(g_a) < 2 or len(g_b) < 2:
            continue

        t, p = stats.ttest_ind(g_a, g_b, equal_var=False)
        p_bonf = min(p * n_comparisons, 1.0)

        rows.append({
            "level_a":       lv_a,
            "level_b":       lv_b,
            "mean_a":        float(g_a.mean()),
            "mean_b":        float(g_b.mean()),
            "mean_diff":     float(g_a.mean() - g_b.mean()),
            "t":             float(t),
            "p_raw":         float(p),
            "p_bonferroni":  float(p_bonf),
            "significant":   p_bonf < alpha,
        })

    return pd.DataFrame(rows)


# =========================================================
# HELPERS PRIVÉS
# =========================================================

def _eta_squared_groups(groups: list[np.ndarray]) -> float:
    """η² à partir d'une liste de groupes."""
    all_vals   = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_total   = np.sum((all_vals - grand_mean) ** 2)

    if ss_total < 1e-12:
        return 0.0

    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for g in groups
    )
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def _interpret_eta2(eta2: float) -> str:
    """Cohen 1988 : 0.01 petit, 0.06 moyen, 0.14 grand."""
    if eta2 < 0.01:
        return "négligeable"
    elif eta2 < 0.06:
        return "petit"
    elif eta2 < 0.14:
        return "moyen"
    else:
        return "grand"
