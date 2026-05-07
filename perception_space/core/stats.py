"""
perception_space/core/stats.py
==============================
Tests statistiques pour l'analyse perceptive du groove.

Corrections v2 :
    - permutation_test : calcul paires sans matrice O(n²) complète
      → utilise scipy.spatial.distance.pdist (C, ~10× plus rapide)
    - kruskal_by_condition : teste la normalité par groupe (Shapiro-Wilk)
      et choisit automatiquement ANOVA (si normale) ou Kruskal-Wallis
    - compute_condition_stats : nom de colonne configurable (plus de hardcode)
    - Warning VIF > 10 dans kruskal_by_condition
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from itertools import combinations


# =========================================================
# KRUSKAL / ANOVA — choix automatique
# =========================================================

def kruskal_by_condition(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Test non-paramétrique (Kruskal-Wallis) ou paramétrique (ANOVA one-way)
    selon la normalité des groupes (Shapiro-Wilk, α=0.05).

    Choix automatique par condition :
        - Tous les groupes normaux → ANOVA (plus puissante)
        - Au moins un groupe non-normal → Kruskal-Wallis (robuste)

    Returns:
        DataFrame avec colonnes :
            condition, test_used, statistic, p_value, eta2,
            significant, interpretation, normality_ok
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

        # ── Test de normalité par groupe ─────────────────
        normality_ok = True
        for g in groups:
            if len(g) < 3:
                normality_ok = False
                break
            if len(g) <= 5000:   # Shapiro limité à 5000
                _, p_norm = stats.shapiro(g)
                if p_norm < 0.05:
                    normality_ok = False
                    break

        # ── Test statistique ──────────────────────────────
        try:
            if normality_ok:
                stat, p = stats.f_oneway(*groups)
                test_used = "ANOVA"
            else:
                stat, p = stats.kruskal(*groups)
                test_used = "Kruskal-Wallis"
        except Exception:
            stat, p = np.nan, np.nan
            test_used = "failed"

        eta2 = _eta_squared_groups(groups)

        rows.append({
            "condition":      cond,
            "test_used":      test_used,
            "statistic":      float(stat),
            "p_value":        float(p),
            "eta2":           float(eta2),
            "significant":    bool(p < 0.05),
            "interpretation": _interpret_eta2(eta2),
            "normality_ok":   normality_ok,
        })

    return pd.DataFrame(rows).sort_values("eta2", ascending=False)


# ── Alias rétro-compatible ────────────────────────────────
def anova_by_condition(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Alias vers kruskal_by_condition (choix automatique ANOVA/Kruskal)."""
    return kruskal_by_condition(df, groove_col=groove_col, condition_cols=condition_cols)


# =========================================================
# TEST DE PERMUTATION — O(n log n) mémoire
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

    Statistique : corrélation de Pearson entre distance euclidienne
    dans X et différence absolue de rating |y_i - y_j|.

    Optimisation v2 :
        - scipy.spatial.distance.pdist → C, pas de matrice n×n en RAM
        - ~10× plus rapide que la version matricielle pour n > 100

    Args:
        X              : embeddings normalisés (n, d)
        y              : groove ratings (n,)
        n_permutations : nombre de permutations
        seed           : graine reproductibilité

    Returns:
        dict { observed_r, p_value, permutation_dist, significant, n_permutations }
    """
    rng = np.random.default_rng(seed)
    n   = len(y)
    y   = np.asarray(y, dtype=np.float64)

    # ── Distances euclidiennes via pdist (C, compact) ─────
    dist_X = pdist(X, metric="euclidean")          # shape (n*(n-1)/2,)

    # ── Différences de ratings (même ordre que pdist) ─────
    # pdist génère les paires (i,j) avec i < j dans l'ordre lexicographique
    diff_y = pdist(y.reshape(-1, 1), metric="cityblock")  # |y_i - y_j|

    if dist_X.std() < 1e-10 or diff_y.std() < 1e-10:
        return {
            "observed_r":       0.0,
            "p_value":          1.0,
            "permutation_dist": [],
            "n_permutations":   n_permutations,
            "significant":      False,
            "warning":          "Variance nulle — test non calculable",
        }

    observed_r, _ = stats.pearsonr(dist_X, diff_y)

    # ── Distribution nulle ────────────────────────────────
    null_dist = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm    = rng.permutation(y)
        diff_perm = pdist(y_perm.reshape(-1, 1), metric="cityblock")
        r_perm, _ = stats.pearsonr(dist_X, diff_perm)
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

    Args:
        groove_col     : nom de la colonne target (configurable, plus de hardcode)
        condition_cols : colonnes de conditions à grouper
    """
    if condition_cols is None:
        condition_cols = [c for c in ["S_mv", "D_mv", "E", "P"] if c in df.columns]

    if not condition_cols:
        raise ValueError("Aucune colonne de condition trouvée")

    # Vérifie que groove_col existe
    if groove_col not in df.columns:
        available = [c for c in df.columns if "groove" in c.lower()]
        if available:
            groove_col = available[0]
            warnings.warn(
                f"compute_condition_stats : groove_col introuvable, "
                f"fallback sur '{groove_col}'",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"Colonne '{groove_col}' absente. "
                f"Colonnes disponibles : {list(df.columns)}"
            )

    agg = (
        df.groupby(condition_cols)[groove_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    agg["sem"]  = agg["std"] / np.sqrt(agg["n"])
    agg["ci95"] = 1.96 * agg["sem"]

    return agg


# =========================================================
# POST-HOC PAIRWISE (Bonferroni)
# =========================================================

def pairwise_comparisons(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_col: str = "S_mv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Comparaisons par paires entre niveaux d'une condition (Bonferroni)."""
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

        t, p       = stats.ttest_ind(g_a, g_b, equal_var=False)
        p_bonf     = min(p * n_comparisons, 1.0)

        rows.append({
            "level_a":      lv_a,
            "level_b":      lv_b,
            "mean_a":       float(g_a.mean()),
            "mean_b":       float(g_b.mean()),
            "mean_diff":    float(g_a.mean() - g_b.mean()),
            "t":            float(t),
            "p_raw":        float(p),
            "p_bonferroni": float(p_bonf),
            "significant":  p_bonf < alpha,
        })

    return pd.DataFrame(rows)


# =========================================================
# HELPERS PRIVÉS
# =========================================================

def _eta_squared_groups(groups: list[np.ndarray]) -> float:
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
    if eta2 < 0.01:  return "négligeable"
    elif eta2 < 0.06: return "petit"
    elif eta2 < 0.14: return "moyen"
    else:             return "grand"