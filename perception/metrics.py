"""
perception/metrics.py
=====================
Métriques de perception pour l'analyse des ratings groove.

Fonctions :
    correlation_score         — Pearson r avec gestion des edge cases
    cluster_perception_diff   — séparation perceptive inter-clusters (stats complètes)
    effect_size_eta2          — η² (effect size ANOVA) entre clusters
    perception_summary        — résumé complet des ratings d'un dataset
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, f_oneway, kruskal


# =========================================================
# CORRÉLATION
# =========================================================

def correlation_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcule le coefficient de corrélation de Pearson entre y_true et y_pred.

    Gère les edge cases :
        - n < 3 → retourne 0.0 (Pearson non défini)
        - variance nulle (signal constant) → retourne 0.0
        - NaN dans les inputs → retourne 0.0

    Args:
        y_true : ratings observés, shape (n,)
        y_pred : ratings prédits,  shape (n,)

    Returns:
        r : float dans [-1, 1], ou 0.0 si non calculable
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Masque NaN commun
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 3:
        return 0.0

    if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
        return 0.0

    r, _ = pearsonr(y_true, y_pred)
    return float(r)


# =========================================================
# SÉPARATION INTER-CLUSTERS
# =========================================================

def cluster_perception_diff(
    labels:  np.ndarray,
    ratings: np.ndarray,
) -> dict[int, dict]:
    """
    Mesure la séparation perceptive entre clusters.

    Pour chaque cluster retourne :
        mean    — moyenne groove_mean
        std     — écart-type
        n       — effectif
        ci95    — intervalle de confiance 95% (± 1.96 * SEM)

    En plus des statistiques par cluster, retourne des tests globaux :
        anova_p      — p-value ANOVA one-way (si hypothèses vérifiées)
        kruskal_p    — p-value Kruskal-Wallis (non-paramétrique, plus robuste)
        eta2         — η² (effect size, entre 0 et 1)

    Args:
        labels  : cluster labels, shape (n,)  — -1 ignoré (bruit DBSCAN)
        ratings : groove_mean par stimulus, shape (n,)

    Returns:
        dict {
            "clusters": {0: {mean, std, n, ci95}, 1: …},
            "anova_p":  float,
            "kruskal_p": float,
            "eta2":     float,
        }
    """
    labels  = np.asarray(labels,  dtype=np.int64)
    ratings = np.asarray(ratings, dtype=np.float64)

    unique_labels = [c for c in np.unique(labels) if c != -1]

    if len(unique_labels) < 2:
        return {
            "clusters":  {},
            "anova_p":   np.nan,
            "kruskal_p": np.nan,
            "eta2":      np.nan,
            "warning":   "Moins de 2 clusters valides — tests non calculables",
        }

    # ── Stats par cluster ─────────────────────────────────
    groups: dict[int, np.ndarray] = {}
    cluster_stats: dict[int, dict] = {}

    for c in unique_labels:
        vals = ratings[labels == c]
        groups[int(c)] = vals

        n   = len(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0

        cluster_stats[int(c)] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if n > 1 else 0.0,
            "n":    int(n),
            "ci95": float(1.96 * sem),
        }

    # ── Tests statistiques globaux ────────────────────────
    group_arrays = [groups[c] for c in sorted(groups.keys())]

    # ANOVA one-way (suppose normalité et homoscédasticité)
    try:
        _, anova_p = f_oneway(*group_arrays)
        anova_p = float(anova_p)
    except Exception:
        anova_p = np.nan

    # Kruskal-Wallis (non-paramétrique — plus robuste pour petits effectifs)
    try:
        _, kruskal_p = kruskal(*group_arrays)
        kruskal_p = float(kruskal_p)
    except Exception:
        kruskal_p = np.nan

    # η² (eta squared) — effect size ANOVA
    eta2 = float(_eta_squared(ratings[labels != -1], labels[labels != -1]))

    return {
        "clusters":  cluster_stats,
        "anova_p":   anova_p,
        "kruskal_p": kruskal_p,
        "eta2":      eta2,
    }


# =========================================================
# RÉSUMÉ GLOBAL
# =========================================================

def perception_summary(df) -> dict:
    """
    Résumé complet des ratings perceptifs d'un dataset.

    Args:
        df : DataFrame avec colonnes groove_mean (requis),
             complexity_mean et n_participants (optionnels)

    Returns:
        dict avec statistiques descriptives globales
    """
    import pandas as pd
    df = pd.DataFrame(df)

    if "groove_mean" not in df.columns:
        raise ValueError("df doit contenir 'groove_mean'")

    g = df["groove_mean"].dropna()

    summary: dict = {
        "n_stimuli":     int(len(g)),
        "groove_mean":   float(g.mean()),
        "groove_std":    float(g.std()),
        "groove_min":    float(g.min()),
        "groove_max":    float(g.max()),
        "groove_median": float(g.median()),
        "groove_q25":    float(g.quantile(0.25)),
        "groove_q75":    float(g.quantile(0.75)),
    }

    if "n_participants" in df.columns:
        summary["total_responses"]  = int(df["n_participants"].sum())
        summary["median_responses"] = float(df["n_participants"].median())

    if "complexity_mean" in df.columns:
        c = df["complexity_mean"].dropna()
        r = correlation_score(g.values, c.values)
        summary["groove_complexity_r"] = r

    return summary


def print_perception_summary(summary: dict) -> None:
    """Affiche le résumé perceptif dans le terminal."""
    w = 48
    print(f"\n{'─'*w}")
    print(f"  Résumé perceptif")
    print(f"{'─'*w}")
    print(f"  Stimuli évalués     : {summary['n_stimuli']}")
    if "total_responses" in summary:
        print(f"  Réponses totales    : {summary['total_responses']}")
        print(f"  Médiane / stimulus  : {summary['median_responses']:.1f}")
    print(f"  Groove mean         : {summary['groove_mean']:.2f} ± {summary['groove_std']:.2f}")
    print(f"  Groove range        : [{summary['groove_min']:.1f} – {summary['groove_max']:.1f}]")
    print(f"  Groove Q25–Q75      : [{summary['groove_q25']:.2f} – {summary['groove_q75']:.2f}]")
    if "groove_complexity_r" in summary:
        r = summary["groove_complexity_r"]
        print(f"  Groove × Complexity : r = {r:.3f}")
    print(f"{'─'*w}\n")


# =========================================================
# HELPER PRIVÉ
# =========================================================

def _eta_squared(y: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcule η² (eta squared) — proportion de variance expliquée par les clusters.

    η² = SS_between / SS_total
    Interprétation : 0.01 = petit, 0.06 = moyen, 0.14 = grand (Cohen 1988)
    """
    y = np.asarray(y, dtype=np.float64)

    grand_mean = y.mean()
    ss_total   = np.sum((y - grand_mean) ** 2)

    if ss_total < 1e-12:
        return 0.0

    ss_between = 0.0
    for c in np.unique(labels):
        group = y[labels == c]
        ss_between += len(group) * (group.mean() - grand_mean) ** 2

    return float(np.clip(ss_between / ss_total, 0.0, 1.0))

def effect_size_eta2(y: np.ndarray, labels: np.ndarray) -> float:
    """
    Public wrapper for eta squared effect size.
    """
    return _eta_squared(y, labels)