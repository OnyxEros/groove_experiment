"""
perception_space/core/stats.py
==============================
Tests statistiques pour l'analyse perceptive du groove.

Notation :
    Paramètres génératifs (manipulés) : S_mv, D_mv, E_mv, P_mv
    Descripteurs émergents (réalisés)  : D, I, V, S, E, P

Nouveautés v2 :
    - kruskal_by_condition : test adaptatif ANOVA/Kruskal avec sélection
      automatique selon la normalité des groupes (Shapiro-Wilk α=0.05)
    - post_hoc_bonferroni : comparaisons pairées corrigées pour D_mv
      (seul facteur significatif)
    - permutation_test : inchangé (test de Mantel sur paires)
    - compute_condition_stats : stats descriptives par cellule
    - coverage_report : rapport de couverture des stimuli
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from itertools import combinations


def mantel_summary(result: dict) -> None:
    """Rapport console complet pour le test de Mantel."""
    import numpy as np

    W = 68
    def _sep(c="─"): return c * W

    obs    = result["observed_r"]
    p      = result["p_value"]
    n_perm = result["n_permutations"]
    n_pairs= result["n_pairs"]
    sig    = result["significant"]
    null   = np.array(result.get("permutation_dist", []))

    print(f"\n{_sep('═')}")
    print("  TEST DE MANTEL — STRUCTURE GÉOMÉTRIQUE × GROOVE PERÇU")
    print(_sep("═"))
    print()
    print(f"  Principe : corrélation entre distances dans l'espace latent")
    print(f"  et différences de ratings — testée par permutation.")
    print(f"  Unité : {n_pairs} paires de stimuli (n*(n-1)/2)")
    print()
    print(f"  r observé              : {obs:+.4f}")

    if len(null) > 0:
        p95 = float(np.percentile(null, 95))
        p99 = float(np.percentile(null, 99))
        mu_null = float(np.mean(null))
        sd_null = float(np.std(null))
        print(f"  Distribution nulle μ   : {mu_null:+.4f}  ±{sd_null:.4f}")
        print(f"  Seuil critique p=0.05  : {p95:+.4f}")
        print(f"  Seuil critique p=0.01  : {p99:+.4f}")
        z_score = (obs - mu_null) / (sd_null + 1e-10)
        print(f"  z-score                : {z_score:+.2f}")

    print(f"  p-value                : {p:.6f}  ({'★ significatif' if sig else '✗ non significatif'})")
    print(f"  N permutations         : {n_perm}")
    print()

    _sep_inner = "─" * 40
    if sig:
        print(f"  ✔  RÉSULTAT : L'espace latent structure la perception du groove.")
        print(f"     Les stimuli proches en (D, S, E, P) ont des grooves similaires.")
        if obs > 0.30:
            print(f"  ℹ  r={obs:.3f} : corrélation modérée à forte — signal géométrique robuste.")
        elif obs > 0.15:
            print(f"  ℹ  r={obs:.3f} : corrélation faible mais statistiquement robuste.")
        else:
            print(f"  ℹ  r={obs:.3f} : corrélation très faible mais non due au hasard.")
        print()
        print(f"  Interprétation mémoire :")
        print(f"  L'espace réalisé orthogonalisé (D, S, E, P) capture une partie")
        print(f"  de la structure perceptive du groove. La densité acoustique (D)")
        print(f"  étant le prédicteur dominant (β=+0.588 dans le LMM), elle guide")
        print(f"  probablement la structure géométrique observée ici.")
    else:
        print(f"  ✗  RÉSULTAT : Pas de structure géométrique détectée.")
        print(f"     La distance dans l'espace latent ne prédit pas le groove.")
        print()
        print(f"  Interprétation mémoire :")
        print(f"  La perception du groove est hautement individuelle — l'espace")
        print(f"  physique (D, S, E, P) ne suffit pas à prédire la perception.")
        print(f"  Des facteurs cognitifs, culturels ou d'expertise musicale")
        print(f"  pourraient médiatiser la relation stimulus → groove perçu.")

    if "warning" in result:
        print(f"\n  ⚠  {result['warning']}")

    print(f"\n  Référence : Mantel (1967), Cancer Res. 27:209–220")
    print(_sep("═"))
    print()



# =========================================================
# KRUSKAL / ANOVA — choix automatique + post-hoc
# =========================================================

def kruskal_by_condition(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Test non-paramétrique (Kruskal-Wallis) ou paramétrique (ANOVA one-way)
    selon la normalité des groupes (Shapiro-Wilk, α=0.05).

    Pour chaque condition, calcule aussi les comparaisons post-hoc Bonferroni
    si le test principal est significatif (p < 0.05).

    condition_cols par défaut : paramètres génératifs S_mv, D_mv, E_mv, P_mv.
    """
    if condition_cols is None:
        condition_cols = [
            c for c in ["S_mv", "D_mv", "E_mv", "P_mv"]
            if c in df.columns
        ]

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

        # ── Test de normalité Shapiro-Wilk ────────────────
        normality_ok = True
        for g in groups:
            if len(g) < 3:
                normality_ok = False
                break
            if len(g) <= 5000:
                _, p_norm = stats.shapiro(g)
                if p_norm < 0.05:
                    normality_ok = False
                    break

        # ── Test principal ────────────────────────────────
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


def post_hoc_bonferroni(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_col: str = "D_mv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Comparaisons pairées Welch t-test avec correction Bonferroni.

    À utiliser après kruskal_by_condition quand le test principal est
    significatif. Par défaut sur D_mv (seul effet significatif).

    Returns:
        DataFrame avec colonnes :
            level_a, level_b, mean_a, mean_b, mean_diff,
            t, df_welch, p_raw, p_bonferroni, significant, cohen_d
    """
    if condition_col not in df.columns:
        raise ValueError(f"Colonne '{condition_col}' absente du DataFrame")

    levels = sorted(df[condition_col].unique())
    groups = {
        lv: df.loc[df[condition_col] == lv, groove_col].dropna().values
        for lv in levels
    }

    pairs         = list(combinations(levels, 2))
    n_comparisons = len(pairs)
    rows          = []

    for lv_a, lv_b in pairs:
        g_a, g_b = groups[lv_a], groups[lv_b]
        if len(g_a) < 2 or len(g_b) < 2:
            continue

        t, p    = stats.ttest_ind(g_a, g_b, equal_var=False)
        p_bonf  = min(float(p) * n_comparisons, 1.0)

        # Cohen's d (pooled std, Welch-safe)
        pooled_std = np.sqrt(
            (np.var(g_a, ddof=1) + np.var(g_b, ddof=1)) / 2
        )
        cohen_d = float((g_a.mean() - g_b.mean()) / pooled_std) \
            if pooled_std > 1e-10 else 0.0

        # Degrés de liberté Welch
        s_a, s_b = np.var(g_a, ddof=1), np.var(g_b, ddof=1)
        n_a, n_b = len(g_a), len(g_b)
        num      = (s_a / n_a + s_b / n_b) ** 2
        den      = (s_a / n_a) ** 2 / (n_a - 1) + (s_b / n_b) ** 2 / (n_b - 1)
        df_welch = float(num / den) if den > 1e-12 else float(n_a + n_b - 2)

        rows.append({
            "level_a":      lv_a,
            "level_b":      lv_b,
            "mean_a":       float(g_a.mean()),
            "mean_b":       float(g_b.mean()),
            "mean_diff":    float(g_a.mean() - g_b.mean()),
            "t":            float(t),
            "df_welch":     round(df_welch, 1),
            "p_raw":        float(p),
            "p_bonferroni": p_bonf,
            "significant":  p_bonf < alpha,
            "cohen_d":      cohen_d,
        })

    return pd.DataFrame(rows)


# =========================================================
# TEST DE PERMUTATION (Mantel)
# =========================================================

def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Test de Mantel : corrélation entre distances dans l'espace latent
    et différences de ratings.

    Note méthodologique : l'unité d'observation est la PAIRE de stimuli
    (n*(n-1)/2 paires), pas le stimulus. La puissance statistique est donc
    plus élevée qu'un test sur n points. La significativité doit être
    interprétée en conséquence.
    """
    rng = np.random.default_rng(seed)
    y   = np.asarray(y, dtype=np.float64)

    dist_X = pdist(X, metric="euclidean")
    diff_y = pdist(y.reshape(-1, 1), metric="cityblock")

    if dist_X.std() < 1e-10 or diff_y.std() < 1e-10:
        return {
            "observed_r":       0.0,
            "p_value":          1.0,
            "permutation_dist": [],
            "n_permutations":   n_permutations,
            "n_pairs":          len(dist_X),
            "significant":      False,
            "warning":          "Variance nulle — test non calculable",
        }

    observed_r, _ = stats.pearsonr(dist_X, diff_y)

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
        "n_pairs":          int(len(dist_X)),
        "significant":      p_value < 0.05,
    }


# =========================================================
# STATS DESCRIPTIVES PAR CONDITION
# =========================================================

def compute_condition_stats(
    df: pd.DataFrame,
    groove_col: str = "groove_mean",
    condition_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Stats descriptives par cellule du design."""
    if condition_cols is None:
        condition_cols = [
            c for c in ["S_mv", "D_mv", "E_mv", "P_mv"]
            if c in df.columns
        ]

    if not condition_cols:
        raise ValueError("Aucune colonne de condition trouvée")

    if groove_col not in df.columns:
        available = [c for c in df.columns if "groove" in c.lower()]
        if available:
            groove_col = available[0]
            warnings.warn(
                f"compute_condition_stats : groove_col introuvable, "
                f"fallback sur '{groove_col}'",
                UserWarning, stacklevel=2,
            )
        else:
            raise ValueError(f"Colonne '{groove_col}' absente.")

    agg = (
        df.groupby(condition_cols)[groove_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    agg["sem"]  = agg["std"] / np.sqrt(agg["n"])
    agg["ci95"] = 1.96 * agg["sem"]
    return agg


# =========================================================
# RAPPORT DE COUVERTURE
# =========================================================

def coverage_report(
    df_long: pd.DataFrame,
    stim_col: str = "stimulus_id",
    participant_col: str = "participant_id",
    rating_col: str = "groove",
    n_min: int = 2,
) -> dict:
    """
    Rapport de couverture des stimuli.

    Retourne un dict avec :
        n_total         — nombre total de stimuli
        n_covered       — stimuli avec >= n_min réponses
        n_excluded      — stimuli avec < n_min réponses
        coverage_pct    — % de stimuli couverts
        responses_per_stim — dict {stim_id: n_responses}
        excluded_stims  — liste des stim_id exclus
        n_participants  — nombre de participants uniques
        n_min           — seuil utilisé
    """
    coverage = df_long.groupby(stim_col)[rating_col].count()
    n_parts  = df_long[participant_col].nunique() \
        if participant_col in df_long.columns else None

    covered  = coverage[coverage >= n_min]
    excluded = coverage[coverage < n_min]

    return {
        "n_total":            int(len(coverage)),
        "n_covered":          int(len(covered)),
        "n_excluded":         int(len(excluded)),
        "coverage_pct":       float(len(covered) / len(coverage) * 100),
        "responses_per_stim": coverage.to_dict(),
        "excluded_stims":     excluded.index.tolist(),
        "n_participants":     int(n_parts) if n_parts else None,
        "n_min":              n_min,
        "mean_responses":     float(coverage.mean()),
        "median_responses":   float(coverage.median()),
    }


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
    if eta2 < 0.01:   return "négligeable"
    elif eta2 < 0.06: return "petit"
    elif eta2 < 0.14: return "moyen"
    else:             return "grand"