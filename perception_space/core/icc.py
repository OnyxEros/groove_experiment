"""
perception_space/core/icc.py
============================
Intraclass Correlation Coefficient (ICC) inter-participants.

Implémente ICC(2,1) — two-way random, single measures, absolute agreement.
Standard pour la fiabilité inter-juges en psychologie de la perception
(Shrout & Fleiss 1979, Koo & Mae 2016).

Interprétation (Koo & Mae 2016) :
    ICC < 0.50  : fiabilité faible
    0.50–0.75   : fiabilité modérée
    0.75–0.90   : fiabilité bonne
    > 0.90      : fiabilité excellente

Correction v2 :
    L'imputation des NaN par moyenne de ligne biaise l'ICC vers le haut
    (réduit la variance résiduelle artificiellement). Avec un design incomplet
    (tous les participants n'évaluent pas tous les stimuli), on utilise
    à la place une ANOVA à deux facteurs sur les données non-imputées
    via les sommes de carrés correctement calculées sur les cellules disponibles.

    Pour les IC95%, on conserve la formule Shrout & Fleiss mais on signale
    explicitement quand la matrice est très incomplète (< 50% de remplissage).
"""

from __future__ import annotations

import warnings
import numpy as np
import math
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
                  Les NaN représentent les cellules manquantes (design incomplet).
                  NE PAS imputer avant d'appeler cette fonction.
        model   : "ICC1" | "ICC2" | "ICC3"

    Returns:
        dict {
            icc, ci95_low, ci95_high, F, df1, df2, p_value,
            n_stimuli, n_raters, model, MS_r, MS_e,
            completeness_pct, interpretation, warning (si applicable)
        }
    """
    ratings = np.asarray(ratings, dtype=np.float64)

    if ratings.ndim != 2:
        raise ValueError(
            f"ratings doit être 2D (n_stimuli, n_participants), got {ratings.ndim}D"
        )

    n, k = ratings.shape

    if n < 3:
        raise ValueError(f"Minimum 3 stimuli requis, got {n}")
    if k < 2:
        raise ValueError(f"Minimum 2 participants requis, got {k}")

    # ── Diagnostic de complétude ──────────────────────────
    n_missing      = int(np.isnan(ratings).sum())
    n_cells        = n * k
    completeness   = float((n_cells - n_missing) / n_cells * 100)
    warning_msg    = None

    if completeness < 30:
        warning_msg = (
            f"Matrice très incomplète ({completeness:.0f}% de remplissage). "
            f"L'ICC est peu fiable — collecte plus de données."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
    elif completeness < 60:
        warning_msg = (
            f"Matrice incomplète ({completeness:.0f}% de remplissage). "
            f"L'IC95% est large — interpréter avec précaution."
        )

    # ── ANOVA sans imputation ─────────────────────────────
    # On travaille sur les cellules non-NaN directement.
    # SS_r calculé sur les moyennes par ligne (nanmean ignore les NaN).
    # SS_c calculé sur les moyennes par colonne (idem).
    # SS_e = SS_t - SS_r - SS_c (résiduel).
    #
    # Cette approche est équivalente à une ANOVA à deux facteurs
    # déséquilibrée (type I SS). Acceptable pour des designs faiblement
    # déséquilibrés. Pour un design très déséquilibré, préférer un modèle
    # mixte (lme4 en R), mais hors scope ici.

    grand_mean   = float(np.nanmean(ratings))
    row_means    = np.nanmean(ratings, axis=1)    # shape (n,)
    col_means    = np.nanmean(ratings, axis=0)    # shape (k,)
    row_counts   = np.sum(~np.isnan(ratings), axis=1)  # réponses par stimulus
    col_counts   = np.sum(~np.isnan(ratings), axis=0)  # réponses par participant

    # SS between rows (stimuli) — pondéré par le nombre de réponses
    SS_r = float(np.sum(row_counts * (row_means - grand_mean) ** 2))
    df_r = n - 1

    # SS between columns (participants) — pondéré
    SS_c = float(np.sum(col_counts * (col_means - grand_mean) ** 2))
    df_c = k - 1

    # SS total — sur les cellules non-NaN
    SS_t = float(np.nansum((ratings - grand_mean) ** 2))
    df_t = int(n_cells - n_missing) - 1

    # SS error
    SS_e = max(SS_t - SS_r - SS_c, 0.0)  # clip à 0 pour éviter MS_e négatif
    df_e = max(df_t - df_r - df_c, 1)

    MS_r = SS_r / df_r if df_r > 0 else 1e-10
    MS_c = SS_c / df_c if df_c > 0 else 1e-10
    MS_e = SS_e / df_e if df_e > 0 else 1e-10

    # ── ICC selon le modèle ───────────────────────────────
    k_bar = float(np.mean(row_counts))  # nombre moyen de raters par stimulus

    if model == "ICC1":
        icc_val = (MS_r - MS_e) / (MS_r + (k_bar - 1) * MS_e)
        F_val   = MS_r / MS_e
        df1, df2 = df_r, df_t - df_r

    elif model == "ICC2":
        icc_val = (MS_r - MS_e) / (
            MS_r + (k_bar - 1) * MS_e + k_bar * (MS_c - MS_e) / n
        )
        F_val   = MS_r / MS_e
        df1, df2 = df_r, df_e

    elif model == "ICC3":
        icc_val = (MS_r - MS_e) / (MS_r + (k_bar - 1) * MS_e)
        F_val   = MS_r / MS_e
        df1, df2 = df_r, df_e

    else:
        raise ValueError(f"model doit être ICC1, ICC2 ou ICC3, got '{model}'")

    icc_val = float(np.clip(icc_val, -1.0, 1.0))

    # ── p-value ───────────────────────────────────────────
    p_value = float(1 - stats.f.cdf(F_val, df1, df2))

    # ── IC95% Shrout & Fleiss 1979 ────────────────────────
    alpha   = 0.05
    F_lower = F_val / stats.f.ppf(1 - alpha / 2, df1, df2)
    F_upper = F_val * stats.f.ppf(1 - alpha / 2, df2, df1)

    ci_low  = float(np.clip((F_lower - 1) / (F_lower + k_bar - 1), -1.0, 1.0))
    ci_high = float(np.clip((F_upper - 1) / (F_upper + k_bar - 1), -1.0, 1.0))

    result = {
        "icc":              icc_val,
        "ci95_low":         ci_low,
        "ci95_high":        ci_high,
        "F":                float(F_val),
        "df1":              int(df1),
        "df2":              int(df_e),
        "p_value":          p_value,
        "n_stimuli":        int(n),
        "n_raters":         int(k),
        "k_bar":            round(float(k_bar), 2),
        "model":            model,
        "MS_r":             float(MS_r),
        "MS_e":             float(MS_e),
        "completeness_pct": round(completeness, 1),
        "interpretation":   _interpret_icc(icc_val),
    }

    if warning_msg:
        result["warning"] = warning_msg

    return result


# =========================================================
# ICC PAR STIMULUS
# =========================================================

def compute_per_stimulus_variance(
    ratings_long: pd.DataFrame,
    stim_col: str = "stimulus_id",
    rating_col: str = "groove",
    participant_col: str = "participant_id",
    n_min: int = 2,
) -> pd.DataFrame:
    """
    Variabilité inter-participants par stimulus.

    Retourne uniquement les stimuli avec >= n_min réponses.
    Colonnes : stim_col, mean, std, cv, iqr, n_raters
    Trié par std décroissant (stimuli les plus ambigus en premier).
    """
    rows = []
    for stim_id, group in ratings_long.groupby(stim_col):
        vals = group[rating_col].dropna().values
        if len(vals) < n_min:
            continue
        rows.append({
            stim_col:   stim_id,
            "mean":     float(np.mean(vals)),
            "std":      float(np.std(vals, ddof=1)),
            "cv":       float(np.std(vals, ddof=1) / (np.mean(vals) + 1e-9)),
            "iqr":      float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "n_raters": int(len(vals)),
        })
    return pd.DataFrame(rows).sort_values("std", ascending=False)


# =========================================================
# WIDE FORMAT
# =========================================================

def ratings_to_wide(
    ratings_long: pd.DataFrame,
    stim_col: str = "stimulus_id",
    participant_col: str = "participant_id",
    rating_col: str = "groove",
    n_min: int = 2,
) -> np.ndarray:
    """
    Convertit en matrice wide (n_stimuli × n_participants).
    Exclut les stimuli avec < n_min réponses.
    Cellules manquantes → NaN (ne pas imputer).
    """
    # Filtre couverture minimale
    coverage = ratings_long.groupby(stim_col)[rating_col].count()
    valid    = coverage[coverage >= n_min].index
    df       = ratings_long[ratings_long[stim_col].isin(valid)]

    pivot = df.pivot_table(
        index=stim_col,
        columns=participant_col,
        values=rating_col,
        aggfunc="mean",
    )
    return pivot.values.astype(np.float64)


# =========================================================
# HELPERS
# =========================================================

def _interpret_icc(icc: float) -> str:
    if icc < 0:     return "négatif (variance résiduelle > variance inter-stimuli)"
    elif icc < 0.50: return "faible"
    elif icc < 0.75: return "modérée"
    elif icc < 0.90: return "bonne"
    else:            return "excellente"


def icc_summary(result: dict, label: str = "Groove") -> None:
    """
    Rapport console complet et très détaillé pour l'ICC.
    Inclut zones de référence, diagnostic, interprétation narrative.
    """
    W = 68

    def _sep(c="─"): return c * W
    def _bar(v, w=22):
        v = max(min(float(v), 1.0), 0.0)
        n = int(v * w)
        return "█" * n + "░" * (w - n)
    def _pstars(p):
        if p < 0.001: return "★★★ p<0.001"
        if p < 0.01:  return "★★  p<0.01"
        if p < 0.05:  return "★   p<0.05"
        if p < 0.10:  return "†   p<0.10"
        return "    n.s."

    icc  = result["icc"]
    low  = result["ci95_low"]
    high = result["ci95_high"]
    p    = result["p_value"]
    F    = result["F"]
    df1  = result["df1"]
    df2  = result["df2"]
    ns   = result["n_stimuli"]
    nr   = result["n_raters"]
    kb   = result["k_bar"]
    comp = result["completeness_pct"]
    ms_r = result["MS_r"]
    ms_e = result["MS_e"]
    interp = result["interpretation"]
    model  = result.get("model", "ICC2")

    print(f"\n{_sep('═')}")
    print(f"  ICC INTER-PARTICIPANTS  [{model}] — {label.upper()}")
    print(_sep("═"))

    # ── Design ───────────────────────────────────────────────────────────────
    print(f"\n  Structure des données")
    print(_sep())
    print(f"  Stimuli évalués        : {ns}")
    print(f"  Participants (raters)  : {nr}")
    print(f"  Réponses/stimulus (k̄)  : {kb:.2f}")
    print(f"  Complétude matricielle : {comp:.1f}%  "
          + ("⚠  matrice très incomplète" if comp < 30 else
             "⚠  matrice incomplète" if comp < 60 else "✔  complétude correcte"))
    print(f"  Total réponses         : ~{int(ns * kb)}")

    # ── Statistiques ANOVA ────────────────────────────────────────────────────
    print(f"\n  Composantes ANOVA (sans imputation des NaN)")
    print(_sep())
    print(f"  MS_r (between stimuli) : {ms_r:.4f}")
    print(f"  MS_e (résiduel)        : {ms_e:.4f}")
    print(f"  Rapport MS_r/MS_e      : {ms_r/ms_e:.3f}  → F({df1}, {df2}) = {F:.3f}")
    print(f"  p-value                : {p:.6f}  {_pstars(p)}")
    print()
    if ms_r > ms_e:
        print(f"  ✔  MS_r > MS_e : variance inter-stimuli > résiduelle")
        print(f"     → les stimuli diffèrent systématiquement en groove perçu")
    else:
        print(f"  ⚠  MS_r ≤ MS_e : variance inter-stimuli ≤ résiduelle")
        print(f"     → les stimuli ne se distinguent pas clairement en groove perçu")

    # ── ICC ───────────────────────────────────────────────────────────────────
    print(f"\n  ICC et intervalle de confiance à 95%")
    print(_sep())
    print(f"  ICC(2,1) = {icc:.4f}")
    print(f"  IC95%    = [{low:.4f} – {high:.4f}]")
    print(f"  Largeur IC95%          : {high-low:.4f}  "
          + ("(large — données insuffisantes)" if high-low > 0.3 else "(acceptable)"))
    print()
    print(f"  Jauge  [0 ────────── 0.5 ─────────── 1.0]")
    print(f"         [{_bar(icc, 22)}]  {icc*100:.1f}%")
    print()

    # Zones de référence
    zones = [
        (0.90, 1.00, "Excellente ",  "██████████"),
        (0.75, 0.90, "Bonne      ",  "████████  "),
        (0.50, 0.75, "Modérée    ",  "█████     "),
        (0.00, 0.50, "Faible     ",  "██        "),
    ]
    print(f"  Zones de référence (Koo & Mae, 2016) :")
    for lo, hi, lbl, _ in zones:
        marker = "◄ VOTRE ICC" if lo <= icc < hi else "          "
        print(f"    [{lo:.2f} – {hi:.2f}]  {lbl}  {marker}")

    # ── Interprétation ────────────────────────────────────────────────────────
    print(f"\n  Interprétation")
    print(_sep())
    print(f"  Fiabilité : {interp.upper()}")
    print()

    if icc < 0:
        print(f"  ⚠  ICC négatif : variance résiduelle > variance inter-stimuli.")
        print(f"     Les participants varient plus entre eux qu'entre les stimuli.")
        print(f"     → Problème de design ou de collecte des données.")
    elif icc < 0.50:
        print(f"  ⚠  Fiabilité FAIBLE : les participants s'accordent peu.")
        print(f"     La perception du groove est très idiosyncratique.")
        print(f"     Les analyses de groupe doivent être interprétées avec précaution.")
        print(f"     → Explorer les différences de bagage musical (bg_amateur, etc.)")
    elif icc < 0.75:
        print(f"  ℹ  Fiabilité MODÉRÉE : accord partiel entre participants.")
        print(f"     Le signal perceptif existe mais est bruité.")
        print(f"     Acceptable pour l'exploration, insuffisant pour diagnostic clinique.")
    elif icc < 0.90:
        print(f"  ✔  Fiabilité BONNE : les participants s'accordent bien.")
        print(f"     Les ratings de groove sont reproductibles et fiables.")
        print(f"     Les analyses comparatives sont justifiées.")
    else:
        print(f"  ✔  Fiabilité EXCELLENTE : accord quasi-parfait.")
        print(f"     Le groove est perçu de manière très consensuelle dans ce corpus.")

    print()
    # Lien avec le LMM
    icc_lmm_note = (
        f"  Note : ICC_perception={icc:.3f} est distinct de l'ICC_LMM={0.103:.3f} "
        f"rapporté dans la régression.\n"
        f"  L'ICC_LMM mesure la variabilité participant→stimulus dans le modèle mixte,\n"
        f"  l'ICC_perception mesure la concordance entre juges sur les mêmes stimuli."
    )
    print(icc_lmm_note)

    print(f"\n  Référence : Koo & Mae (2016), J. Chiropr. Med. 15(2):155–163")
    print(f"             Shrout & Fleiss (1979), Psychol. Bull. 86(2):420–428")
    print(_sep())
    print()