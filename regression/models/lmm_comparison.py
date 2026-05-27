"""
regression/models/lmm_comparison.py
=====================================
Comparaison formelle de deux LMM via AIC/BIC (likelihood ratio test).

Utilisation :
    from regression.models.lmm_comparison import compare_lmm_models

    result = compare_lmm_models(df_raw, features_base, features_interactions)

Critères d'interprétation (Burnham & Anderson 2002) :
    ΔAIC < 2   : différence négligeable
    ΔAIC 2–10  : amélioration substantielle du modèle avec interactions
    ΔAIC > 10  : amélioration très forte

Note sur REML vs ML :
    La comparaison AIC/BIC de modèles avec des effets fixes différents
    doit être faite en ML (pas REML). On utilise ML ici uniquement
    pour la comparaison — les coefficients rapportés ailleurs restent REML.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


BACKGROUND_REF    = "non_musician"
BACKGROUND_LEVELS = ["amateur", "semi_pro", "pro"]

INTERACTION_TERMS = {
    "D_sq": ("D", "D"),
    "DxP":  ("D", "P"),
    "SxE":  ("S", "E"),
    "DxS":  ("D", "S"),
}


def compare_lmm_models(
    df_raw:               pd.DataFrame,
    features_base:        list[str],
    features_interaction: list[str],
    verbose:              bool = True,
) -> dict:
    """
    Fit deux LMM en ML et compare leurs AIC/BIC.

    Modèle M0 : effets fixes = features_base (sans interactions)
    Modèle M1 : effets fixes = features_interaction (avec D_sq, DxP, SxE, DxS)

    Retourne un dict avec AIC/BIC des deux modèles, ΔAIC, ΔBIC,
    et une interprétation textuelle.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  [LMM compare] statsmodels absent")
        return {}

    df = df_raw.copy()

    # ── Normalisation features de base ───────────────────────────────────────
    base_cols = [f for f in features_interaction if f not in INTERACTION_TERMS]
    for col in base_cols:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 1e-10:
            df[col] = (df[col] - mu) / sigma

    # ── Calcul des termes d'interaction après normalisation ───────────────────
    for term, (f1, f2) in INTERACTION_TERMS.items():
        if f1 in df.columns and f2 in df.columns:
            df[term] = df[f1].values * df[f2].values

    # ── Background dummies ────────────────────────────────────────────────────
    bg_dummies = []
    if "musical_background" in df.columns:
        df["musical_background"] = df["musical_background"].astype(str)
        for lvl in BACKGROUND_LEVELS:
            col = f"bg_{lvl}"
            df[col] = (df["musical_background"] == lvl).astype(float)
            if df[col].sum() >= 3:
                bg_dummies.append(col)

    df = df.dropna(subset=["groove"])

    # ── Formules ─────────────────────────────────────────────────────────────
    # M0 : features de base seulement (orthogonalisées : D, S, E, P + génératifs)
    base_fixed  = [f for f in features_base if f in df.columns and f not in INTERACTION_TERMS]
    base_fixed += bg_dummies
    formula_m0  = "groove ~ " + " + ".join(base_fixed)

    # M1 : M0 + termes d'interaction
    inter_fixed  = list(base_fixed)
    inter_fixed += [t for t in INTERACTION_TERMS if t in df.columns and t not in inter_fixed]
    formula_m1   = "groove ~ " + " + ".join(inter_fixed)

    # ── Fit ML (pas REML pour comparaison de modèles à effets fixes différents) ─
    def _fit_ml(formula: str) -> object | None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm(formula, data=df, groups=df["participant_id"])
                try:
                    return m.fit(reml=False, method="lbfgs")
                except Exception:
                    return m.fit(reml=False, method="powell")
        except Exception as e:
            print(f"  [LMM compare] erreur fit : {e}")
            return None

    if verbose:
        print("\n  [LMM compare] Fit M0 (sans interactions, ML)…")
    r0 = _fit_ml(formula_m0)

    if verbose:
        print("  [LMM compare] Fit M1 (avec interactions, ML)…")
    r1 = _fit_ml(formula_m1)

    if r0 is None or r1 is None:
        print("  [LMM compare] Échec d'un des fits — comparaison impossible")
        return {}

    aic0 = float(r0.aic) if np.isfinite(r0.aic) else float("nan")
    bic0 = float(r0.bic) if np.isfinite(r0.bic) else float("nan")
    aic1 = float(r1.aic) if np.isfinite(r1.aic) else float("nan")
    bic1 = float(r1.bic) if np.isfinite(r1.bic) else float("nan")

    delta_aic = aic0 - aic1   # positif = M1 meilleur
    delta_bic = bic0 - bic1

    # ── Interprétation ───────────────────────────────────────────────────────
    def _interp(delta: float, criterion: str) -> str:
        if np.isnan(delta):
            return "indisponible (nan)"
        if delta < 0:
            return f"M0 préféré (Δ{criterion}={delta:.1f} — interactions pénalisées)"
        elif delta < 2:
            return f"différence négligeable (Δ{criterion}={delta:.1f})"
        elif delta < 10:
            return f"amélioration substantielle (Δ{criterion}={delta:.1f} ★)"
        else:
            return f"amélioration très forte (Δ{criterion}={delta:.1f} ★★)"

    result = {
        "aic_m0":       aic0,
        "bic_m0":       bic0,
        "aic_m1":       aic1,
        "bic_m1":       bic1,
        "delta_aic":    delta_aic,
        "delta_bic":    delta_bic,
        "formula_m0":   formula_m0,
        "formula_m1":   formula_m1,
        "n_obs":        int(len(df)),
        "converged_m0": bool(r0.converged),
        "converged_m1": bool(r1.converged),
        "interp_aic":   _interp(delta_aic, "AIC"),
        "interp_bic":   _interp(delta_bic, "BIC"),
    }

    if verbose:
        _print_comparison(result)

    return result


def _print_comparison(r: dict) -> None:
    w = 64
    print(f"\n{'─'*w}")
    print(f"  Comparaison LMM — M0 (additif) vs M1 (interactions)")
    print(f"{'─'*w}")
    print(f"  Estimation : ML  (requis pour comparer effets fixes)")
    print(f"  n obs      : {r['n_obs']}")
    print(f"  M0 conv.   : {'✔' if r['converged_m0'] else '⚠️'}")
    print(f"  M1 conv.   : {'✔' if r['converged_m1'] else '⚠️'}")
    print()
    print(f"  {'Critère':<10} {'M0 (base)':<14} {'M1 (interact.)':<14} {'Δ (M0−M1)':<12} Interprétation")
    print(f"  {'─'*56}")

    def _fmt(v): return f"{v:.1f}" if np.isfinite(v) else "nan"

    print(
        f"  {'AIC':<10} {_fmt(r['aic_m0']):<14} {_fmt(r['aic_m1']):<14} "
        f"{_fmt(r['delta_aic']):<12} {r['interp_aic']}"
    )
    print(
        f"  {'BIC':<10} {_fmt(r['bic_m0']):<14} {_fmt(r['bic_m1']):<14} "
        f"{_fmt(r['delta_bic']):<12} {r['interp_bic']}"
    )
    print()
    print(f"  M0 : {r['formula_m0'][:80]}…" if len(r['formula_m0']) > 80 else f"  M0 : {r['formula_m0']}")
    print(f"  M1 : {r['formula_m1'][:80]}…" if len(r['formula_m1']) > 80 else f"  M1 : {r['formula_m1']}")
    print(f"\n  Δ > 0 → M1 meilleur · Δ > 2 = substantiel · Δ > 10 = très fort")
    print(f"  Référence : Burnham & Anderson (2002)")
    print(f"{'─'*w}\n")
