"""
regression/model.py
===================
Définition et entraînement des modèles de régression groove.

Modèles :
    Ridge         — linéaire régularisé, interprétable via coefficients
    RandomForest  — non-linéaire, interprétable via feature importances
    LMM           — modèle linéaire mixte avec effet aléatoire participant
                    (u_i ~ N(0, σ²)), fidèle au modèle formel du mémoire :

                    G_ij = β₀ + β₁·S_real + β₂·S_real² + β₃·D + β₄·I
                         + β₅·E_real + β₆·V + β₇·P_real + β₈·BPM
                         + u_i + ε_ij

                    Estimé par REML via statsmodels.MixedLM.

Note sur le LMM :
    Ridge et RF opèrent sur des ratings agrégés par stim_id (groove_mean),
    ce qui ignore la variabilité inter-participants. Le LMM opère sur les
    réponses brutes (une ligne par réponse), avec participant_id comme groupe
    d'effet aléatoire. Cela le rend plus fidèle au modèle décrit dans le mémoire,
    mais nécessite les données brutes — pas les données agrégées.
    Si les données brutes sont absentes (colonne participant_id manquante),
    le LMM est ignoré sans erreur.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# FIT — POINT D'ENTRÉE PRINCIPAL
# =========================================================

def fit_models(
    X:        np.ndarray,
    y:        np.ndarray,
    features: list[str],
    seed:     int = 42,
    df_raw:   pd.DataFrame | None = None,
) -> dict:
    """
    Entraîne Ridge, RandomForest et LMM sur les données groove.

    Args:
        X:        features normalisées (n_stimuli, n_features)  — agrégées par stim_id
        y:        groove_mean (n_stimuli,)                      — agrégé par stim_id
        features: noms des colonnes features
        seed:     graine pour RandomForest
        df_raw:   DataFrame brut avec colonnes [groove, participant_id, features...]
                  Si fourni, le LMM est estimé sur les réponses individuelles.
                  Si None, LMM ignoré silencieusement.

    Returns:
        dict { "Ridge": model, "RandomForest": model, "LMM": result_dict (optionnel) }
    """
    models = {}

    # ── Ridge ────────────────────────────────────────────
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)),
    ])
    ridge.fit(X, y)
    models["Ridge"] = ridge

    # ── RandomForest ─────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    models["RandomForest"] = rf

    # ── LMM ──────────────────────────────────────────────
    lmm = fit_lmm(df_raw, features) if df_raw is not None else None
    if lmm is not None:
        models["LMM"] = lmm

    return models


# =========================================================
# LMM
# =========================================================

def fit_lmm(
    df_raw:   pd.DataFrame,
    features: list[str],
) -> dict | None:
    """
    Estime le modèle linéaire mixte sur les réponses brutes.

    Modèle :
        G_ij = β·X_j + u_i + ε_ij
        u_i ~ N(0, σ_u²),  ε_ij ~ N(0, σ_ε²)

    Args:
        df_raw:   DataFrame brut avec colonnes : groove, participant_id, + features
        features: liste des predicteurs fixes

    Returns:
        dict avec résultats LMM ou None si estimation impossible
    """
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except ImportError:
        print("  [LMM] statsmodels absent — pip install statsmodels")
        return None

    # ── Validation ───────────────────────────────────────
    required = {"groove", "participant_id"} | set(features)
    missing  = required - set(df_raw.columns)
    if missing:
        print(f"  [LMM] colonnes manquantes dans df_raw : {missing} — LMM ignoré")
        return None

    df = df_raw[list(required)].dropna().copy()

    if len(df) < 20:
        print(f"  [LMM] trop peu de lignes ({len(df)}) — LMM ignoré")
        return None

    n_participants = df["participant_id"].nunique()
    if n_participants < 3:
        print(f"  [LMM] trop peu de participants ({n_participants}) — LMM ignoré")
        return None

    # ── Terme quadratique S_real (fidèle au mémoire) ─────
    if "S_real" in features:
        df["S_real_sq"] = df["S_real"] ** 2
        fixed_features = [f for f in features if f != "S_real"]
        fixed_features = ["S_real", "S_real_sq"] + fixed_features
    else:
        fixed_features = list(features)

    # ── Normalisation des predicteurs fixes ──────────────
    for col in fixed_features:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 1e-10:
                df[col] = (df[col] - mu) / sigma

    # ── Formule ──────────────────────────────────────────
    formula = "groove ~ " + " + ".join(fixed_features)

    # ── Estimation REML ──────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lmm_model = smf.mixedlm(
                formula,
                data=df,
                groups=df["participant_id"],
            )
            result = lmm_model.fit(reml=True, method="lbfgs")

        if not result.converged:
            print("  [LMM] ⚠️  convergence non atteinte — résultats indicatifs")

        # ── Extraction des résultats ──────────────────────
        coefs = {}
        for name, val in result.params.items():
            if name == "Intercept":
                continue
            coefs[name] = {
                "coef":    float(val),
                "se":      float(result.bse.get(name, np.nan)),
                "z":       float(result.tvalues.get(name, np.nan)),
                "p_value": float(result.pvalues.get(name, np.nan)),
                "ci_low":  float(result.conf_int().loc[name, 0])
                           if name in result.conf_int().index else np.nan,
                "ci_high": float(result.conf_int().loc[name, 1])
                           if name in result.conf_int().index else np.nan,
            }

        # ── Variance components ───────────────────────────
        sigma_u   = float(np.sqrt(result.cov_re.values[0, 0]))
        sigma_eps = float(np.sqrt(result.scale))
        icc_lmm   = float(sigma_u ** 2 / (sigma_u ** 2 + sigma_eps ** 2))

        # ── R² marginal et conditionnel (Nakagawa & Schielzeth 2013) ─
        r2_marginal, r2_conditional = _r2_lmm(result, df, fixed_features)

        lmm_out = {
            # Modèle sklearn-compatible pour predict (wrapper)
            "_result":          result,
            "_features":        fixed_features,
            "_df_stats":        {"mu": {}, "sigma": {}},  # déjà normalisé inline

            # Métriques rapportables
            "coefs":            coefs,
            "sigma_u":          sigma_u,
            "sigma_eps":        sigma_eps,
            "icc_participant":  icc_lmm,
            "r2_marginal":      r2_marginal,
            "r2_conditional":   r2_conditional,
            "n_obs":            int(len(df)),
            "n_participants":   int(n_participants),
            "converged":        bool(result.converged),
            "log_likelihood":   float(result.llf),
            "aic":              float(result.aic),
            "bic":              float(result.bic),

            # Pour l'évaluation cross-validée (approximée, voir note)
            "r2_cv_mean":       r2_marginal,   # proxy — LMM n'a pas de CV standard
            "r2_cv_std":        0.0,
            "mae_cv_mean":      _mae_lmm(result, df, fixed_features),
            "mae_cv_std":       0.0,
        }

        _print_lmm_summary(lmm_out)
        return lmm_out

    except Exception as e:
        print(f"  [LMM] Erreur d'estimation : {e}")
        import traceback
        traceback.print_exc()
        return None


# =========================================================
# R² MARGINAL & CONDITIONNEL (Nakagawa & Schielzeth 2013)
# =========================================================

def _r2_lmm(result, df: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    """
    R²_m (effets fixes seulement) et R²_c (effets fixes + aléatoires).

    Nakagawa & Schielzeth (2013) :
        R²_m = var(ŷ_fixed) / (var(ŷ_fixed) + σ_u² + σ_ε²)
        R²_c = (var(ŷ_fixed) + σ_u²) / (var(ŷ_fixed) + σ_u² + σ_ε²)
    """
    try:
        # Prédiction effets fixes seulement
        X_fixed = df[features].copy()
        X_fixed.insert(0, "Intercept", 1.0)

        fixed_params = result.params
        common_cols  = [c for c in X_fixed.columns if c in fixed_params.index]
        y_hat_fixed  = X_fixed[common_cols].values @ fixed_params[common_cols].values

        var_fixed = float(np.var(y_hat_fixed))
        sigma_u2  = float(result.cov_re.values[0, 0])
        sigma_e2  = float(result.scale)

        denom = var_fixed + sigma_u2 + sigma_e2
        if denom < 1e-10:
            return 0.0, 0.0

        r2_m = var_fixed / denom
        r2_c = (var_fixed + sigma_u2) / denom

        return float(np.clip(r2_m, 0, 1)), float(np.clip(r2_c, 0, 1))

    except Exception:
        return np.nan, np.nan


def _mae_lmm(result, df: pd.DataFrame, features: list[str]) -> float:
    """MAE sur les prédictions en sample (effets fixes seulement)."""
    try:
        X_fixed = df[features].copy()
        X_fixed.insert(0, "Intercept", 1.0)
        fixed_params = result.params
        common_cols  = [c for c in X_fixed.columns if c in fixed_params.index]
        y_hat        = X_fixed[common_cols].values @ fixed_params[common_cols].values
        return float(np.mean(np.abs(df["groove"].values - y_hat)))
    except Exception:
        return np.nan


# =========================================================
# RAPPORT CONSOLE
# =========================================================

def _print_lmm_summary(lmm: dict) -> None:
    w = 60
    print(f"\n{'─'*w}")
    print(f"  Modèle Linéaire Mixte (REML)")
    print(f"{'─'*w}")
    print(f"  Observations      : {lmm['n_obs']}")
    print(f"  Participants       : {lmm['n_participants']}")
    print(f"  Convergence        : {'✔' if lmm['converged'] else '⚠️  non atteinte'}")
    print()
    print(f"  σ_u  (inter-part.) : {lmm['sigma_u']:.4f}")
    print(f"  σ_ε  (résiduel)    : {lmm['sigma_eps']:.4f}")
    print(f"  ICC participant    : {lmm['icc_participant']:.3f}")
    print()
    print(f"  R²_marginal        : {lmm['r2_marginal']:.3f}  (effets fixes)")
    print(f"  R²_conditionnel    : {lmm['r2_conditional']:.3f}  (effets fixes + aléatoires)")
    print(f"  MAE in-sample      : {lmm['mae_cv_mean']:.3f}")
    print(f"  AIC / BIC          : {lmm['aic']:.1f} / {lmm['bic']:.1f}")
    print()
    print(f"  {'Predictor':<16} {'β':>8} {'SE':>8} {'z':>8} {'p':>8}")
    print(f"  {'─'*52}")
    for name, v in lmm["coefs"].items():
        sig = "★" if v["p_value"] < 0.05 else ""
        print(
            f"  {name:<16} {v['coef']:>8.3f} {v['se']:>8.3f} "
            f"{v['z']:>8.2f} {v['p_value']:>8.3f} {sig}"
        )
    print(f"{'─'*w}\n")