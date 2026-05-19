"""
regression/model.py
===================
Définition et entraînement des modèles de régression groove.

Modèles :
    Ridge         — linéaire régularisé, interprétable via coefficients
    RandomForest  — non-linéaire, interprétable via feature importances
    LMM           — modèle linéaire mixte avec effet aléatoire participant

                    Modèle formel du mémoire :
                    G_ij = β₀ + β₁·S_z + β₂·S_z² + β₃·D_z + β₄·I_z
                         + β₅·E_z + β₆·V_z + β₇·P_z + β₈·BPM_z
                         + β₉·musical_background   (si disponible)
                         + u_i + ε_ij

                    Où _z désigne la variable centrée-réduite.
                    S_z² est calculé APRÈS normalisation de S, ce qui
                    garantit que β₁ et β₂ sont interprétables conjointement
                    (S_z=0 correspond à la moyenne de S dans l'échantillon).

                    musical_background est dummy-encodé avec "non_musician"
                    comme référence.

v3 :
    - S_sq est désormais calculé sur S normalisé (S_z), pas sur S brut.
      Correction critique pour l'interprétabilité des coefficients LMM.
    - _compat_rename retiré de ce module (délégué à data_loader.py).

Notation :
    S, E, P = descripteurs émergents (sans indice)
    S_mv, E_mv, P_mv, D_mv = paramètres génératifs
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Référence catégorielle pour musical_background
BACKGROUND_REF    = "non_musician"
BACKGROUND_LEVELS = ["amateur", "semi_pro", "pro"]
BACKGROUND_LABELS = {
    "amateur":  "Musicien·ne amateur·rice",
    "semi_pro": "Musicien·ne semi-pro",
    "pro":      "Musicien·ne professionnel·le",
}


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
    models = {}

    # ── Ridge
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)),
    ])
    ridge.fit(X, y)
    models["Ridge"] = ridge

    # ── RandomForest
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    models["RandomForest"] = rf

    # ── LMM
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
    Modèle linéaire mixte sur les réponses brutes.

    G_ij = β·X_z_j + β_bg·musical_background_i + u_i + ε_ij

    ORDRE DES OPÉRATIONS (important pour S_sq) :
        1. Sélection et nettoyage des colonnes
        2. Normalisation des prédicteurs continus  ← S devient S_z ici
        3. Création de S_sq = S_z²                 ← APRÈS normalisation
        4. Ajout des dummies musical_background
        5. Estimation REML

    Cet ordre garantit que β_S et β_S_sq sont interprétables conjointement :
    S_z=0 correspond à la moyenne de S dans l'échantillon, et S_sq mesure
    la courbure autour de cette moyenne — ce qui correspond au modèle
    décrit dans le mémoire (§ régression groove).
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  [LMM] statsmodels absent — pip install statsmodels")
        return None

    required = {"groove", "participant_id"} | set(features)
    missing  = required - set(df_raw.columns)
    if missing:
        print(f"  [LMM] colonnes manquantes dans df_raw : {missing} — LMM ignoré")
        return None

    df = df_raw.copy()

    # ── Dummy encoding musical_background ────────────────
    bg_dummies_used: list[str] = []
    has_bg = (
        "musical_background" in df.columns
        and df["musical_background"].notna().sum() >= 10
    )

    if has_bg:
        df["musical_background"] = df["musical_background"].astype(str)
        df.loc[~df["musical_background"].isin(
            [BACKGROUND_REF] + BACKGROUND_LEVELS
        ), "musical_background"] = None

        for lvl in BACKGROUND_LEVELS:
            col = f"bg_{lvl}"
            df[col] = (df["musical_background"] == lvl).astype(float)
            if df[col].sum() >= 3:
                bg_dummies_used.append(col)

        if bg_dummies_used:
            print(
                f"  [LMM] musical_background → dummies : {bg_dummies_used} "
                f"(réf. : {BACKGROUND_REF})"
            )
        else:
            print("  [LMM] musical_background : niveaux insuffisants — ignoré")
            has_bg = False

    # ── Sélection des colonnes ────────────────────────────
    all_fixed = list(features) + bg_dummies_used
    keep      = ["groove", "participant_id", "stim_id"] + all_fixed
    df        = df[[c for c in keep if c in df.columns]].dropna(subset=["groove"])

    if len(df) < 20:
        print(f"  [LMM] trop peu de lignes ({len(df)}) — LMM ignoré")
        return None

    n_participants = df["participant_id"].nunique()
    if n_participants < 3:
        print(f"  [LMM] trop peu de participants ({n_participants}) — LMM ignoré")
        return None

    # ── ÉTAPE 1 : Normalisation des prédicteurs continus ──
    # Les dummies bg_* ne sont PAS normalisés (déjà 0/1).
    # IMPORTANT : S est normalisé ICI, avant la création de S_sq.
    for col in features:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 1e-10:
                df[col] = (df[col] - mu) / sigma

    # ── ÉTAPE 2 : Terme quadratique sur S normalisé ───────
    # S_sq = S_z² où S_z = (S - mean(S)) / std(S)
    # β_S  → effet linéaire de S autour de sa moyenne
    # β_S_sq → courbure (concavité/convexité) de la relation groove ~ S
    # Les deux coefficients sont maintenant interprétables conjointement.
    has_S_quadratic = "S" in features
    if has_S_quadratic:
        df["S_sq"] = df["S"] ** 2   # S est déjà normalisé à ce stade
        fixed_features = ["S", "S_sq"] + [f for f in all_fixed if f != "S"]
    else:
        fixed_features = list(all_fixed)

    formula = "groove ~ " + " + ".join(fixed_features)

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

        coefs = {}
        for name, val in result.params.items():
            if name == "Intercept":
                continue
            ci = result.conf_int()
            coefs[name] = {
                "coef":          float(val),
                "se":            float(result.bse.get(name, np.nan)),
                "z":             float(result.tvalues.get(name, np.nan)),
                "p_value":       float(result.pvalues.get(name, np.nan)),
                "ci_low":        float(ci.loc[name, 0]) if name in ci.index else np.nan,
                "ci_high":       float(ci.loc[name, 1]) if name in ci.index else np.nan,
                "is_background": name in bg_dummies_used,
            }

        sigma_u   = float(np.sqrt(result.cov_re.values[0, 0]))
        sigma_eps = float(np.sqrt(result.scale))
        icc_lmm   = float(sigma_u ** 2 / (sigma_u ** 2 + sigma_eps ** 2))

        r2_marginal, r2_conditional = _r2_lmm(result, df, fixed_features)

        lmm_out = {
            "_result":           result,
            "_features":         fixed_features,
            "_bg_dummies":       bg_dummies_used,
            "_bg_ref":           BACKGROUND_REF if has_bg else None,
            "coefs":             coefs,
            "sigma_u":           sigma_u,
            "sigma_eps":         sigma_eps,
            "icc_participant":   icc_lmm,
            "r2_marginal":       r2_marginal,
            "r2_conditional":    r2_conditional,
            "n_obs":             int(len(df)),
            "n_participants":    int(n_participants),
            "converged":         bool(result.converged),
            "log_likelihood":    float(result.llf),
            "aic":               float(result.aic),
            "bic":               float(result.bic),
            "has_background":    has_bg,
            "s_sq_on_normalized": True,   # flag de traçabilité
            # Alias pour evaluate_models
            "r2_cv_mean":        r2_marginal,
            "r2_cv_std":         0.0,
            "mae_cv_mean":       _mae_lmm(result, df, fixed_features),
            "mae_cv_std":        0.0,
        }

        _print_lmm_summary(lmm_out)
        return lmm_out

    except Exception as e:
        print(f"  [LMM] Erreur d'estimation : {e}")
        import traceback
        traceback.print_exc()
        return None


# =========================================================
# HELPERS LMM
# =========================================================

def _r2_lmm(result, df, features):
    try:
        X_fixed = df[features].copy()
        X_fixed.insert(0, "Intercept", 1.0)
        fixed_params = result.params
        common_cols  = [c for c in X_fixed.columns if c in fixed_params.index]
        y_hat_fixed  = X_fixed[common_cols].values @ fixed_params[common_cols].values
        var_fixed    = float(np.var(y_hat_fixed))
        sigma_u2     = float(result.cov_re.values[0, 0])
        sigma_e2     = float(result.scale)
        denom        = var_fixed + sigma_u2 + sigma_e2
        if denom < 1e-10:
            return 0.0, 0.0
        return (
            float(np.clip(var_fixed / denom, 0, 1)),
            float(np.clip((var_fixed + sigma_u2) / denom, 0, 1)),
        )
    except Exception:
        return np.nan, np.nan


def _mae_lmm(result, df, features):
    try:
        X_fixed      = df[features].copy()
        X_fixed.insert(0, "Intercept", 1.0)
        fixed_params = result.params
        common_cols  = [c for c in X_fixed.columns if c in fixed_params.index]
        y_hat        = X_fixed[common_cols].values @ fixed_params[common_cols].values
        return float(np.mean(np.abs(df["groove"].values - y_hat)))
    except Exception:
        return np.nan


def _print_lmm_summary(lmm: dict) -> None:
    w = 64
    print(f"\n{'─'*w}")
    print(f"  Modèle Linéaire Mixte (REML)")
    print(f"{'─'*w}")
    print(f"  Observations         : {lmm['n_obs']}")
    print(f"  Participants         : {lmm['n_participants']}")
    print(f"  Convergence          : {'✔' if lmm['converged'] else '⚠️  non atteinte'}")
    print(f"  S² calculé sur S_z   : ✔  (après normalisation — interprétable)")
    if lmm.get("has_background"):
        print(f"  Musical background   : ✔  (réf. = {lmm['_bg_ref']})")
    else:
        print(f"  Musical background   : —  (absent ou insuffisant)")
    print()
    print(f"  σ_u  (inter-part.)   : {lmm['sigma_u']:.4f}")
    print(f"  σ_ε  (résiduel)      : {lmm['sigma_eps']:.4f}")
    print(f"  ICC participant      : {lmm['icc_participant']:.3f}")
    print()
    print(f"  R²_marginal          : {lmm['r2_marginal']:.3f}  (effets fixes)")
    print(f"  R²_conditionnel      : {lmm['r2_conditional']:.3f}  (fixes + aléatoires)")
    print(f"  MAE in-sample        : {lmm['mae_cv_mean']:.3f}")
    print(f"  AIC / BIC            : {lmm['aic']:.1f} / {lmm['bic']:.1f}")
    print()

    acoustic_coefs = {k: v for k, v in lmm["coefs"].items() if not v.get("is_background")}
    bg_coefs       = {k: v for k, v in lmm["coefs"].items() if v.get("is_background")}

    print(f"  {'Predictor':<18} {'β':>8} {'SE':>8} {'z':>8} {'p':>8}")
    print(f"  {'─'*56}")
    for name, v in acoustic_coefs.items():
        sig = "★" if v["p_value"] < 0.05 else ""
        print(
            f"  {name:<18} {v['coef']:>8.3f} {v['se']:>8.3f} "
            f"{v['z']:>8.2f} {v['p_value']:>8.3f} {sig}"
        )

    if bg_coefs:
        print(f"\n  Bagage musical  (réf. = {lmm['_bg_ref']})")
        print(f"  {'─'*56}")
        for name, v in bg_coefs.items():
            level     = name.replace("bg_", "")
            label     = BACKGROUND_LABELS.get(level, name)
            sig       = "★" if v["p_value"] < 0.05 else ""
            direction = "↑" if v["coef"] > 0 else "↓"
            print(
                f"  {label:<28} β={v['coef']:>+7.3f}  "
                f"p={v['p_value']:.3f} {sig}  {direction}"
            )
        print(
            f"\n  Interprétation : β > 0 → rating groove plus élevé "
            f"que les non-musiciens,\n"
            f"  toutes choses égales par ailleurs."
        )

    print(f"{'─'*w}\n")