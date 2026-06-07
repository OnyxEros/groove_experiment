"""
regression/models/lmm.py
=========================
Modèle Linéaire Mixte (LMM) pour l'inférence sur le groove perçu.

Modèle formel (mémoire) :
    G_ij = β₀ + β_D·D_j + β_S·S_j + β_E·E_j + β_P·P_j
           + β_bg·musical_background_i + u_i + ε_ij

    où  _j  = stimulus (effets fixes),
        _i  = participant (effet aléatoire u_i ~ N(0, σ_u²)),
        ε_ij ~ N(0, σ_ε²).

Estimation : REML (estimateur non-biaisé des composantes de variance).
AIC/BIC    : re-fit ML pour comparaison (non rapportés comme résultats principaux).

Justification du LMM vs régression simple :
    - Chaque participant a noté ~30 stimuli → observations non indépendantes.
    - L'ICC_participant quantifie la variance due aux différences de sensibilité
      individuelles au groove, indépendamment des stimuli.
    - Sans effet aléatoire, les SE seraient sous-estimés (pseudo-réplication).

Référence R² : Nakagawa & Schielzeth (2013), Methods Ecol. Evol. 4(2):133–142.
"""

from __future__ import annotations

import warnings
import math
import numpy as np
import pandas as pd

from regression.models.base import GrooveModel

BACKGROUND_REF    = "non_musician"
BACKGROUND_LEVELS = ["amateur", "semi_pro", "pro"]
BACKGROUND_LABELS = {
    "amateur":  "Amateur·rice",
    "semi_pro": "Semi-professionnel·le",
    "pro":      "Professionnel·le",
}


class LMMModel(GrooveModel):

    name = "LMM"
    supports_raw_data = True

    def __init__(self, seed: int = 42):
        self.seed    = seed
        self._result = None
        self._fitted = False
        self._summary: dict = {}

    def fit(self, X, y, features, df_raw=None) -> "LMMModel":
        if df_raw is None:
            print("  [LMM] df_raw=None — LMM ignoré")
            return self

        summary = _fit_lmm(df_raw, features)
        if summary is None:
            return self

        self._summary       = summary
        self._result        = summary.get("_result")
        self._features_used = summary.get("_features", features)
        self._fitted        = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LMMModel.predict() non supporté. "
            "Évaluation via R²_marginal in-sample."
        )

    def get_results(self) -> dict:
        if not self._fitted:
            return {"name": self.name, "_not_fitted": True}

        return {
            "name":             self.name,
            "coefs":            self._summary.get("coefs", {}),
            "r2_marginal":      self._summary.get("r2_marginal",    float("nan")),
            "r2_conditional":   self._summary.get("r2_conditional", float("nan")),
            "mae_in_sample":    self._summary.get("mae_in_sample",  float("nan")),
            "icc_participant":  self._summary.get("icc_participant", float("nan")),
            "sigma_u":          self._summary.get("sigma_u",  float("nan")),
            "sigma_eps":        self._summary.get("sigma_eps", float("nan")),
            "n_obs":            self._summary.get("n_obs", 0),
            "n_participants":   self._summary.get("n_participants", 0),
            "converged":        self._summary.get("converged", False),
            "aic_ml":           self._summary.get("aic_ml", float("nan")),
            "bic_ml":           self._summary.get("bic_ml", float("nan")),
            "has_background":   self._summary.get("has_background", False),
            "_is_lmm":          True,
            # Accès interne pour fig13 (résidus)
            "_result":          self._summary.get("_result"),
        }


# ============================================================
# ESTIMATION
# ============================================================

def _fit_lmm(df_raw: pd.DataFrame, features: list[str]) -> dict | None:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  [LMM] statsmodels absent — pip install statsmodels")
        return None

    # Colonnes requises
    required = {"groove", "participant_id"} | set(features)
    missing  = required - set(df_raw.columns)
    if missing:
        print(f"  [LMM] colonnes manquantes : {missing}")
        return None

    df = df_raw.copy()

    # ── Musical background ────────────────────────────────────────────────────
    bg_dummies, has_bg = _encode_background(df)

    # ── Normalisation des features ────────────────────────────────────────────
    for col in features:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 1e-10:
            df[col] = (df[col] - mu) / sigma

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    keep = ["groove", "participant_id"] + features + bg_dummies
    df   = df[[c for c in keep if c in df.columns]].dropna(subset=["groove"])

    if len(df) < 20:
        print(f"  [LMM] trop peu de lignes ({len(df)}) — ignoré")
        return None
    if df["participant_id"].nunique() < 3:
        print(f"  [LMM] trop peu de participants ({df['participant_id'].nunique()}) — ignoré")
        return None

    # ── Formule ───────────────────────────────────────────────────────────────
    fixed   = list(features) + bg_dummies
    formula = "groove ~ " + " + ".join(fixed)
    print(f"  [LMM] formule : {formula}")
    print(f"  [LMM] n={len(df)} obs, {df['participant_id'].nunique()} participants")

    # ── Estimation REML ───────────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lmm = smf.mixedlm(formula, data=df, groups=df["participant_id"])
            try:
                result = lmm.fit(reml=True, method="lbfgs")
            except np.linalg.LinAlgError:
                print("  [LMM] lbfgs → singulière, fallback powell")
                result = lmm.fit(reml=True, method="powell")

        if not result.converged:
            print("  [LMM] ⚠  convergence non atteinte — résultats indicatifs")

    except Exception as exc:
        import traceback
        print(f"  [LMM] erreur : {exc}")
        traceback.print_exc()
        return None

    # ── Extraction ────────────────────────────────────────────────────────────
    coefs     = _extract_coefs(result, bg_dummies)
    sigma_u   = float(np.sqrt(max(result.cov_re.values[0, 0], 0.0)))
    sigma_eps = float(np.sqrt(result.scale))
    icc       = float(sigma_u**2 / (sigma_u**2 + sigma_eps**2 + 1e-10))

    r2_marg, r2_cond = _r2_nakagawa(result, df, fixed)
    mae              = _mae_insample(result, df, fixed)
    aic_ml, bic_ml   = _aic_bic_ml(formula, df)

    summary = {
        "_result":         result,
        "_features":       fixed,
        "coefs":           coefs,
        "sigma_u":         sigma_u,
        "sigma_eps":       sigma_eps,
        "icc_participant": icc,
        "r2_marginal":     r2_marg,
        "r2_conditional":  r2_cond,
        "mae_in_sample":   mae,
        "n_obs":           int(len(df)),
        "n_participants":  int(df["participant_id"].nunique()),
        "converged":       bool(result.converged),
        "aic_ml":          aic_ml,
        "bic_ml":          bic_ml,
        "has_background":  has_bg,
    }

    _print_lmm_summary(summary)
    return summary


# ============================================================
# HELPERS ESTIMATION
# ============================================================

def _encode_background(df: pd.DataFrame) -> tuple[list[str], bool]:
    if "musical_background" not in df.columns:
        return [], False
    if df["musical_background"].notna().sum() < 10:
        return [], False

    df["musical_background"] = df["musical_background"].astype(str)
    valid = [BACKGROUND_REF, *BACKGROUND_LEVELS]
    df.loc[~df["musical_background"].isin(valid), "musical_background"] = None

    dummies = []
    for lvl in BACKGROUND_LEVELS:
        col = f"bg_{lvl}"
        df[col] = (df["musical_background"] == lvl).astype(float)
        if df[col].sum() >= 3:
            dummies.append(col)

    if not dummies:
        print("  [LMM] musical_background : niveaux insuffisants — ignoré")
        return [], False

    print(f"  [LMM] background dummies : {dummies}  (réf. : {BACKGROUND_REF})")
    return dummies, True


def _extract_coefs(result, bg_dummies: list[str]) -> dict:
    coefs = {}
    ci    = result.conf_int()
    for name, val in result.params.items():
        if name == "Intercept":
            continue
        coefs[name] = {
            "coef":          float(val),
            "se":            float(result.bse.get(name, float("nan"))),
            "z":             float(result.tvalues.get(name, float("nan"))),
            "p_value":       float(result.pvalues.get(name, float("nan"))),
            "ci_low":        float(ci.loc[name, 0]) if name in ci.index else float("nan"),
            "ci_high":       float(ci.loc[name, 1]) if name in ci.index else float("nan"),
            "is_background": name in bg_dummies,
        }
    return coefs


def _r2_nakagawa(result, df: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    """
    R²_marginal  = var_fixed / (var_fixed + σ²_u + σ²_ε)
    R²_conditionnel = (var_fixed + σ²_u) / (var_fixed + σ²_u + σ²_ε)

    Nakagawa & Schielzeth (2013).
    """
    try:
        sigma_u2  = float(result.cov_re.values[0, 0])
        sigma_e2  = float(result.scale)

        X_fixed = df[features].copy()
        X_fixed.insert(0, "Intercept", 1.0)
        params  = result.params
        common  = [c for c in X_fixed.columns if c in params.index]
        y_fixed = X_fixed[common].values @ params.loc[common].values
        var_fixed = float(np.var(y_fixed, ddof=0))

        denom = var_fixed + sigma_u2 + sigma_e2
        if denom < 1e-10:
            return 0.0, 0.0

        r2_marg = float(np.clip(var_fixed / denom, 0.0, 1.0))
        r2_cond = float(np.clip((var_fixed + sigma_u2) / denom, 0.0, 1.0))
        return r2_marg, r2_cond

    except Exception as e:
        print(f"  [LMM] _r2_nakagawa erreur : {e}")
        return float("nan"), float("nan")


def _mae_insample(result, df: pd.DataFrame, features: list[str]) -> float:
    try:
        X_fixed = df[features].copy()
        X_fixed.insert(0, "Intercept", 1.0)
        params = result.params
        common = [c for c in X_fixed.columns if c in params.index]
        y_hat  = X_fixed[common].values @ params.loc[common].values
        return float(np.mean(np.abs(df["groove"].values - y_hat)))
    except Exception as e:
        print(f"  [LMM] _mae_insample erreur : {e}")
        return float("nan")


def _aic_bic_ml(formula: str, df: pd.DataFrame) -> tuple[float, float]:
    """Re-fit ML uniquement pour AIC/BIC — coefficients REML restent la référence."""
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(formula, data=df, groups=df["participant_id"])
            try:
                r = m.fit(reml=False, method="lbfgs")
            except Exception:
                r = m.fit(reml=False, method="powell")
        aic = float(r.aic) if np.isfinite(r.aic) else float("nan")
        bic = float(r.bic) if np.isfinite(r.bic) else float("nan")
        return aic, bic
    except Exception as e:
        print(f"  [LMM] _aic_bic_ml erreur : {e}")
        return float("nan"), float("nan")


# ============================================================
# RAPPORT CONSOLE
# ============================================================

def _print_lmm_summary(s: dict) -> None:
    w = 64
    conv = "✔" if s["converged"] else "⚠  non atteinte (résultats indicatifs)"
    print(f"\n{'─'*w}")
    print(f"  LMM — Modèle Linéaire Mixte (REML)")
    print(f"{'─'*w}")
    print(f"  n observations       : {s['n_obs']}")
    print(f"  n participants       : {s['n_participants']}")
    print(f"  Convergence          : {conv}")
    print()
    print(f"  σ_u  (inter-part.)   : {s['sigma_u']:.4f}")
    print(f"  σ_ε  (résiduel)      : {s['sigma_eps']:.4f}")
    print(f"  ICC participant      : {s['icc_participant']:.3f}")
    print()
    print(f"  R²_marginal          : {s['r2_marginal']:.3f}  (effets fixes seuls)")
    print(f"  R²_conditionnel      : {s['r2_conditional']:.3f}  (fixes + aléatoires)")
    print(f"  MAE in-sample        : {s['mae_in_sample']:.3f}")
    aic_s = f"{s['aic_ml']:.1f}" if math.isfinite(s['aic_ml']) else "nan"
    bic_s = f"{s['bic_ml']:.1f}" if math.isfinite(s['bic_ml']) else "nan"
    print(f"  AIC_ML / BIC_ML      : {aic_s} / {bic_s}  (ML, pour info uniquement)")
    print()

    # Effets principaux
    main_c = {k: v for k, v in s["coefs"].items() if not v.get("is_background")}
    bg_c   = {k: v for k, v in s["coefs"].items() if v.get("is_background")}

    header = f"  {'Prédicteur':<18} {'β':>8} {'SE':>7} {'z':>7} {'p':>8}  IC 95%"
    sep    = f"  {'─'*62}"
    print(header)
    print(sep)
    for name, v in main_c.items():
        sig = "★" if v["p_value"] < 0.05 else " "
        ci  = f"[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"
        print(
            f"  {name:<18} {v['coef']:>+8.3f} {v['se']:>7.3f} "
            f"{v['z']:>7.2f} {v['p_value']:>8.3f} {sig}  {ci}"
        )

    if bg_c:
        print(f"\n  Bagage musical (réf. = {BACKGROUND_REF})")
        print(sep)
        for name, v in bg_c.items():
            lvl   = name.replace("bg_", "")
            label = BACKGROUND_LABELS.get(lvl, name)
            sig   = "★" if v["p_value"] < 0.05 else " "
            print(
                f"  {label:<26} β={v['coef']:>+7.3f}  "
                f"SE={v['se']:.3f}  p={v['p_value']:.3f} {sig}"
            )

    print(f"{'─'*w}\n")