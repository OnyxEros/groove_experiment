"""
regression/models/lmm.py
=========================
Modèle Linéaire Mixte (LMM) pour la régression groove.

Modèle formel (mémoire) :
    G_ij = β₀ + β·X_z_j + β_bg·musical_background_i + u_i + ε_ij

    Où _z = centré-réduit, u_i ~ N(0, σ_u²), i = participant, j = stimulus.

Changements v6 :
    - Support des termes d'interaction et quadratiques dans la formule LMM.
      Quand feature_set='interactions', les colonnes D_sq, DxP, SxE, DxS
      sont présentes dans df_raw et incluses dans la formule automatiquement.
    - _build_formula() : construction propre de la formule statsmodels,
      avec gestion du terme quadratique S_sq (existant) et des nouveaux termes.
    - Rapport console étendu : section dédiée aux termes d'interaction.

ORDRE DES OPÉRATIONS (important) :
    1. Normalisation de D, S, E, P → D_z, S_z, E_z, P_z
    2. D_sq = D_z²  (après normalisation)
    3. S_sq = S_z²  (après normalisation — existant)
    4. DxP  = D_z × P_z  etc.
    5. Dummies musical_background
    6. Estimation REML
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from regression.models.base import GrooveModel

BACKGROUND_REF    = "non_musician"
BACKGROUND_LEVELS = ["amateur", "semi_pro", "pro"]
BACKGROUND_LABELS = {
    "amateur":  "Musicien·ne amateur·rice",
    "semi_pro": "Musicien·ne semi-pro",
    "pro":      "Musicien·ne professionnel·le",
}

# Termes d'interaction reconnus — même mapping que features.py
INTERACTION_TERMS = {
    "D_sq": ("D", "D"),
    "DxP":  ("D", "P"),
    "SxE":  ("S", "E"),
    "DxS":  ("D", "S"),
}


class LMMModel(GrooveModel):

    name = "LMM"
    supports_raw_data = True

    def __init__(self, seed: int = 42):
        self.seed     = seed
        self._result  = None
        self._fitted  = False
        self._summary: dict = {}

    def fit(self, X, y, features, df_raw=None) -> "LMMModel":
        if df_raw is None:
            print("  [LMM] df_raw=None — LMM non entraîné pour ce run")
            return self

        summary = _fit_lmm(df_raw, features)
        if summary is None:
            return self

        self._summary        = summary
        self._result         = summary.get("_result")
        self._features_used  = summary.get("_features", features)
        self._fitted         = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LMMModel.predict() n'est pas supporté. "
            "Le LMM est évalué in-sample via R²_marginal."
        )

    def get_results(self) -> dict:
        if not self._fitted:
            return {"name": self.name, "coefs": None, "importances": None, "_not_fitted": True}

        return {
            "name":             self.name,
            "coefs":            self._summary.get("coefs", {}),
            "importances":      None,
            "r2_cv_mean":       self._summary.get("r2_marginal", float("nan")),
            "r2_marginal":      self._summary.get("r2_marginal", float("nan")),
            "r2_conditional":   self._summary.get("r2_conditional", float("nan")),
            "mae_cv_mean":      self._summary.get("mae_cv_mean", float("nan")),
            "icc_participant":  self._summary.get("icc_participant", float("nan")),
            "sigma_u":          self._summary.get("sigma_u", float("nan")),
            "sigma_eps":        self._summary.get("sigma_eps", float("nan")),
            "n_obs":            self._summary.get("n_obs", 0),
            "n_participants":   self._summary.get("n_participants", 0),
            "converged":        self._summary.get("converged", False),
            "aic":              self._summary.get("aic", float("nan")),
            "bic":              self._summary.get("bic", float("nan")),
            "has_background":   self._summary.get("has_background", False),
            "interaction_terms": self._summary.get("interaction_terms", []),
            "_is_lmm":          True,
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

    base_features_check = [f for f in features if f not in INTERACTION_TERMS]
    required = {"groove", "participant_id"} | set(base_features_check)
    missing  = required - set(df_raw.columns)
    if missing:
        print(f"  [LMM] colonnes manquantes : {missing} — ignoré")
        return None

    df = df_raw.copy()

    # ── Musical background ────────────────────────────────────────────────────
    bg_dummies_used, has_bg = _encode_background(df)

    # ── Colonnes à garder ─────────────────────────────────────────────────────
    base_features    = [f for f in features if f not in INTERACTION_TERMS]
    interaction_cols = [f for f in features if f in INTERACTION_TERMS]
    all_fixed        = base_features + bg_dummies_used

    keep = ["groove", "participant_id", "stim_id"] + all_fixed + interaction_cols
    df   = df[[c for c in keep if c in df.columns]].dropna(subset=["groove"])

    if len(df) < 20:
        print(f"  [LMM] trop peu de lignes ({len(df)}) — ignoré")
        return None
    if df["participant_id"].nunique() < 3:
        print("  [LMM] trop peu de participants — ignoré")
        return None

    # ── Normalisation des features de base ───────────────────────────────────
    for col in base_features:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 1e-10:
            df[col] = (df[col] - mu) / sigma

    # ── Termes quadratiques et d'interaction (après normalisation) ───────────
    computed_interactions = []

    # S_sq : terme historique (syncopation quadratique)
    has_S_quad = "S" in base_features and "S_mv" not in base_features
    if has_S_quad and "S_sq" not in df.columns:
        df["S_sq"] = df["S"] ** 2
        computed_interactions.append("S_sq")

    # Nouveaux termes d'interaction
    for term, (f1, f2) in INTERACTION_TERMS.items():
        if term not in interaction_cols:
            continue
        if f1 in df.columns and f2 in df.columns:
            df[term] = df[f1].values * df[f2].values
            computed_interactions.append(term)
        else:
            print(f"  [LMM] interaction {term} ignorée : {f1} ou {f2} absent")

    if computed_interactions:
        print(f"  [LMM] termes calculés : {computed_interactions}")

    # ── Construction de la formule ────────────────────────────────────────────
    formula, fixed_features = _build_formula(
        base_features   = base_features,
        bg_dummies      = bg_dummies_used,
        has_S_quad      = has_S_quad,
        interaction_cols= [t for t in computed_interactions if t != "S_sq"],
        df              = df,
    )

    print(f"  [LMM] formule : {formula}")

    # ── Estimation REML ───────────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lmm_model = smf.mixedlm(formula, data=df, groups=df["participant_id"])
            try:
                result = lmm_model.fit(reml=True, method="lbfgs")
            except np.linalg.LinAlgError:
                print("  [LMM] lbfgs → matrice singulière, fallback powell")
                result = lmm_model.fit(reml=True, method="powell")

        if not result.converged:
            print("  [LMM] ⚠️ convergence non atteinte — résultats indicatifs")

    except Exception as exc:
        import traceback
        print(f"  [LMM] erreur d'estimation : {exc}")
        traceback.print_exc()
        return None

    # ── Extraction ───────────────────────────────────────────────────────────
    coefs     = _extract_coefs(result, bg_dummies_used)
    sigma_u   = float(np.sqrt(result.cov_re.values[0, 0]))
    sigma_eps = float(np.sqrt(result.scale))
    icc       = float(sigma_u**2 / (sigma_u**2 + sigma_eps**2))

    r2_marg, r2_cond = _r2_lmm(result, df, fixed_features)
    mae              = _mae_lmm(result, df, fixed_features)

    summary = {
        "_result":          result,
        "_features":        fixed_features,
        "_bg_dummies":      bg_dummies_used,
        "_bg_ref":          BACKGROUND_REF if has_bg else None,
        "coefs":            coefs,
        "sigma_u":          sigma_u,
        "sigma_eps":        sigma_eps,
        "icc_participant":  icc,
        "r2_marginal":      r2_marg,
        "r2_conditional":   r2_cond,
        "mae_cv_mean":      mae,
        "n_obs":            int(len(df)),
        "n_participants":   int(df["participant_id"].nunique()),
        "converged":        bool(result.converged),
        "aic":              float(result.aic),
        "bic":              float(result.bic),
        "has_background":   has_bg,
        "interaction_terms": computed_interactions,
    }

    _print_lmm_summary(summary)
    return summary


def _build_formula(
    base_features:    list[str],
    bg_dummies:       list[str],
    has_S_quad:       bool,
    interaction_cols: list[str],
    df:               pd.DataFrame,
) -> tuple[str, list[str]]:
    """
    Construit la formule statsmodels et la liste ordonnée des fixed features.
    """
    # Features de base (sans S si S_sq présent — on le remet avec S_sq)
    if has_S_quad:
        base_no_S = [f for f in base_features if f != "S"]
        fixed = ["S", "S_sq"] + base_no_S
    else:
        fixed = list(base_features)

    # Termes d'interaction
    for term in interaction_cols:
        if term in df.columns and term not in fixed:
            fixed.append(term)

    # Background dummies
    fixed += [b for b in bg_dummies if b not in fixed]

    formula = "groove ~ " + " + ".join(fixed)
    return formula, fixed


# ============================================================
# HELPERS
# ============================================================

def _encode_background(df: pd.DataFrame) -> tuple[list[str], bool]:
    if "musical_background" not in df.columns:
        return [], False
    if df["musical_background"].notna().sum() < 10:
        return [], False

    df["musical_background"] = df["musical_background"].astype(str)
    valid = [BACKGROUND_REF, *BACKGROUND_LEVELS]
    df.loc[~df["musical_background"].isin(valid), "musical_background"] = None

    dummies_used = []
    for lvl in BACKGROUND_LEVELS:
        col = f"bg_{lvl}"
        df[col] = (df["musical_background"] == lvl).astype(float)
        if df[col].sum() >= 3:
            dummies_used.append(col)

    if not dummies_used:
        print("  [LMM] musical_background : niveaux insuffisants — ignoré")
        return [], False

    print(f"  [LMM] musical_background → dummies : {dummies_used} (réf. : {BACKGROUND_REF})")
    return dummies_used, True


def _extract_coefs(result, bg_dummies: list[str]) -> dict:
    coefs = {}
    ci    = result.conf_int()
    for name, val in result.params.items():
        if name == "Intercept":
            continue
        is_interaction = any(
            name == term for term in list(INTERACTION_TERMS.keys()) + ["S_sq"]
        )
        coefs[name] = {
            "coef":           float(val),
            "se":             float(result.bse.get(name, np.nan)),
            "z":              float(result.tvalues.get(name, np.nan)),
            "p_value":        float(result.pvalues.get(name, np.nan)),
            "ci_low":         float(ci.loc[name, 0]) if name in ci.index else float("nan"),
            "ci_high":        float(ci.loc[name, 1]) if name in ci.index else float("nan"),
            "is_background":  name in bg_dummies,
            "is_interaction": is_interaction,
        }
    return coefs


def _r2_lmm(result, df: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    try:
        X_fixed = df[features].copy()
        X_fixed.insert(0, "Intercept", 1.0)
        fixed_params  = result.params
        common_cols   = [c for c in X_fixed.columns if c in fixed_params.index]
        coef_ordered  = fixed_params.loc[common_cols].values
        y_hat_fixed   = X_fixed[common_cols].values @ coef_ordered
        var_fixed     = float(np.var(y_hat_fixed))
        sigma_u2      = float(result.cov_re.values[0, 0])
        sigma_e2      = float(result.scale)
        denom         = var_fixed + sigma_u2 + sigma_e2
        if denom < 1e-10:
            return 0.0, 0.0
        return (
            float(np.clip(var_fixed / denom, 0, 1)),
            float(np.clip((var_fixed + sigma_u2) / denom, 0, 1)),
        )
    except Exception as exc:
        print(f"  [LMM] _r2_lmm erreur : {exc}")
        return float("nan"), float("nan")


def _mae_lmm(result, df: pd.DataFrame, features: list[str]) -> float:
    try:
        X_fixed = df[features].copy()
        X_fixed.insert(0, "Intercept", 1.0)
        fixed_params = result.params
        common_cols  = [c for c in X_fixed.columns if c in fixed_params.index]
        coef_ordered = fixed_params.loc[common_cols].values
        y_hat        = X_fixed[common_cols].values @ coef_ordered
        return float(np.mean(np.abs(df["groove"].values - y_hat)))
    except Exception as exc:
        print(f"  [LMM] _mae_lmm erreur : {exc}")
        return float("nan")


def _print_lmm_summary(s: dict) -> None:
    w = 64
    print(f"\n{'─'*w}\n  Modèle Linéaire Mixte (REML)\n{'─'*w}")
    print(f"  Observations         : {s['n_obs']}")
    print(f"  Participants         : {s['n_participants']}")
    print(f"  Convergence          : {'✔' if s['converged'] else '⚠️  non atteinte'}")

    its = s.get("interaction_terms", [])
    if its:
        print(f"  Termes d'interaction : {its}")

    print(f"\n  σ_u  (inter-part.)   : {s['sigma_u']:.4f}")
    print(f"  σ_ε  (résiduel)      : {s['sigma_eps']:.4f}")
    print(f"  ICC participant      : {s['icc_participant']:.3f}")
    print(f"\n  R²_marginal          : {s['r2_marginal']:.3f}  (effets fixes)")
    print(f"  R²_conditionnel      : {s['r2_conditional']:.3f}  (fixes + aléatoires)")
    print(f"  MAE in-sample        : {s['mae_cv_mean']:.3f}")
    print(f"  AIC / BIC            : {s['aic']:.1f} / {s['bic']:.1f}")
    print()

    coefs      = s["coefs"]
    main_c     = {k: v for k, v in coefs.items() if not v.get("is_background") and not v.get("is_interaction")}
    interact_c = {k: v for k, v in coefs.items() if v.get("is_interaction")}
    bg_c       = {k: v for k, v in coefs.items() if v.get("is_background")}

    header = f"  {'Predictor':<18} {'β':>8} {'SE':>8} {'z':>8} {'p':>8}"
    sep    = f"  {'─'*56}"

    # Effets principaux
    print(header)
    print(sep)
    for name, v in main_c.items():
        sig = "★" if v["p_value"] < 0.05 else " "
        print(f"  {name:<18} {v['coef']:>8.3f} {v['se']:>8.3f} {v['z']:>8.2f} {v['p_value']:>8.3f} {sig}")

    # Termes d'interaction
    if interact_c:
        print(f"\n  Termes d'interaction")
        print(sep)
        for name, v in interact_c.items():
            sig = "★" if v["p_value"] < 0.05 else " "
            label = {
                "D_sq": "D²",
                "DxP":  "D × P",
                "SxE":  "S × E",
                "DxS":  "D × S",
                "S_sq": "S²",
            }.get(name, name)
            print(f"  {label:<18} {v['coef']:>8.3f} {v['se']:>8.3f} {v['z']:>8.2f} {v['p_value']:>8.3f} {sig}")

    # Background
    if bg_c:
        print(f"\n  Bagage musical (réf. = {BACKGROUND_REF})")
        print(sep)
        for name, v in bg_c.items():
            lvl       = name.replace("bg_", "")
            label     = BACKGROUND_LABELS.get(lvl, name)
            sig       = "★" if v["p_value"] < 0.05 else " "
            direction = "↑" if v["coef"] > 0 else "↓"
            print(f"  {label:<28} β={v['coef']:>+7.3f}  p={v['p_value']:.3f} {sig}  {direction}")
        print("\n  Interprétation : β > 0 → rating groove plus élevé que les non-musiciens.")

    print(f"{'─'*w}\n")