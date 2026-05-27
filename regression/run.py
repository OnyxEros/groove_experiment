"""
regression/run.py  (v3)
========================
Point d'entrée du module de régression groove.

Changements v3 :
    - Fix LMM interactions : features_for_lmm depuis df_raw.attrs
    - Comparaison AIC/BIC automatique quand feature_set='interactions' :
      fit M0 (additif) vs M1 (avec interactions) en ML pour justification
      formelle de l'inclusion des termes croisés (Burnham & Anderson 2002).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from regression.data.loader   import load_aggregated, load_raw_responses, describe_dataset
from regression.models         import build_models
from regression.evaluation     import evaluate_all, print_report, save_report
from config import get_current_run


# ============================================================
# RUN — UN FEATURE SET
# ============================================================

def run_regression(
    feature_set:      str  = "all",
    refresh:          bool = False,
    min_participants: int  = 1,
    normalize:        bool = True,
    exclude_single:   bool = True,
    save:             bool = True,
    seed:             int  = 42,
    check_db:         bool = True,
    out_dir:          Path | None = None,
) -> dict:
    _header(f"Régression groove  |  features={feature_set}  |  exclude_single={exclude_single}")

    if check_db:
        _run_db_check(refresh)

    # ── 1. Données agrégées ───────────────────────────────────────────────────
    df, X, y, features = load_aggregated(
        feature_set=feature_set,
        refresh=refresh,
        min_participants=min_participants,
        normalize=normalize,
        exclude_single=exclude_single,
    )
    describe_dataset(df, features)

    if len(df) < 10:
        print(f"\n⚠️  Seulement {len(df)} stimuli après jointure.")

    # ── 2. Données brutes (LMM) ───────────────────────────────────────────────
    df_raw = load_raw_responses(
        feature_set=feature_set,
        refresh=False,
        exclude_single=exclude_single,
    )

    # Features pour le LMM : base + noms des termes d'interaction
    if df_raw is not None and "features_requested" in df_raw.attrs:
        features_for_lmm = df_raw.attrs["features_requested"]
    else:
        features_for_lmm = features

    # ── 3. Comparaison AIC/BIC si feature_set='interactions' ─────────────────
    aic_bic_comparison = {}
    if feature_set == "interactions" and df_raw is not None:
        aic_bic_comparison = _run_aic_bic_comparison(df_raw, features_for_lmm)

    # ── 4. Fit des modèles ────────────────────────────────────────────────────
    models = build_models(seed=seed)
    for model in models:
        f = features_for_lmm if model.supports_raw_data else features
        model.fit(X, y, features=f, df_raw=df_raw)

    # ── 5. Évaluation ─────────────────────────────────────────────────────────
    results = evaluate_all(models, X, y)
    print_report(results, feature_set=feature_set)

    # ── 6. Sauvegarde ─────────────────────────────────────────────────────────
    if out_dir is None:
        out_dir = _make_output_dir(feature_set)

    if save:
        save_report(
            results, df=df, features=features,
            out_dir=out_dir, df_raw=df_raw,
            extra={"aic_bic_comparison": aic_bic_comparison},
        )
        print(f"\n  💾  Résultats → {out_dir}")

    best = _best_model(results)

    return {
        "feature_set":          feature_set,
        "features":             features,
        "n_stimuli":            int(len(df)),
        "exclude_single":       exclude_single,
        "models":               results,
        "best_model":           best,
        "best_r2":              results[best].get("r2_cv_mean") if best else None,
        "out_dir":              out_dir,
        "aic_bic_comparison":   aic_bic_comparison,
    }


# ============================================================
# RUN — TOUS LES FEATURE SETS
# ============================================================

def run_regression_all(
    refresh:          bool = False,
    min_participants: int  = 1,
    normalize:        bool = True,
    exclude_single:   bool = True,
    save:             bool = True,
    seed:             int  = 42,
    check_db:         bool = True,
) -> dict:
    _header("Régression groove  |  3 feature sets — mode mémoire")

    run_root    = _make_run_root()
    all_results = {}
    check_done  = False

    for fs in ("design", "acoustic", "all"):
        result = run_regression(
            feature_set=fs,
            refresh=(refresh and not check_done),
            min_participants=min_participants,
            normalize=normalize,
            exclude_single=exclude_single,
            save=save,
            seed=seed,
            check_db=(check_db and not check_done),
            out_dir=run_root / fs if save else None,
        )
        all_results[fs] = result
        check_done = True

    _print_comparison_summary(all_results)
    return {**all_results, "run_root": run_root}


# ============================================================
# AIC/BIC COMPARISON
# ============================================================

def _run_aic_bic_comparison(
    df_raw:           pd.DataFrame,
    features_with_interactions: list[str],
) -> dict:
    """
    Compare M0 (additif) vs M1 (avec interactions) via AIC/BIC en ML.

    M0 : features de base sans termes croisés
    M1 : M0 + D_sq, DxP, SxE, DxS
    """
    from regression.models.lmm_comparison import compare_lmm_models
    from regression.data.features import INTERACTION_TERMS

    features_base = [
        f for f in features_with_interactions
        if f not in INTERACTION_TERMS
    ]

    print("\n" + "═" * 65)
    print("  Comparaison formelle M0 vs M1 — AIC/BIC (ML)")
    print("═" * 65)

    return compare_lmm_models(
        df_raw=df_raw,
        features_base=features_base,
        features_interaction=features_with_interactions,
        verbose=True,
    )


# ============================================================
# HELPERS PRIVÉS
# ============================================================

def _run_db_check(refresh: bool) -> None:
    print("\n🔍  Vérification Supabase…")
    try:
        from perception.check_supabase import check_supabase
        ok = check_supabase(refresh=False, verbose=True)
        if not ok:
            print("⚠️  Supabase check failed — tentative sur cache local")
            if refresh:
                raise RuntimeError("--refresh demandé mais Supabase inaccessible.")
    except ImportError:
        print("⚠️  perception.check_supabase introuvable — diagnostic ignoré")


def _make_output_dir(feature_set: str) -> Path:
    out = get_current_run() / "regression" / feature_set
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_run_root() -> Path:
    out = get_current_run() / "regression"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _best_model(results: dict) -> str | None:
    preferred = {"Ridge", "ElasticNet", "SVR"}
    candidates = {
        k: v for k, v in results.items()
        if k in preferred
        and not v.get("_is_lmm")
        and not v.get("_not_fitted")
    }
    if not candidates:
        candidates = {
            k: v for k, v in results.items()
            if not v.get("_is_lmm") and not v.get("_not_fitted")
        }
    if not candidates:
        return None
    return max(candidates, key=lambda k: candidates[k].get("r2_cv_mean", -np.inf))


def _header(msg: str) -> None:
    w = 65
    print(f"\n{'═'*w}\n  {msg}\n{'═'*w}")


def _print_comparison_summary(all_results: dict) -> None:
    w = 70
    print(f"\n{'─'*w}")
    print(f"  {'Feature set':<14} {'Model':<16} {'R² CV':<14} {'MAE CV':<10}")
    print(f"{'─'*w}")

    for fs in ("design", "acoustic", "all"):
        if fs not in all_results:
            continue
        for model_name, res in all_results[fs].get("models", {}).items():
            r2  = res.get("r2_cv_mean", float("nan"))
            mae = res.get("mae_cv_mean", float("nan"))
            r2s = res.get("r2_cv_std", 0)
            tag = "  ★" if model_name == all_results[fs].get("best_model") else ""
            if res.get("_is_lmm"):
                r2  = res.get("r2_marginal", float("nan"))
                r2s = 0
                tag += "  [in-sample]"
            print(f"  {fs:<14} {model_name:<16} {r2:.3f} ±{r2s:.2f}   {mae:.3f}{tag}")

    best_fs, best_model, best_r2 = _find_global_best(all_results)
    print(f"{'─'*w}")
    print(f"  🏆  Meilleur (CV) : {best_fs} / {best_model}  →  R² CV = {best_r2:.3f}")
    print(f"{'─'*w}\n")


def _find_global_best(all_results: dict) -> tuple[str, str, float]:
    preferred = {"Ridge", "ElasticNet", "SVR"}
    best = ("—", "—", -np.inf)
    for fs, res in all_results.items():
        if not isinstance(res, dict):
            continue
        for model_name, mres in res.get("models", {}).items():
            if model_name not in preferred:
                continue
            r2 = mres.get("r2_cv_mean", -np.inf)
            if isinstance(r2, float) and r2 > best[2]:
                best = (fs, model_name, r2)
    return best