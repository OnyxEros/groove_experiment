"""
regression/run.py  (v4 — VERSION MÉMOIRE)
=========================================
Point d'entrée du module de régression groove.

Simplifications v4 :
    - Feature set unique : acoustic (D, I, V, S, E, P)
    - Modèles : Ridge + ElasticNet + LMM uniquement
    - run_regression_all() supprimé (un seul feature set pertinent)
    - Comparaison AIC/BIC M0/M1 supprimée (degrés de liberté insuffisants)

Correctif de polarité P :
    P > 0  →  Rushing (avance temporelle du hi-hat)
    P < 0  →  Laid-back (retard temporel du hi-hat)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from regression.data.loader   import load_aggregated, load_raw_responses, describe_dataset
from regression.models         import build_models
from regression.evaluation     import evaluate_all, print_report, save_report
from config import get_current_run


# ============================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================

def run_regression(
    feature_set:      str  = "acoustic",
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

    # ── 1. Données agrégées (Ridge, ElasticNet) ───────────────────────────────
    df, X, y, features = load_aggregated(
        feature_set=feature_set,
        refresh=refresh,
        min_participants=min_participants,
        normalize=normalize,
        exclude_single=exclude_single,
    )

    # ── Correctif de polarité P ───────────────────────────────────────────────
    from config import apply_polarity_fix_array
    X = apply_polarity_fix_array(X, features)

    describe_dataset(df, features)

    if len(df) < 10:
        print(f"\n⚠️  Seulement {len(df)} stimuli après jointure.")

    # ── 2. Données brutes (LMM) ───────────────────────────────────────────────
    df_raw = load_raw_responses(
        feature_set=feature_set,
        refresh=False,
        exclude_single=exclude_single,
    )

    if df_raw is not None:
        from config import apply_polarity_fix_df
        df_raw = apply_polarity_fix_df(df_raw)

    features_for_lmm = (
        df_raw.attrs.get("features_requested", features)
        if df_raw is not None else features
    )

    # ── 3. Fit des modèles ────────────────────────────────────────────────────
    models = build_models(seed=seed)
    for model in models:
        f = features_for_lmm if model.supports_raw_data else features
        model.fit(X, y, features=f, df_raw=df_raw)

    # ── 4. Évaluation ─────────────────────────────────────────────────────────
    results = evaluate_all(models, X, y)
    print_report(results, feature_set=feature_set)

    # ── 5. Sauvegarde ─────────────────────────────────────────────────────────
    if out_dir is None:
        out_dir = _make_output_dir(feature_set)

    if save:
        save_report(
            results, df=df, features=features,
            out_dir=out_dir, df_raw=df_raw,
        )
        print(f"\n  💾  Résultats → {out_dir}")

    best = _best_model(results)

    return {
        "feature_set":    feature_set,
        "features":       features,
        "n_stimuli":      int(len(df)),
        "exclude_single": exclude_single,
        "models":         results,
        "best_model":     best,
        "best_r2":        results[best].get("r2_cv_mean") if best else None,
        "out_dir":        out_dir,
    }


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


def _best_model(results: dict) -> str | None:
    """Meilleur modèle CV parmi Ridge et ElasticNet."""
    candidates = {
        k: v for k, v in results.items()
        if k in {"Ridge", "ElasticNet"}
        and not v.get("_not_fitted")
    }
    if not candidates:
        return None
    return max(candidates, key=lambda k: candidates[k].get("r2_cv_mean", -np.inf))


def _header(msg: str) -> None:
    w = 65
    print(f"\n{'═'*w}\n  {msg}\n{'═'*w}")