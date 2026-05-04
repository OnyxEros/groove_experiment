"""
regression/run.py
=================
Point d'entrée du module de régression groove.

Deux modes :
    run_regression(feature_set)  — un seul feature set
    run_regression_all()         — les 3 feature sets + figure comparative

Sauve les résultats dans data/analysis/run_<timestamp>/regression/.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from datetime import datetime

from regression.data_loader import load_regression_data, describe_dataset
from regression.model import fit_models
from regression.evaluation import evaluate_models, print_report, save_report


# =========================================================
# RUN — UN FEATURE SET
# =========================================================

def run_regression(
    feature_set:      str  = "all",
    refresh:          bool = False,
    min_participants: int  = 1,
    normalize:        bool = True,
    save:             bool = True,
    seed:             int  = 42,
    check_db:         bool = True,
    out_dir:          Path | None = None,
) -> dict:
    """
    Lance le pipeline de régression pour un feature set.

    Args:
        feature_set:      "design" | "acoustic" | "all"
        refresh:          re-fetch Supabase même si cache local existe
        min_participants: filtre les stimuli sous-représentés
        normalize:        centre-réduit les features (recommandé pour Ridge)
        save:             sauve les résultats + figures sur disque
        seed:             graine pour reproductibilité
        check_db:         diagnostic Supabase avant la régression
        out_dir:          dossier de sortie (auto si None)

    Returns:
        dict {
            feature_set, features, n_stimuli,
            models: {nom: {r2_cv_mean, mae_cv_mean, coefs | importances}},
            best_model, best_r2,
            out_dir,
        }
    """
    _header(f"Régression groove  |  features={feature_set}")

    # ── 0. Diagnostic Supabase ────────────────────────────
    if check_db:
        print("\n🔍  Vérification Supabase…")
        try:
            from perception.check_supabase import check_supabase
            ok = check_supabase(refresh=False, verbose=True)
            if not ok:
                print("⚠️  Supabase check failed — tentative sur cache local")
                if refresh:
                    raise RuntimeError(
                        "--refresh demandé mais Supabase inaccessible. "
                        "Vérifie SUPABASE_URL / SUPABASE_KEY."
                    )
        except ImportError:
            print("⚠️  perception.check_supabase introuvable — diagnostic ignoré")

    # ── 1. Données ────────────────────────────────────────
    df, X, y, features = load_regression_data(
        feature_set=feature_set,
        refresh=refresh,
        min_participants=min_participants,
        normalize=normalize,
    )
    describe_dataset(df, features)

    if len(df) < 10:
        print(
            f"\n⚠️  Seulement {len(df)} stimuli après jointure.\n"
            "   Les résultats seront peu fiables.\n"
            "   → Collecte plus de réponses puis relance avec --refresh."
        )

    # ── 2. Entraînement ──────────────────────────────────
    models = fit_models(X, y, features=features, seed=seed)

    # ── 3. Évaluation ────────────────────────────────────
    results = evaluate_models(models, X, y, features=features)
    print_report(results, feature_set=feature_set)

    # ── 4. Sauvegarde ────────────────────────────────────
    if out_dir is None:
        out_dir = _make_output_dir(feature_set)

    if save:
        save_report(results, df=df, features=features, out_dir=out_dir)
        print(f"\n  💾  Résultats → {out_dir}")

    # ── 5. Résumé ─────────────────────────────────────────
    best = max(results, key=lambda k: results[k].get("r2_cv_mean", -np.inf))

    return {
        "feature_set": feature_set,
        "features":    features,
        "n_stimuli":   int(len(df)),
        "models":      results,
        "best_model":  best,
        "best_r2":     results[best].get("r2_cv_mean"),
        "out_dir":     out_dir,
    }


# =========================================================
# RUN — TOUS LES FEATURE SETS (mode mémoire)
# =========================================================

def run_regression_all(
    refresh:          bool = False,
    min_participants: int  = 1,
    normalize:        bool = True,
    save:             bool = True,
    seed:             int  = 42,
    check_db:         bool = True,
) -> dict:
    """
    Lance la régression sur les 3 feature sets et produit la figure comparative.

    Séquence :
        1. design   — paramètres manipulés (S_mv, D_mv, E)
        2. acoustic — métriques réalisées  (D, I, V, S_real, E_real)
        3. all      — union des deux

    La figure comparison_bar.png est produite à la racine du dossier de run,
    au même niveau que les 3 sous-dossiers design / acoustic / all.

    Returns:
        dict {
            "design":   résultat run_regression("design"),
            "acoustic": résultat run_regression("acoustic"),
            "all":      résultat run_regression("all"),
            "comparison_figure": Path ou None,
        }
    """
    _header("Régression groove  |  3 feature sets — mode mémoire")

    # Dossier partagé pour le run complet
    run_root = _make_run_root()

    all_results: dict[str, dict] = {}
    check_done = False

    for fs in ("design", "acoustic", "all"):
        result = run_regression(
            feature_set=fs,
            refresh=(refresh and not check_done),   # fetch Supabase une seule fois
            min_participants=min_participants,
            normalize=normalize,
            save=save,
            seed=seed,
            check_db=(check_db and not check_done),
            out_dir=run_root / fs if save else None,
        )
        all_results[fs] = result
        check_done = True

    # ── Figure comparative ────────────────────────────────
    fig_path = None
    if save:
        try:
            from regression.figures import plot_comparison_bar

            # Restructure pour plot_comparison_bar :
            # {feature_set: {model_name: {r2_cv_mean, …}}}
            plot_data: dict[str, dict] = {}
            for fs, res in all_results.items():
                plot_data[fs] = res["models"]

            fig_path = run_root / "comparison_bar.png"
            plot_comparison_bar(all_results=plot_data, out_path=fig_path)
            print(f"\n  📊  Figure comparative → {fig_path}")

        except Exception as exc:
            print(f"  ⚠️  Figure comparative échouée : {exc}")

    # ── Résumé final ──────────────────────────────────────
    _print_comparison_summary(all_results)

    return {**all_results, "comparison_figure": fig_path, "run_root": run_root}


# =========================================================
# HELPERS
# =========================================================

def _make_output_dir(feature_set: str) -> Path:
    from config import ANALYSIS_DIR
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ANALYSIS_DIR / f"run_{ts}" / "regression" / feature_set
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_run_root() -> Path:
    from config import ANALYSIS_DIR
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ANALYSIS_DIR / f"run_{ts}" / "regression"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _header(msg: str) -> None:
    w = 55
    print(f"\n{'═'*w}\n  {msg}\n{'═'*w}")


def _print_comparison_summary(all_results: dict) -> None:
    """Affiche un tableau récapitulatif des 3 feature sets."""
    w = 65
    print(f"\n{'─'*w}")
    print(f"  {'Feature set':<14} {'Model':<16} {'R² CV':<12} {'MAE CV':<10}")
    print(f"{'─'*w}")

    for fs in ("design", "acoustic", "all"):
        if fs not in all_results:
            continue
        models_res = all_results[fs].get("models", {})
        for model_name, res in models_res.items():
            r2  = res.get("r2_cv_mean", np.nan)
            mae = res.get("mae_cv_mean", np.nan)
            r2s = res.get("r2_cv_std",  0)
            tag = "  ★" if fs == all_results[fs].get("best_model") else ""
            print(
                f"  {fs:<14} {model_name:<16} "
                f"{r2:.3f} ±{r2s:.2f}   {mae:.3f}{tag}"
            )

    # Meilleur modèle global
    best_fs, best_model, best_r2 = _find_global_best(all_results)
    print(f"{'─'*w}")
    print(f"  🏆  Meilleur : {best_fs} / {best_model}  →  R² CV = {best_r2:.3f}")
    print(f"{'─'*w}\n")


def _find_global_best(
    all_results: dict,
) -> tuple[str, str, float]:
    """Retourne (feature_set, model_name, r2) du meilleur modèle global."""
    best = ("—", "—", -np.inf)
    for fs, res in all_results.items():
        if not isinstance(res, dict):
            continue
        for model_name, mres in res.get("models", {}).items():
            r2 = mres.get("r2_cv_mean", -np.inf)
            if r2 > best[2]:
                best = (fs, model_name, r2)
    return best