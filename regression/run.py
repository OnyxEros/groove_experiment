"""
regression/run.py
=================
Point d'entrée du module de régression.

Lance une comparaison de modèles sur les données groove :
  - Ridge (linéaire, interprétable)
  - RandomForest (non-linéaire, SHAP-compatible)

Sauve les résultats dans data/analysis/run_<timestamp>/regression/.

Usage :
    from regression.run import run_regression
    result = run_regression(feature_set="all", refresh=False)

    # ou via CLI :
    python cli.py --regression --feature-set design
    python cli.py --regression --feature-set all --refresh
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from regression.data_loader import load_regression_data, describe_dataset
from regression.model import fit_models
from regression.evaluation import evaluate_models, print_report, save_report


# =========================================================
# MAIN ENTRY POINT
# =========================================================

def run_regression(
    feature_set: str = "all",
    refresh: bool = False,
    min_participants: int = 1,
    normalize: bool = True,
    save: bool = True,
    seed: int = 42,
) -> dict:
    """
    Lance le pipeline de régression complet.

    Args:
        feature_set:      "design" | "acoustic" | "all"
        refresh:          re-fetch Supabase même si cache local existe
        min_participants: filtre les stimuli sous-représentés
        normalize:        centre-réduit les features
        save:             sauve les résultats sur disque
        seed:             graine pour reproductibilité

    Returns:
        dict avec clés :
            feature_set, features, n_stimuli,
            models: {nom: {r2, mae, coefs_or_importances}},
            best_model, best_r2
    """
    print(f"\n{'═'*55}")
    print(f"  Régression groove  |  features={feature_set}")
    print(f"{'═'*55}")

    # ── 1. Données ───────────────────────────────────────
    df, X, y, features = load_regression_data(
        feature_set=feature_set,
        refresh=refresh,
        min_participants=min_participants,
        normalize=normalize,
    )
    describe_dataset(df, features)

    if len(df) < 10:
        print(
            f"⚠️  Seulement {len(df)} stimuli après jointure. "
            "Les résultats seront peu fiables. "
            "Collecte plus de réponses ou utilise --refresh."
        )

    # ── 2. Entraînement ──────────────────────────────────
    models = fit_models(X, y, features=features, seed=seed)

    # ── 3. Évaluation ────────────────────────────────────
    results = evaluate_models(models, X, y, features=features)
    print_report(results, feature_set=feature_set)

    # ── 4. Sauvegarde ────────────────────────────────────
    if save:
        out_dir = _make_output_dir(feature_set)
        save_report(results, df=df, features=features, out_dir=out_dir)
        print(f"\n  💾 Résultats sauvés → {out_dir}")

    # ── 5. Résumé retourné ───────────────────────────────
    best = max(results, key=lambda k: results[k].get("r2_cv_mean", -np.inf))

    return {
        "feature_set": feature_set,
        "features":    features,
        "n_stimuli":   len(df),
        "models":      results,
        "best_model":  best,
        "best_r2":     results[best].get("r2_cv_mean", None),
    }


# =========================================================
# HELPERS
# =========================================================

def _make_output_dir(feature_set: str) -> Path:
    """Crée et retourne le dossier de sortie horodaté."""
    from config import ANALYSIS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ANALYSIS_DIR / f"run_{timestamp}" / "regression" / feature_set
    out.mkdir(parents=True, exist_ok=True)
    return out