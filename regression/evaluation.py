"""
regression/evaluation.py
========================
Évaluation et rapport des modèles de régression groove.

Métriques :
    R²  cross-validé (5-fold)  — mesure principale
    MAE cross-validé           — erreur absolue moyenne (unité : rating)
    Coefficients (Ridge)       — contribution linéaire de chaque feature
    Feature importances (RF)   — contribution non-linéaire
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


# =========================================================
# ÉVALUATION
# =========================================================

def evaluate_models(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    cv: int = 5,
) -> dict:
    """
    Évalue chaque modèle par cross-validation et extrait l'interprétabilité.

    Returns:
        dict {
            "Ridge": {
                r2_cv_mean, r2_cv_std,
                mae_cv_mean, mae_cv_std,
                coefs: {feature: coef},
            },
            "RandomForest": {
                r2_cv_mean, r2_cv_std,
                mae_cv_mean, mae_cv_std,
                importances: {feature: importance},
            }
        }
    """
    results = {}

    for name, model in models.items():
        cv_scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring=["r2", "neg_mean_absolute_error"],
            return_train_score=False,
        )

        entry = {
            "r2_cv_mean":  float(np.mean(cv_scores["test_r2"])),
            "r2_cv_std":   float(np.std(cv_scores["test_r2"])),
            "mae_cv_mean": float(np.mean(-cv_scores["test_neg_mean_absolute_error"])),
            "mae_cv_std":  float(np.std(-cv_scores["test_neg_mean_absolute_error"])),
        }

        # ── Interprétabilité ─────────────────────────────
        if name == "Ridge":
            coefs = _extract_ridge_coefs(model, features)
            entry["coefs"] = coefs

        elif name == "RandomForest":
            importances = dict(zip(features, model.feature_importances_.tolist()))
            # tri décroissant
            entry["importances"] = dict(
                sorted(importances.items(), key=lambda x: x[1], reverse=True)
            )

        results[name] = entry

    return results


# =========================================================
# RAPPORT CONSOLE
# =========================================================

def print_report(results: dict, feature_set: str = "") -> None:
    """Affiche un rapport lisible dans le terminal."""
    header = f"Résultats régression" + (f" [{feature_set}]" if feature_set else "")
    print(f"\n{'─'*55}")
    print(f"  {header}")
    print(f"{'─'*55}")

    for name, res in results.items():
        r2   = res["r2_cv_mean"]
        r2s  = res["r2_cv_std"]
        mae  = res["mae_cv_mean"]
        maes = res["mae_cv_std"]

        print(f"\n  {name}")
        print(f"    R²  (CV)  : {r2:.3f}  ±{r2s:.3f}")
        print(f"    MAE (CV)  : {mae:.3f} ±{maes:.3f}")

        if "coefs" in res:
            print(f"    Coefficients (Ridge) :")
            for feat, coef in sorted(res["coefs"].items(), key=lambda x: abs(x[1]), reverse=True):
                bar = _bar(coef, scale=0.5)
                print(f"      {feat:<14} {coef:+.3f}  {bar}")

        elif "importances" in res:
            print(f"    Importances (RF) :")
            for feat, imp in res["importances"].items():
                bar = _bar(imp, scale=1.0, signed=False)
                print(f"      {feat:<14} {imp:.3f}  {bar}")

    print(f"\n{'─'*55}\n")


# =========================================================
# SAUVEGARDE
# =========================================================

def save_report(
    results: dict,
    df: pd.DataFrame,
    features: list[str],
    out_dir: Path,
) -> None:
    """
    Sauve le rapport JSON et le CSV des prédictions.

    Fichiers :
        out_dir/report.json      — métriques + coefficients
        out_dir/predictions.csv  — stim_id, groove_mean, groove_pred_ridge, groove_pred_rf
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "report.json", "w") as f:
        json.dump({"features": features, "results": results}, f, indent=2)


# =========================================================
# HELPERS
# =========================================================

def _extract_ridge_coefs(model, features: list[str]) -> dict:
    """Extrait les coefficients du Ridge (gère le Pipeline avec StandardScaler)."""
    if isinstance(model, Pipeline):
        ridge = model.named_steps["ridge"]
    else:
        ridge = model
    return dict(sorted(
        zip(features, ridge.coef_.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    ))


def _bar(value: float, scale: float = 1.0, signed: bool = True, width: int = 20) -> str:
    """Petite barre ASCII pour visualiser un coefficient ou une importance."""
    if signed:
        n = int(abs(value) / scale * (width // 2))
        n = min(n, width // 2)
        if value >= 0:
            return " " * (width // 2) + "█" * n
        else:
            return " " * (width // 2 - n) + "█" * n
    else:
        n = int(value / scale * width)
        n = min(n, width)
        return "█" * n