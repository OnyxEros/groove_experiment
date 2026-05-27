"""
regression/evaluation/metrics.py
==================================
Cross-validation et métriques pour tous les modèles.

Plus de isinstance(model, dict) — on utilise model.supports_raw_data
et model.get_results() pour différencier LMM des modèles sklearn.

Corrections intégrées :
    M1 — LMM évalué séparément (pas de cross_validate sklearn sur lui)
    M2 — SHAP calculé sur un fold OOF (pas de data leakage)
    P3 — KFold random_state=42 systématique
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.base import clone

from regression.models.base import GrooveModel

CV_RANDOM_STATE = 42   # documenter dans le mémoire : seed=42 pour tous les splits


def evaluate_all(
    models:   list[GrooveModel],
    X:        np.ndarray,
    y:        np.ndarray,
    cv:       int = 5,
) -> dict[str, dict]:
    """
    Évalue chaque modèle et retourne un dict {nom: métriques}.

    Les modèles sklearn (Ridge, RF) sont cross-validés.
    Le LMM est évalué in-sample via ses propres métriques (R²_marginal, MAE).
    """
    kf      = KFold(n_splits=cv, shuffle=True, random_state=CV_RANDOM_STATE)
    results = {}

    for model in models:
        if not model.is_fitted():
            print(f"  [eval] {model.name} non fitté — ignoré")
            continue

        if model.supports_raw_data:
            # LMM : métriques déjà calculées lors du fit
            results[model.name] = model.get_results()
        else:
            results[model.name] = _evaluate_sklearn(model, X, y, kf)

    return results


# ============================================================
# ÉVALUATION SKLEARN (Ridge, RandomForest)
# ============================================================

def _evaluate_sklearn(
    model: GrooveModel,
    X:     np.ndarray,
    y:     np.ndarray,
    kf:    KFold,
) -> dict:
    # On a besoin de l'estimateur sklearn sous-jacent pour cross_validate
    estimator = _get_sklearn_estimator(model)

    cv_scores = cross_validate(
        estimator, X, y, cv=kf,
        scoring=["r2", "neg_mean_absolute_error"],
        return_train_score=False,
    )

    entry = {
        **model.get_results(),
        "r2_cv_mean":  float(np.mean(cv_scores["test_r2"])),
        "r2_cv_std":   float(np.std(cv_scores["test_r2"])),
        "mae_cv_mean": float(np.mean(-cv_scores["test_neg_mean_absolute_error"])),
        "mae_cv_std":  float(np.std(-cv_scores["test_neg_mean_absolute_error"])),
        "y_pred_oof":  _oof_predictions(estimator, X, y, kf).tolist(),
    }

    # SHAP OOF pour RandomForest uniquement
    if model.name == "RandomForest":
        shap_vals, shap_X = _compute_shap_oof(estimator, X, y, kf)
        if shap_vals is not None:
            entry["shap_values"] = shap_vals
            entry["shap_X"]      = shap_X

    return entry


def _get_sklearn_estimator(model: GrooveModel):
    """Extrait l'estimateur sklearn depuis le modèle wrapper."""
    # RidgeModel expose _model directement
    # RandomForestModel expose aussi .estimator
    if hasattr(model, "estimator"):
        return model.estimator
    if hasattr(model, "_model"):
        return model._model
    raise AttributeError(f"Impossible d'extraire l'estimateur sklearn de {model.name}")


def _oof_predictions(estimator, X, y, kf: KFold) -> np.ndarray:
    """Prédictions out-of-fold pour le scatter prédit vs observé."""
    y_pred = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        m = clone(estimator)
        m.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = m.predict(X[test_idx])
    return y_pred


def _compute_shap_oof(
    estimator,
    X:  np.ndarray,
    y:  np.ndarray,
    kf: KFold,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """SHAP values sur le premier fold OOF — correction M2."""
    try:
        import shap
    except ImportError:
        print("  [shap] pip install shap pour activer SHAP")
        return None, None

    try:
        train_idx, val_idx = next(iter(kf.split(X)))
        m = clone(estimator)
        m.fit(X[train_idx], y[train_idx])
        explainer = shap.TreeExplainer(m)
        sv        = explainer.shap_values(X[val_idx])
        sv_array  = np.array(sv[0] if isinstance(sv, list) else sv)
        return sv_array, X[val_idx]
    except Exception as e:
        print(f"  [shap] Erreur : {e}")
        return None, None
