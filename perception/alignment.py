"""
perception/alignment.py
=======================
Alignement de l'espace latent acoustique avec les ratings perceptifs.

Modèle : Ridge avec sélection d'alpha par CV interne (RidgeCV).
Évaluation : R² cross-validé 5-fold — seule mesure défendable dans un mémoire.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================================================
# FIT
# =========================================================

def fit_alignment(
    Z:       np.ndarray,
    ratings: np.ndarray,
    cv:      int = 5,
    seed:    int = 42,
) -> tuple[Pipeline, dict]:
    """
    Apprend le mapping espace latent → perception (groove_mean).

    Args:
        Z:       features acoustiques, shape (n_stimuli, n_features)
        ratings: groove_mean par stimulus, shape (n_stimuli,)
        cv:      nombre de folds pour la cross-validation (défaut : 5)
        seed:    graine pour la reproductibilité

    Returns:
        model   : Pipeline sklearn entraîné sur l'ensemble du dataset
        metrics : dict {
            r2_cv_mean, r2_cv_std,   ← R² cross-validé (mesure principale)
            r2_train,                ← R² in-sample (à titre indicatif seulement)
            n_samples, n_features,
            best_alpha,
        }

    Note :
        r2_train est toujours supérieur ou égal à r2_cv_mean.
        Pour le mémoire, seul r2_cv_mean est rapportable.
        Un r2_cv_mean négatif signifie que le modèle est pire qu'une constante.
    """
    Z       = np.asarray(Z,       dtype=np.float64)
    ratings = np.asarray(ratings, dtype=np.float64)

    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if len(Z) != len(ratings):
        raise ValueError(
            f"Mismatch : Z a {len(Z)} lignes, ratings en a {len(ratings)}."
        )

    if len(Z) < cv + 1:
        raise ValueError(
            f"Pas assez de données ({len(Z)} stimuli) pour une CV à {cv} folds. "
            f"Collecte plus de réponses ou baisse cv."
        )

    # ── Pipeline : StandardScaler + RidgeCV ──────────────
    # RidgeCV sélectionne alpha par LOO interne (rapide, pas de data leakage
    # car c'est une CV séparée de la CV d'évaluation externe).
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(
            alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv=5,
            scoring="r2",
        )),
    ])

    # ── Évaluation cross-validée (externe) ───────────────
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, Z, ratings, cv=kf, scoring="r2")

    # ── Fit final sur l'ensemble du dataset ──────────────
    model.fit(Z, ratings)
    r2_train   = float(model.score(Z, ratings))
    best_alpha = float(model.named_steps["ridge"].alpha_)

    metrics = {
        "r2_cv_mean":  float(np.mean(cv_scores)),
        "r2_cv_std":   float(np.std(cv_scores)),
        "r2_train":    r2_train,
        "n_samples":   int(len(Z)),
        "n_features":  int(Z.shape[1]),
        "best_alpha":  best_alpha,
        "cv_scores":   cv_scores.tolist(),
    }

    return model, metrics


# =========================================================
# PREDICT
# =========================================================

def predict_perception(model: Pipeline, Z: np.ndarray) -> np.ndarray:
    """
    Prédit le groove perçu depuis l'espace latent.

    Args:
        model : Pipeline retourné par fit_alignment
        Z     : features, shape (n, n_features)

    Returns:
        y_pred : np.ndarray shape (n,)
    """
    return model.predict(np.asarray(Z, dtype=np.float64))


# =========================================================
# RAPPORT CONSOLE
# =========================================================

def print_alignment_report(metrics: dict, label: str = "") -> None:
    """Affiche les métriques d'alignement dans le terminal."""
    w = 48
    header = f"Alignement perceptif" + (f"  [{label}]" if label else "")
    print(f"\n{'─'*w}")
    print(f"  {header}")
    print(f"{'─'*w}")
    print(f"  Stimuli      : {metrics['n_samples']}")
    print(f"  Features     : {metrics['n_features']}")
    print(f"  Alpha Ridge  : {metrics['best_alpha']}")
    print(f"  R² train     : {metrics['r2_train']:.3f}  (in-sample, indicatif)")
    print(f"  R² CV ({len(metrics['cv_scores'])}-fold) : "
          f"{metrics['r2_cv_mean']:.3f}  ±{metrics['r2_cv_std']:.3f}  ← rapportable")

    if metrics["r2_cv_mean"] < 0:
        print("  ⚠️  R² CV négatif — le modèle est pire qu'une constante")
        print("     Possible causes : trop peu de données, features non pertinentes")
    elif metrics["r2_cv_mean"] < 0.1:
        print("  ⚠️  R² CV faible — signal perceptif difficile à capturer")

    print(f"{'─'*w}\n")