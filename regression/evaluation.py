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
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
 
def evaluate_models(models, X, y, features, cv=5):
    results = {}
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    for name, model in models.items():
        cv_scores = cross_validate(
            model, X, y, cv=kf,
            scoring=["r2", "neg_mean_absolute_error"],
            return_train_score=False,
        )
        entry = {
            "r2_cv_mean":  float(np.mean(cv_scores["test_r2"])),
            "r2_cv_std":   float(np.std(cv_scores["test_r2"])),
            "mae_cv_mean": float(np.mean(-cv_scores["test_neg_mean_absolute_error"])),
            "mae_cv_std":  float(np.std(-cv_scores["test_neg_mean_absolute_error"])),
        }
        # Out-of-fold predictions
        from sklearn.base import clone
        y_pred_oof = np.zeros(len(y))
        for train_idx, test_idx in kf.split(X):
            m = clone(model); m.fit(X[train_idx], y[train_idx])
            y_pred_oof[test_idx] = m.predict(X[test_idx])
        entry["y_pred_oof"] = y_pred_oof.tolist()
        # Fit final sur tout le dataset
        model.fit(X, y)
        if name == "Ridge":
            entry["coefs"] = _extract_ridge_coefs(model, features)
        elif name == "RandomForest":
            importances = dict(zip(features, model.feature_importances_.tolist()))
            entry["importances"] = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            shap_vals = _compute_shap(model, X)
            if shap_vals is not None:
                entry["shap_values"] = shap_vals
        results[name] = entry
    return results
 
def print_report(results, feature_set=""):
    header = "Résultats régression" + (f" [{feature_set}]" if feature_set else "")
    w = 55
    print(f"\n{'─'*w}\n  {header}\n{'─'*w}")
    for name, res in results.items():
        print(f"\n  {name}")
        print(f"    R²  (CV) : {res['r2_cv_mean']:.3f}  ±{res['r2_cv_std']:.3f}")
        print(f"    MAE (CV) : {res['mae_cv_mean']:.3f} ±{res['mae_cv_std']:.3f}")
        if "coefs" in res:
            print("    Coefficients Ridge :")
            scale = max(abs(c) for c in res["coefs"].values()) + 1e-9
            for feat, coef in list(res["coefs"].items())[:8]:
                print(f"      {feat:<14} {coef:+.3f}  {_bar(coef, scale)}")
        elif "importances" in res:
            print("    Importances RF :")
            for feat, imp in list(res["importances"].items())[:8]:
                print(f"      {feat:<14} {imp:.3f}  {_bar(imp, 1.0, signed=False)}")
        if "shap_values" in res:
            sv = res["shap_values"]
            mean_abs = np.abs(sv).mean(axis=0)
            feats = list(res.get("importances", {}).keys())
            if feats:
                top = sorted(zip(feats, mean_abs), key=lambda x: x[1], reverse=True)[:5]
                print("    SHAP top-5 :")
                for feat, v in top:
                    print(f"      {feat:<14} |SHAP|={v:.4f}")
    print(f"\n{'─'*w}\n")
 
def save_report(results, df, features, out_dir):
    from regression.figures import (
        plot_coefficients, plot_prediction_scatter, plot_shap_summary
    )
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # JSON (sans arrays numpy)
    serializable = {
        name: {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
        for name, res in results.items()
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump({"features": features, "results": serializable}, f, indent=2)
    # Figures
    ridge_res = results.get("Ridge", {})
    rf_res    = results.get("RandomForest", {})
    if "coefs" in ridge_res and "importances" in rf_res:
        plot_coefficients(
            ridge_coefs=ridge_res["coefs"],
            rf_importances=rf_res["importances"],
            out_path=fig_dir / "coefficients.png",
        )
    if "shap_values" in rf_res and all(f in df.columns for f in features):
        plot_shap_summary(
            shap_values=rf_res["shap_values"],
            X=df[features].values,
            features=features,
            out_path=fig_dir / "shap_summary.png",
        )
    best = max(results, key=lambda k: results[k].get("r2_cv_mean", -np.inf))
    best_res = results[best]
    if "y_pred_oof" in best_res and "groove_mean" in df.columns:
        plot_prediction_scatter(
            y_true=df["groove_mean"].values,
            y_pred=np.array(best_res["y_pred_oof"]),
            model_name=best,
            r2=best_res["r2_cv_mean"],
            mae=best_res["mae_cv_mean"],
            out_path=fig_dir / "prediction_scatter.png",
        )
 
def _extract_ridge_coefs(model, features):
    ridge = model.named_steps["ridge"] if isinstance(model, Pipeline) else model
    return dict(sorted(zip(features, ridge.coef_.tolist()), key=lambda x: abs(x[1]), reverse=True))
 
def _compute_shap(model, X):
    try:
        import shap
        rf = model.named_steps.get("rf", model)
        sv = shap.TreeExplainer(rf).shap_values(X)
        return np.array(sv[0] if isinstance(sv, list) else sv)
    except ImportError:
        print("  [shap] pip install shap pour activer SHAP")
        return None
    except Exception as e:
        print(f"  [shap] Erreur : {e}")
        return None
 
def _bar(value, scale=1.0, signed=True, width=18):
    if signed:
        n = min(int(abs(value) / (scale + 1e-9) * (width // 2)), width // 2)
        return (" " * (width // 2 - n) + "█" * n) if value < 0 else (" " * (width // 2) + "█" * n)
    return "█" * min(int(value / (scale + 1e-9) * width), width)