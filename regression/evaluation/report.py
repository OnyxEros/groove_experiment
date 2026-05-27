"""
regression/evaluation/report.py
=================================
Rapport console et sauvegarde des résultats.

Responsabilité unique : afficher et persister les résultats.
Ni cross-validation, ni figures — juste le rapport.

Changements v3 :
    - save_report accepte extra=None (dict optionnel sauvegardé dans report.json)
      Utilisé pour stocker la comparaison AIC/BIC M0 vs M1.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

from regression.evaluation.metrics import CV_RANDOM_STATE


# ============================================================
# RAPPORT CONSOLE
# ============================================================

def print_report(results: dict[str, dict], feature_set: str = "") -> None:
    header = "Résultats régression" + (f" [{feature_set}]" if feature_set else "")
    w = 55
    print(f"\n{'─'*w}\n  {header}\n{'─'*w}")

    for name, res in results.items():
        print(f"\n  {name}")
        if res.get("_is_lmm"):
            _print_lmm_block(res)
        else:
            _print_sklearn_block(res)

    print(f"\n{'─'*w}\n")


def _print_lmm_block(res: dict) -> None:
    print(f"    R²_marginal  : {res.get('r2_marginal', float('nan')):.3f}  (effets fixes, in-sample)")
    print(f"    R²_cond.     : {res.get('r2_conditional', float('nan')):.3f}  (fixes + aléatoires)")
    print(f"    MAE          : {res.get('mae_cv_mean', float('nan')):.3f}  (in-sample)")
    print(f"    ICC part.    : {res.get('icc_participant', float('nan')):.3f}")
    print(f"    Convergence  : {'✔' if res.get('converged') else '⚠️  non atteinte'}")
    aic = res.get("aic", float("nan"))
    bic = res.get("bic", float("nan"))
    aic_s = f"{aic:.1f}" if math.isfinite(aic) else "nan"
    bic_s = f"{bic:.1f}" if math.isfinite(bic) else "nan"
    print(f"    AIC / BIC    : {aic_s} / {bic_s}")

    coefs = res.get("coefs", {})
    sig   = {k: v for k, v in coefs.items()
             if isinstance(v, dict) and v.get("p_value", 1) < 0.05}
    if sig:
        print("    Coefficients LMM significatifs (p<0.05) :")
        for feat, v in sig.items():
            print(f"      {feat:<16} β={v['coef']:+.3f}  p={v['p_value']:.3f} ★")


def _print_sklearn_block(res: dict) -> None:
    print(f"    R²  (CV, {CV_RANDOM_STATE=}) : {res.get('r2_cv_mean', float('nan')):.3f}  ±{res.get('r2_cv_std', 0):.3f}")
    print(f"    MAE (CV)          : {res.get('mae_cv_mean', float('nan')):.3f} ±{res.get('mae_cv_std', 0):.3f}")

    if res.get("coefs"):
        print("    Coefficients Ridge :")
        scale = max(abs(c) for c in res["coefs"].values()) + 1e-9
        for feat, coef in list(res["coefs"].items())[:8]:
            print(f"      {feat:<14} {coef:+.3f}  {_bar(coef, scale)}")

    if res.get("importances"):
        print("    Importances RF :")
        for feat, imp in list(res["importances"].items())[:8]:
            print(f"      {feat:<14} {imp:.3f}  {_bar(imp, 1.0, signed=False)}")


# ============================================================
# SAUVEGARDE
# ============================================================

def save_report(
    results:  dict[str, dict],
    df:       pd.DataFrame,
    features: list[str],
    out_dir:  Path,
    df_raw:   pd.DataFrame | None = None,
    extra:    dict | None = None,
) -> None:
    """
    Sauvegarde les résultats (JSON) et génère toutes les figures.

    extra : dict optionnel fusionné dans report.json
            Typiquement : {"aic_bic_comparison": {...}} depuis run.py
    """
    from regression.viz import RegressionFigure

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ─────────────────────────────────────────────────────────────────
    serializable = {
        k: {kk: vv for kk, vv in v.items()
            if not isinstance(vv, np.ndarray) and not kk.startswith("_")}
        for k, v in results.items()
    }

    report_data: dict = {"features": features, "results": serializable}
    if extra:
        report_data["extra"] = extra

    with open(out_dir / "report.json", "w") as f:
        json.dump(_make_serializable(report_data), f, indent=2)

    # ── Figures ──────────────────────────────────────────────────────────────
    RegressionFigure().plot(
        results=results,
        df=df,
        features=features,
        out_dir=fig_dir,
        df_raw=df_raw,
        verbose=True,
    )


# ============================================================
# HELPERS
# ============================================================

def _make_serializable(obj):
    if isinstance(obj, bool):          return obj
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, dict):          return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_make_serializable(v) for v in obj]
    return obj


def _bar(value: float, scale: float = 1.0, signed: bool = True, width: int = 18) -> str:
    if signed:
        n = min(int(abs(value) / (scale + 1e-9) * (width // 2)), width // 2)
        return (
            (" " * (width // 2 - n) + "█" * n)
            if value < 0
            else (" " * (width // 2) + "█" * n)
        )
    return "█" * min(int(value / (scale + 1e-9) * width), width)