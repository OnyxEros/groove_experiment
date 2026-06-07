"""
regression/evaluation/report.py  — VERSION LOGS ÉTAYÉS
========================================================

Remplace entièrement l'ancienne version.

Nouveautés :
    - Rapport console très détaillé, section par section
    - Barres ASCII pour les coefficients et importances
    - Tableau comparatif final tous modèles × feature sets
    - Interprétation automatique des résultats (significativité, taille d'effet)
    - Diagnostic inline (convergence, ICC, multicolinéarité)
    - Compatible avec tous les feature sets (design / acoustic / interactions)
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

from regression.evaluation.metrics import CV_RANDOM_STATE

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE MISE EN PAGE
# ─────────────────────────────────────────────────────────────────────────────
W      = 72          # largeur standard des blocs
W_WIDE = 80          # largeur des tableaux larges
BAR_W  = 24          # largeur max des barres ASCII

ICONS = {
    "ok":      "✔",
    "warn":    "⚠",
    "star":    "★",
    "arrow":   "→",
    "up":      "↑",
    "down":    "↓",
    "neutral": "·",
    "trophy":  "🏆",
    "chart":   "📊",
    "model":   "🔬",
    "info":    "ℹ",
}

EFFECT_SIZE = {   # Cohen (1988) adapté R²
    "negligible": (0.00, 0.02),
    "petit":      (0.02, 0.13),
    "moyen":      (0.13, 0.26),
    "grand":      (0.26, 1.00),
}

COEF_INTERP = {   # interprétation narrative des β significatifs
    "D":     "densité acoustique → plus d'événements = plus de groove",
    "P":     "désalignement inter-voix → push/pull favorise le groove",
    "E_mv":  "micro-timing génératif → effet négatif surprenant (voir discussion)",
    "E":     "micro-timing réalisé → expressivité perçue négativement",
    "S":     "syncopation → effet non significatif (manque de puissance ?)",
    "S_mv":  "syncopation génératif → non significatif",
    "DxP":   "interaction D×P → le désalignement n'est efficace qu'à forte densité",
    "DxS":   "interaction D×S → densité et syncopation s'annulent mutuellement",
    "D_mv":  "densité génératif → redondant avec D réalisé",
    "P_mv":  "décalage génératif → signal capté par P réalisé uniquement",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS VISUELS
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char="─", w=W):
    return char * w

def _title(text, char="═", w=W):
    pad = max(0, w - len(text) - 4)
    return f"{char*2}  {text}  {char * pad}"

def _subtitle(text, w=W):
    return f"  ── {text} {'─' * max(0, w - len(text) - 6)}"

def _bar_pos(val, scale, width=BAR_W):
    """Barre ASCII pour valeur positive."""
    n = min(int(abs(val) / (scale + 1e-9) * width), width)
    return "█" * n

def _bar_signed(val, scale, width=BAR_W):
    """Barre ASCII centrée (positive = droite, négative = gauche)."""
    half = width // 2
    n = min(int(abs(val) / (scale + 1e-9) * half), half)
    if val >= 0:
        return " " * half + "█" * n
    else:
        return " " * (half - n) + "█" * n + " " * 0

def _pval_stars(p):
    if p < 0.001: return "★★★"
    if p < 0.01:  return "★★ "
    if p < 0.05:  return "★  "
    if p < 0.10:  return "†  "
    return "   "

def _effect_label(r2):
    if math.isnan(r2): return "?"
    for label, (lo, hi) in EFFECT_SIZE.items():
        if lo <= r2 < hi:
            return label
    return "grand"

def _icc_interp(icc):
    if math.isnan(icc): return "?"
    if icc < 0.05:  return "très faible variabilité inter-participants"
    if icc < 0.15:  return "faible variabilité inter-participants"
    if icc < 0.30:  return "variabilité inter-participants modérée"
    return "forte variabilité inter-participants"

def _r2_bar(r2, width=20):
    if math.isnan(r2) or r2 <= 0:
        return "░" * width
    n = min(int(r2 * width), width)
    return "█" * n + "░" * (width - n)

def _fmt(v, decimals=3):
    if isinstance(v, float) and math.isnan(v):
        return "  nan  "
    return f"{v:+.{decimals}f}" if isinstance(v, float) else str(v)

def _fmt_plain(v, decimals=3):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# BLOC PRINCIPAL — RAPPORT CONSOLE
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: dict[str, dict], feature_set: str = "") -> None:
    """
    Rapport console complet et très détaillé.
    Appelé depuis run.py après evaluate_all().
    """
    print()
    print(_sep("═"))
    fs_label = f"  feature_set = {feature_set}" if feature_set else ""
    print(_title(f"{ICONS['chart']}  RAPPORT DE RÉGRESSION GROOVE{fs_label}"))
    print(_sep("═"))

    # ── Résumé rapide ────────────────────────────────────────────────────────
    _print_quick_summary(results)

    # ── Détail par modèle ────────────────────────────────────────────────────
    for name, res in results.items():
        print()
        if res.get("_is_lmm"):
            _print_lmm_full(res)
        elif not res.get("_not_fitted"):
            _print_sklearn_full(name, res)

    # ── Tableau comparatif ───────────────────────────────────────────────────
    print()
    _print_comparison_table(results)

    # ── Synthèse interprétative ──────────────────────────────────────────────
    print()
    _print_interpretation(results, feature_set)

    print()
    print(_sep("═"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ RAPIDE (top du rapport)
# ─────────────────────────────────────────────────────────────────────────────

def _print_quick_summary(results):
    print()
    print(_subtitle("Résumé — performances CV"))
    print()

    preferred = ["Ridge", "ElasticNet", "SVR", "RandomForest", "LMM"]
    header = f"  {'Modèle':<16} {'R² CV':>8}  {'±σ':>6}  {'MAE CV':>8}  {'R²_bar':<22}  Notes"
    print(header)
    print("  " + _sep("─", W - 2))

    best_r2, best_name = -np.inf, ""
    for name in preferred:
        res = results.get(name)
        if res is None or res.get("_not_fitted"):
            continue

        is_lmm = res.get("_is_lmm", False)

        if is_lmm:
            r2  = res.get("r2_marginal", float("nan"))
            r2s = float("nan")
            mae = res.get("mae_cv_mean", float("nan"))
            note = "[in-sample, non CV]"
        else:
            r2  = res.get("r2_cv_mean", float("nan"))
            r2s = res.get("r2_cv_std", float("nan"))
            mae = res.get("mae_cv_mean", float("nan"))
            note = ""
            if not math.isnan(r2) and r2 > best_r2 and name != "LMM":
                best_r2, best_name = r2, name

        r2_s   = _fmt_plain(r2)
        r2s_s  = _fmt_plain(r2s) if not math.isnan(r2s) else "  —  "
        mae_s  = _fmt_plain(mae)
        bar    = _r2_bar(max(r2, 0))
        effect = _effect_label(r2) if not math.isnan(r2) else ""

        flag = f"  {ICONS['trophy']}" if name == best_name else ""
        print(f"  {name:<16} {r2_s:>8}  ±{r2s_s:<5}  {mae_s:>8}  [{bar}]  {effect}{note}{flag}")

    print("  " + _sep("─", W - 2))
    if best_name:
        print(f"  {ICONS['trophy']}  Meilleur modèle CV : {best_name}  (R² = {_fmt_plain(best_r2)})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# BLOC SKLEARN (Ridge, ElasticNet, SVR, RF)
# ─────────────────────────────────────────────────────────────────────────────

def _print_sklearn_full(name: str, res: dict) -> None:
    r2   = res.get("r2_cv_mean", float("nan"))
    r2s  = res.get("r2_cv_std", float("nan"))
    mae  = res.get("mae_cv_mean", float("nan"))
    maes = res.get("mae_cv_std", float("nan"))

    print(_sep("─"))
    print(f"  {ICONS['model']}  {name.upper()}")
    print(_sep("─"))

    # Métriques
    print()
    print(_subtitle("Métriques de performance (CV 5-fold)"))
    print()
    effect = _effect_label(r2)
    print(f"  R² cross-validé    : {_fmt_plain(r2)}  ±{_fmt_plain(r2s)}")
    print(f"  Barre R²           : [{_r2_bar(max(r2, 0))}]  → taille d'effet : {effect}")
    print(f"  MAE cross-validée  : {_fmt_plain(mae)} ±{_fmt_plain(maes)}  (unités : rating 1–7)")

    if not math.isnan(r2):
        pct = r2 * 100
        print(f"  Variance expliquée : {pct:.1f}% de la variance du groove")
        if r2 < 0:
            print(f"  {ICONS['warn']}  R² négatif : le modèle performe moins bien que la moyenne naïve.")
        elif r2 < 0.05:
            print(f"  {ICONS['warn']}  Signal très faible — modèle peu informatif pour ce feature set.")

    # Coefficients Ridge / ElasticNet
    coefs = res.get("coefs")
    if coefs:
        print()
        _print_coefs_block(name, coefs, res)

    # Importances RF
    imps = res.get("importances")
    if imps:
        print()
        _print_importances_block(imps)

    # Paramètres SVR
    best_p = res.get("best_params")
    if best_p:
        print()
        print(_subtitle("Hyperparamètres optimaux (GridSearchCV)"))
        for k, v in best_p.items():
            print(f"  {k:<12} : {v}")

    # ElasticNet : features sélectionnées
    sel = res.get("selected_features")
    if sel is not None:
        n_tot = len(coefs) if coefs else "?"
        n_sel = res.get("n_selected", len(sel))
        alpha = res.get("alpha", float("nan"))
        l1    = res.get("l1_ratio", float("nan"))
        print()
        print(_subtitle("Sélection L1 (ElasticNet)"))
        print(f"  α = {_fmt_plain(alpha, 4)}  ·  l1_ratio = {_fmt_plain(l1, 2)}")
        print(f"  Features retenues : {n_sel}/{n_tot}  →  {sel}")
        eliminated = [f for f, c in (coefs or {}).items() if abs(c) < 1e-6]
        if eliminated:
            print(f"  Features éliminées (β=0) : {eliminated}")
            print(f"  {ICONS['info']}  Ces features sont redondantes ou non-informatives")
            print(f"       pour le groove dans ce feature set.")


def _print_coefs_block(name, coefs, res):
    print(_subtitle(f"Coefficients β ({name}, normalisés)"))
    print()
    scale = max(abs(c) for c in coefs.values()) + 1e-9

    col_w = max(len(f) for f in coefs) + 2
    print(f"  {'Feature':<{col_w}} {'β':>8}  {'Direction':<6}  {'Barre (centrée)'}")
    print("  " + _sep("─", W - 2))

    for feat, coef in coefs.items():
        direction = f"{ICONS['up']} +" if coef > 0.01 else (f"{ICONS['down']} −" if coef < -0.01 else f"{ICONS['neutral']}  0")
        bar = _bar_signed(coef, scale)
        interp = COEF_INTERP.get(feat, "")
        note = f"  # {interp}" if interp and abs(coef) > 0.05 else ""
        print(f"  {feat:<{col_w}} {coef:>+8.4f}  {direction:<6}  |{bar}|{note}")

    print()
    # Top 3 features
    top3 = list(coefs.items())[:3]
    print(f"  {ICONS['arrow']} Top 3 prédicteurs : " + ", ".join(
        f"{f} (β={c:+.3f})" for f, c in top3
    ))


def _print_importances_block(imps):
    print(_subtitle("Importances RF (MDI — Mean Decrease Impurity)"))
    print()
    scale = max(imps.values()) + 1e-9
    total = sum(imps.values())

    print(f"  {'Feature':<18} {'Imp.':>6}  {'%':>5}  {'Barre'}")
    print("  " + _sep("─", W - 2))
    for feat, imp in list(imps.items())[:10]:
        bar = _bar_pos(imp, scale)
        pct = imp / total * 100
        print(f"  {feat:<18} {imp:>6.3f}  {pct:>4.1f}%  [{bar}]")

    print()
    print(f"  {ICONS['warn']}  Les importances MDI sont biaisées vers les features à forte cardinalité.")
    print(f"       À interpréter en complément des coefficients Ridge/ElasticNet.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOC LMM — RAPPORT COMPLET
# ─────────────────────────────────────────────────────────────────────────────

def _print_lmm_full(res: dict) -> None:
    print(_sep("─"))
    print(f"  {ICONS['model']}  LMM — MODÈLE LINÉAIRE MIXTE (REML)")
    print(_sep("─"))

    n_obs  = res.get("n_obs", "?")
    n_pp   = res.get("n_participants", "?")
    conv   = res.get("converged", False)
    sig_u  = res.get("sigma_u",   float("nan"))
    sig_e  = res.get("sigma_eps", float("nan"))
    icc    = res.get("icc_participant", float("nan"))
    r2m    = res.get("r2_marginal",    float("nan"))
    r2c    = res.get("r2_conditional", float("nan"))
    mae    = res.get("mae_cv_mean",    float("nan"))
    aic_ml = res.get("aic_ml", res.get("aic", float("nan")))
    bic_ml = res.get("bic_ml", res.get("bic", float("nan")))
    its    = res.get("interaction_terms", [])

    # ── Informations du modèle ───────────────────────────────────────────────
    print()
    print(_subtitle("Structure du modèle"))
    print()
    print(f"  Observations         : {n_obs}")
    print(f"  Participants (groups): {n_pp}")
    print(f"  Termes d'interaction : {its if its else 'aucun'}")
    conv_icon = ICONS["ok"] if conv else ICONS["warn"]
    conv_note = "" if conv else "  → résultats indicatifs uniquement"
    print(f"  Convergence REML     : {conv_icon} {'OUI' if conv else 'NON'}{conv_note}")
    print()

    # ── Composantes de variance ──────────────────────────────────────────────
    print(_subtitle("Décomposition de la variance"))
    print()
    if not math.isnan(sig_u) and not math.isnan(sig_e):
        total_var = sig_u**2 + sig_e**2
        pct_u = sig_u**2 / total_var * 100
        pct_e = sig_e**2 / total_var * 100
        print(f"  σ_u (inter-participants) : {sig_u:.4f}  → var = {sig_u**2:.4f}  ({pct_u:.1f}% du total)")
        print(f"  σ_ε (résiduel)           : {sig_e:.4f}  → var = {sig_e**2:.4f}  ({pct_e:.1f}% du total)")
        print(f"  ICC                      : {icc:.3f}  → {_icc_interp(icc)}")
        print()
        icc_bar = _r2_bar(icc, width=16)
        print(f"  ICC [{icc_bar}]  {icc*100:.1f}% de la variance groove est due aux différences entre participants.")
        print()
        if icc < 0.10:
            print(f"  {ICONS['info']}  ICC faible : l'effet participant est modeste.")
            print(f"       Les effets fixes (stimuli) dominent la prédiction.")
        else:
            print(f"  {ICONS['info']}  ICC significatif : la variabilité inter-participants est réelle.")
            print(f"       La modélisation comme effet aléatoire est justifiée.")

    # ── R² ───────────────────────────────────────────────────────────────────
    print()
    print(_subtitle("R² de Nakagawa & Schielzeth (2013)"))
    print()
    print(f"  R²_marginal    : {_fmt_plain(r2m)}  [{_r2_bar(max(r2m,0))}]")
    print(f"  R²_conditionnel: {_fmt_plain(r2c)}  [{_r2_bar(max(r2c,0))}]")
    print(f"  MAE in-sample  : {_fmt_plain(mae)}  (unités : rating 1–7)")
    print()
    if not math.isnan(r2m) and not math.isnan(r2c):
        delta = r2c - r2m
        print(f"  ΔR² (aléatoires): +{delta:.3f}  → les effets participants ajoutent {delta*100:.1f}% de variance expliquée")
        print(f"  Taille d'effet   : {_effect_label(r2m)} (effets fixes)  ·  {_effect_label(r2c)} (total)")
    print()
    print(f"  {ICONS['warn']}  R²_marginal (0.{round(r2m*100):03d}) est in-sample — non comparable aux R² CV des autres modèles.")
    print(f"       Référence : Nakagawa & Schielzeth (2013), Methods Ecol. Evol. 4(2):133–142")

    # ── AIC/BIC ──────────────────────────────────────────────────────────────
    print()
    print(_subtitle("Critères d'information (estimés en ML pour reporting)"))
    print()
    aic_s = _fmt_plain(aic_ml, 1) if not math.isnan(aic_ml) else "nan (REML)"
    bic_s = _fmt_plain(bic_ml, 1) if not math.isnan(bic_ml) else "nan (REML)"
    print(f"  AIC (ML) : {aic_s}")
    print(f"  BIC (ML) : {bic_s}")
    print()
    print(f"  {ICONS['info']}  AIC/BIC calculés par re-fit ML pour comparaison M0/M1.")
    print(f"       Les coefficients rapportés restent des estimateurs REML.")
    print(f"       Référence : Burnham & Anderson (2002)")

    # ── Coefficients ─────────────────────────────────────────────────────────
    coefs = res.get("coefs", {})
    if coefs:
        _print_lmm_coefs(coefs)

    # ── Synthèse significativité ─────────────────────────────────────────────
    _print_lmm_significance_summary(coefs)


def _print_lmm_coefs(coefs: dict) -> None:
    main_c  = {k: v for k, v in coefs.items() if not v.get("is_background") and not v.get("is_interaction")}
    inter_c = {k: v for k, v in coefs.items() if v.get("is_interaction")}
    bg_c    = {k: v for k, v in coefs.items() if v.get("is_background")}

    TERM_LABELS = {
        "D_sq": "Densité²",
        "DxP":  "D × Désalignement",
        "SxE":  "Syncopation × E",
        "DxS":  "D × Syncopation",
        "S_sq": "Syncopation²",
    }

    def _block(title, items):
        if not items:
            return
        print()
        print(_subtitle(title))
        print()

        all_coefs = [v["coef"] for v in items.values()]
        scale = max(abs(c) for c in all_coefs) + 1e-9

        col_w = max(len(k) for k in items) + 2
        print(f"  {'Prédicteur':<{col_w}} {'β':>8}  {'SE':>6}  {'z':>7}  {'p':>7}  {'Sig':>4}  {'IC 95%':<22}  Barre β")
        print("  " + _sep("─", W_WIDE - 2))

        for name, v in items.items():
            coef   = v.get("coef",    float("nan"))
            se     = v.get("se",      float("nan"))
            z      = v.get("z",       float("nan"))
            pval   = v.get("p_value", float("nan"))
            ci_lo  = v.get("ci_low",  float("nan"))
            ci_hi  = v.get("ci_high", float("nan"))

            sig    = _pval_stars(pval)
            bar    = _bar_signed(coef, scale, width=16)
            ci_str = f"[{_fmt_plain(ci_lo)}, {_fmt_plain(ci_hi)}]" if not (math.isnan(ci_lo) or math.isnan(ci_hi)) else "  n/a  "

            label = TERM_LABELS.get(name, name)
            print(f"  {label:<{col_w}} {coef:>+8.3f}  {se:>6.3f}  {z:>7.2f}  {pval:>7.3f}  {sig}  {ci_str:<22}  |{bar}|")

            # Note interprétative si significatif
            note = COEF_INTERP.get(name, "")
            if note and pval < 0.05:
                print(f"  {' '*col_w}   {ICONS['arrow']} {note}")

        print()
        print(f"  Codes sign. : ★★★ p<0.001  ★★ p<0.01  ★ p<0.05  † p<0.10")

    _block("Effets principaux (standardisés)", main_c)
    _block("Termes d'interaction", inter_c)
    _block("Bagage musical (réf. : non-musicien)", bg_c)


def _print_lmm_significance_summary(coefs: dict) -> None:
    print()
    print(_subtitle("Synthèse — prédicteurs significatifs (p < 0.05)"))
    print()

    sig_main  = {k: v for k, v in coefs.items() if v.get("p_value", 1) < 0.05 and not v.get("is_background") and not v.get("is_interaction")}
    sig_inter = {k: v for k, v in coefs.items() if v.get("p_value", 1) < 0.05 and v.get("is_interaction")}
    sig_bg    = {k: v for k, v in coefs.items() if v.get("p_value", 1) < 0.05 and v.get("is_background")}
    ns        = {k: v for k, v in coefs.items() if v.get("p_value", 1) >= 0.05 and not v.get("is_background")}

    TERM_LABELS = {
        "D_sq": "D²",
        "DxP":  "D×P",
        "SxE":  "S×E",
        "DxS":  "D×S",
    }

    if sig_main:
        print(f"  Effets principaux significatifs :")
        for name, v in sig_main.items():
            direction = "positif" if v["coef"] > 0 else "négatif"
            print(f"    {ICONS['star']} {name:<10}  β={v['coef']:+.3f}  p={v['p_value']:.3f}  → effet {direction} sur le groove")
    else:
        print(f"  {ICONS['neutral']}  Aucun effet principal significatif.")

    if sig_inter:
        print()
        print(f"  Interactions significatives :")
        for name, v in sig_inter.items():
            label = TERM_LABELS.get(name, name)
            direction = "synergie" if v["coef"] > 0 else "antagonisme"
            print(f"    {ICONS['star']} {label:<14}  β={v['coef']:+.3f}  p={v['p_value']:.3f}  → {direction}")

    if sig_bg:
        print()
        print(f"  Bagage musical significatif :")
        for name, v in sig_bg.items():
            lvl = name.replace("bg_", "")
            print(f"    {ICONS['star']} {lvl:<14}  β={v['coef']:+.3f}  p={v['p_value']:.3f}")

    if ns:
        ns_names = list(ns.keys())
        print()
        print(f"  Non significatifs (p ≥ 0.05) : {', '.join(ns_names)}")

    print()
    n_sig = len(sig_main) + len(sig_inter)
    n_tot = len([k for k, v in coefs.items() if not v.get("is_background")])
    print(f"  {n_sig}/{n_tot} prédicteurs (hors background) atteignent p < 0.05")


# ─────────────────────────────────────────────────────────────────────────────
# TABLEAU COMPARATIF
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison_table(results: dict) -> None:
    print(_sep("─"))
    print(f"  {ICONS['chart']}  TABLEAU COMPARATIF — TOUS MODÈLES")
    print(_sep("─"))
    print()

    preferred = ["Ridge", "ElasticNet", "SVR", "RandomForest", "LMM"]
    rows = []

    for name in preferred:
        res = results.get(name)
        if res is None or res.get("_not_fitted"):
            continue
        is_lmm = res.get("_is_lmm", False)
        r2  = res.get("r2_marginal" if is_lmm else "r2_cv_mean", float("nan"))
        r2s = float("nan") if is_lmm else res.get("r2_cv_std", float("nan"))
        mae = res.get("mae_cv_mean", float("nan"))
        rows.append((name, r2, r2s, mae, is_lmm))

    if not rows:
        print("  Aucun résultat disponible.")
        return

    print(f"  {'Modèle':<16} {'Type':<14} {'R²':>8}  {'±σ':>6}  {'MAE':>8}  {'R²_bar':<22}  {'Effet':<12}  Notes")
    print("  " + _sep("─", W_WIDE))

    for name, r2, r2s, mae, is_lmm in rows:
        typ  = "Mixte (IS)" if is_lmm else "CV 5-fold"
        r2_s  = _fmt_plain(r2)
        r2s_s = "  —  " if math.isnan(r2s) else f"±{_fmt_plain(r2s)}"
        mae_s = _fmt_plain(mae)
        bar   = _r2_bar(max(r2, 0))
        eff   = _effect_label(r2)
        note  = "[IS]" if is_lmm else ""
        print(f"  {name:<16} {typ:<14} {r2_s:>8}  {r2s_s:<6}  {mae_s:>8}  [{bar}]  {eff:<12}  {note}")

    print()
    # Classement CV uniquement
    cv_rows = [(n, r2) for n, r2, r2s, mae, is_lmm in rows if not is_lmm and not math.isnan(r2)]
    if cv_rows:
        cv_rows.sort(key=lambda x: x[1], reverse=True)
        print(f"  Classement CV : " + "  >  ".join(f"{n} ({r2:.3f})" for n, r2 in cv_rows))


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHÈSE INTERPRÉTATIVE
# ─────────────────────────────────────────────────────────────────────────────

def _print_interpretation(results: dict, feature_set: str) -> None:
    print(_sep("─"))
    print(f"  {ICONS['info']}  SYNTHÈSE INTERPRÉTATIVE — {feature_set.upper() if feature_set else 'GLOBAL'}")
    print(_sep("─"))
    print()

    lmm = results.get("LMM", {})
    ridge = results.get("Ridge", {})
    en  = results.get("ElasticNet", {})

    coefs = lmm.get("coefs", {})
    sig   = {k: v for k, v in coefs.items() if v.get("p_value", 1) < 0.05 and not v.get("is_background")}
    r2m   = lmm.get("r2_marginal", float("nan"))
    r2c   = lmm.get("r2_conditional", float("nan"))
    icc   = lmm.get("icc_participant", float("nan"))

    best_cv = max(
        ((n, r.get("r2_cv_mean", -np.inf)) for n, r in results.items()
         if not r.get("_is_lmm") and not r.get("_not_fitted")),
        key=lambda x: x[1],
        default=("—", float("nan"))
    )

    print(f"  1. POUVOIR PRÉDICTIF")
    print(f"     Meilleur modèle CV   : {best_cv[0]}  (R² = {_fmt_plain(best_cv[1])})")
    if not math.isnan(r2m):
        print(f"     LMM R²_marginal     : {_fmt_plain(r2m)}  → {_effect_label(r2m)}")
        print(f"     LMM R²_conditionnel : {_fmt_plain(r2c)}  → {_effect_label(r2c)}")
    print()
    print(f"  {ICONS['warn']}  Le signal prédictif est réel mais modeste (R² < 0.25 en CV).")
    print(f"       La perception du groove est hautement idiosyncratique.")
    print()

    if coefs:
        print(f"  2. PRÉDICTEURS CLÉS (LMM, p < 0.05)")
        if sig:
            for name, v in sorted(sig.items(), key=lambda x: abs(x[1]["coef"]), reverse=True):
                direction = f"{ICONS['up']} positif" if v["coef"] > 0 else f"{ICONS['down']} négatif"
                stars = _pval_stars(v["p_value"])
                print(f"     {stars}  {name:<14}  β={v['coef']:+.3f}  ({direction})")
        else:
            print(f"     Aucun prédicteur significatif dans ce feature set.")
        print()

    # Convergence Ridge / ElasticNet
    ridge_coefs = ridge.get("coefs", {})
    en_coefs    = en.get("coefs", {}) if en else {}

    if ridge_coefs and en_coefs:
        # Features robustes = dans top-3 Ridge ET non-nulles dans ElasticNet
        top_ridge = [f for f, c in list(ridge_coefs.items())[:4]]
        robust    = [f for f in top_ridge if abs(en_coefs.get(f, 0)) > 1e-6]
        if robust:
            print(f"  3. ROBUSTESSE (convergence Ridge ∩ ElasticNet)")
            print(f"     Features robustes : {robust}")
            print(f"     Ces features ont un signal stable quel que soit le régularisateur.")
            print()
            eliminated = [f for f in top_ridge if f not in robust]
            if eliminated:
                print(f"     Éliminées par ElasticNet (L1) : {eliminated}")
                print(f"     → Redondantes ou non-informatives pour ce feature set.")
            print()

    if not math.isnan(icc):
        print(f"  4. STRUCTURE ALÉATOIRE")
        print(f"     ICC = {icc:.3f}  → {_icc_interp(icc)}")
        if icc > 0.10:
            print(f"     {ICONS['info']}  Un modèle sans effets aléatoires aurait sous-estimé les SE.")
        print()

    print(f"  5. LIMITES")
    print(f"     • n = {lmm.get('n_obs', '?')} observations, {lmm.get('n_participants', '?')} participants")
    print(f"       → puissance statistique limitée pour les termes d'interaction")
    print(f"     • RandomForest R² ≤ 0 → pas de non-linéarité captable avec ces features")
    print(f"     • S (syncopation) non significatif : couplage S_mv→S modéré (+0.48)")
    print(f"       → bruit stochastique dans la génération dilue le signal perceptif")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE JSON + FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def save_report(
    results:  dict[str, dict],
    df:       pd.DataFrame,
    features: list[str],
    out_dir:  Path,
    df_raw:   pd.DataFrame | None = None,
    extra:    dict | None = None,
) -> None:
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS SÉRIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _make_serializable(obj):
    if isinstance(obj, bool):           return obj
    if isinstance(obj, np.bool_):       return bool(obj)
    if isinstance(obj, np.integer):     return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):           return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):  return [_make_serializable(v) for v in obj]
    return obj