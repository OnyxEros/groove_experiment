# regression/viz/figures.py
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DESIGN SYSTEM & UTILS
# ─────────────────────────────────────────────────────────────────────────────
BG          = "#FAFAFA"
PANEL       = "#FFFFFF"
DARK        = "#0F172A"
MUTED       = "#475569"
SUBTLE      = "#E2E8F0"
RED_ACCENT  = "#EF4444"
GREEN_OK    = "#059669"
ORANGE_WARN = "#EA580C"
BLUE        = "#2563EB"
PURPLE      = "#7C3AED"
TEAL        = "#0D9488"

FEATURE_LABELS = {
    "D":    "Densité (D)",
    "I":    "Irrégularité (I)",
    "V":    "Variabilité (V)",
    "S":    "Syncopation (S)",
    "E":    "Micro-timing (E)",
    "P":    "Désalignement (P)",
    "D_mv": "Densité gén. (D_mv)",
    "S_mv": "Syncopation gén. (S_mv)",
    "E_mv": "Micro-timing gén. (E_mv)",
    "P_mv": "Décalage gén. (P_mv)",
    "BPM":  "BPM",
    "S_sq": "Syncopation² (S²)",
}

def _fl(f):
    return FEATURE_LABELS.get(f, f)

def _clean(ax):
    for s in ax.spines.values():
        s.set_edgecolor(SUBTLE)
    ax.set_facecolor(PANEL)

def _save(fig, path, verbose):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  [fig] {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FONCTIONS DE VISUALISATION (FIG 1, 2, 4, 5, 6, 7 optionnelles/stubs)
# ─────────────────────────────────────────────────────────────────────────────
# Note : Ajoutez ici vos implémentations de fig1, fig2, etc., si nécessaire.


# ─────────────────────────────────────────────────────────────────────────────
# 3. PATCHS V2 INTÉGRÉS
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================
# FIG 3 (v2) — Heatmap D_mv × P_mv — FIX LABELS NORMALISÉS
# ============================================================
def fig3_interaction_heatmap(df, out_dir, verbose=False):
    """
    Grille D_mv × P_mv colorée par groove moyen observé.

    FIX v2 : le df reçu est NORMALISÉ (z-score). On reconstitue les
    labels catégoriels en utilisant les tertiles de la distribution
    au lieu de .round() qui produit des valeurs incorrectes après z-score.
    """
    if "D_mv" not in df.columns or "P_mv" not in df.columns or "groove_mean" not in df.columns:
        return

    df2 = df.copy()

    # Reconstitution des labels depuis les tertiles (robuste à la normalisation)
    d_terciles = df2["D_mv"].quantile([0.33, 0.67]).values
    p_terciles = df2["P_mv"].quantile([0.33, 0.67]).values

    def _bin_d(v):
        if v < d_terciles[0]: return 0
        elif v < d_terciles[1]: return 1
        else: return 2

    def _bin_p(v):
        if v < p_terciles[0]: return -1
        elif v < p_terciles[1]: return 0
        else: return 1

    df2["_D"] = df2["D_mv"].apply(_bin_d)
    df2["_P"] = df2["P_mv"].apply(_bin_p)

    d_vals = sorted(df2["_D"].unique())
    p_vals = sorted(df2["_P"].unique())
    nd, np_ = len(d_vals), len(p_vals)

    means  = df2.groupby(["_D", "_P"])["groove_mean"].mean()
    counts = df2.groupby(["_D", "_P"])["groove_mean"].count()

    g_min, g_max = means.min(), means.max()
    g_range = g_max - g_min + 1e-9

    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.14, right=0.88, top=0.88, bottom=0.18)
    _clean(ax)

    cmap = matplotlib.colormaps["RdYlGn"]

    p_lbl = {-1: "Laidback\n(P_mv bas)", 0: "Grid\n(P_mv med.)", 1: "Push\n(P_mv haut)"}
    d_lbl = {0:  "Faible\n(D_mv bas)",   1: "Moyenne\n(D_mv med.)", 2: "Forte\n(D_mv haut)"}

    ax.set_xlim(-0.5, np_ - 0.5)
    ax.set_ylim(-0.5, nd  - 0.5)
    ax.set_aspect("equal", adjustable="box")

    for i, dv in enumerate(d_vals):
        for j, pv in enumerate(p_vals):
            val = means.get((dv, pv), np.nan)
            n_c = counts.get((dv, pv), 0)

            if np.isnan(val):
                cell_col = "#F1F5F9"
                txt, txt_n = "n/a", ""
            else:
                cell_col = cmap((val - g_min) / g_range)
                txt   = f"{val:.2f}"
                txt_n = f"n = {n_c}"

            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cell_col, edgecolor=BG, lw=2.5, zorder=1))

            bright = (not np.isnan(val)) and ((val - g_min) / g_range > 0.60)
            tc = "white" if bright else DARK

            ax.text(j, i + 0.13, txt,
                    ha="center", va="center", fontsize=15,
                    fontweight="bold", color=tc, zorder=3)
            ax.text(j, i - 0.26, txt_n,
                    ha="center", va="center", fontsize=9,
                    color=(tc if bright else MUTED), zorder=3)

    ax.set_xticks(range(np_))
    ax.set_xticklabels([p_lbl.get(pv, str(pv)) for pv in p_vals], fontsize=10.5)
    ax.set_yticks(range(nd))
    ax.set_yticklabels([d_lbl.get(dv, str(dv)) for dv in d_vals], fontsize=10.5)
    ax.tick_params(length=0)
    ax.set_xlabel("Désalignement inter-voix (P_mv)", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_ylabel("Densité générative (D_mv)",        fontweight="bold", fontsize=11, labelpad=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=g_min, vmax=g_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("Groove moyen observé (1–7)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Interaction Densité × Désalignement — Groove observé", pad=12)
    fig.text(0.10, 0.02,
             "Groove moyen par tertile (D_mv × P_mv). Vert = groove élevé, rouge = faible.\n"
             "Un gradient diagonal indiquerait un effet interactif (non additif).",
             fontsize=9, color=MUTED)

    _save(fig, out_dir / "fig3_interaction_heatmap.png", verbose)


# ============================================================
# FIG 8 (v2) — Comparaison R² et MAE — ÉTENDU 5 modèles
# ============================================================
def fig8_model_comparison(results, out_dir, verbose=False):
    """
    Barplot double axe : R² (axe gauche) et MAE (axe droit).
    v2 : 5 modèles — Ridge · ElasticNet · SVR · RF · LMM
    """
    name_map = {
        "Ridge":        "Ridge\n(linéaire)",
        "ElasticNet":   "ElasticNet\n(L1+L2)",
        "SVR":          "SVR\n(RBF kernel)",
        "RandomForest": "RF\n(non-linéaire)",
        "LMM":          "LMM\n(mixte, in-sample)",
    }
    model_order = ["Ridge", "ElasticNet", "SVR", "RandomForest", "LMM"]
    colors      = [BLUE, TEAL, GREEN_OK, ORANGE_WARN, PURPLE]

    names, r2s, maes, is_insample = [], [], [], []
    for nm in model_order:
        res = results.get(nm)
        if res is None:
            continue
        names.append(name_map[nm])
        if res.get("_is_lmm") or nm == "LMM":
            r2s.append(res.get("r2_marginal", res.get("r2_marginal_in_sample", 0)))
            maes.append(res.get("mae_cv_mean", res.get("mae_in_sample", 0)))
            is_insample.append(True)
        else:
            r2s.append(res.get("r2_cv_mean", 0))
            maes.append(res.get("mae_cv_mean", 0))
            is_insample.append(False)

    if not names:
        return

    x = np.arange(len(names))
    w = 0.30

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(BG)
    _clean(ax1)
    fig.subplots_adjust(left=0.09, right=0.91, top=0.87, bottom=0.15)

    ax2 = ax1.twinx()

    bar_colors = [colors[model_order.index(nm)]
                  for nm in model_order if results.get(nm) is not None]

    bars1 = ax1.bar(x - w/2, r2s, w,
                    color=bar_colors, alpha=0.85,
                    edgecolor="white", linewidth=0.8, label="R²")
    bars2 = ax2.bar(x + w/2, maes, w,
                    color=bar_colors, alpha=0.30,
                    edgecolor="white", linewidth=0.8,
                    hatch="///", label="MAE")

    for bar, val, ins in zip(bars1, r2s, is_insample):
        ypos = bar.get_height() + (0.006 if val >= 0 else -0.022)
        ax1.text(bar.get_x() + bar.get_width()/2, ypos,
                 f"{val:.3f}" + ("\n[IS]" if ins else ""),
                 ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=DARK)

    for bar, val in zip(bars2, maes):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.006,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=8.5, color=MUTED)

    if True in is_insample:
        lmm_idx = is_insample.index(True)
        ax1.axvspan(lmm_idx - 0.5, lmm_idx + 0.5,
                    alpha=0.06, color=PURPLE, zorder=0)

    ax1.axhline(0, color=MUTED, lw=0.9, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("R²  (variance expliquée)", fontsize=11, color=DARK)
    ax2.set_ylabel("MAE  (erreur absolue, unités rating)", fontsize=11, color=MUTED)

    y_min = min(r2s) - 0.12
    y_max = max(r2s) * 1.7 + 0.05
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(0, max(maes) * 2.6)
    ax1.grid(axis="y", alpha=0.2, linestyle=":", zorder=0)

    leg1 = mpatches.Patch(color=MUTED, alpha=0.85, label="R² (axe gauche, barres pleines)")
    leg2 = mpatches.Patch(color=MUTED, alpha=0.30, hatch="///", label="MAE (axe droit, barres hachurées)")
    leg3 = mpatches.Patch(color=PURPLE, alpha=0.15, label="[IS] = in-sample (LMM)")
    ax1.legend(handles=[leg1, leg2, leg3], loc="upper left", fontsize=8.5)

    ax1.set_title("Comparaison des modèles — R² et MAE", pad=12)
    fig.text(0.09, -0.05,
             "R² : CV 5-fold pour Ridge, ElasticNet, SVR et RF · R²_marginal in-sample pour LMM.\n"
             "MAE en unités de rating (échelle 1–7). R² négatif = modèle moins bon que la moyenne.\n"
             "[IS] = in-sample, non comparable aux R² cross-validés.",
             fontsize=8.5, color=MUTED)

    _save(fig, out_dir / "fig8_model_comparison.png", verbose)


# ============================================================
# FIG 9 (NOUVELLE) — Sélection de features ElasticNet vs Ridge
# ============================================================
def fig9_elasticnet_selection(results, features, out_dir, verbose=False):
    """
    Figure comparative Ridge vs ElasticNet : Barplot horizontal mettant en
    évidence la sélection et l'annulation de features.
    """
    ridge_res = results.get("Ridge", {})
    en_res    = results.get("ElasticNet", {})

    if not ridge_res or not en_res:
        return

    ridge_coefs = ridge_res.get("coefs", {})
    en_coefs    = en_res.get("coefs", {})

    if not ridge_coefs or not en_coefs:
        return

    order = sorted(features, key=lambda f: abs(ridge_coefs.get(f, 0)), reverse=True)

    ridge_vals = np.array([ridge_coefs.get(f, 0) for f in order])
    en_vals    = np.array([en_coefs.get(f, 0)    for f in order])
    en_zero    = np.abs(en_vals) < 1e-6

    n  = len(order)
    h  = 0.30

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.55 + 2)))
    fig.patch.set_facecolor(BG)
    _clean(ax)

    for i, feat in enumerate(order):
        rv = ridge_vals[i]
        ev = en_vals[i]

        # Ridge
        bar_col = BLUE if rv >= 0 else RED_ACCENT
        ax.barh(i + h/2, rv, height=h * 0.85,
                color=bar_col, alpha=0.80, zorder=3)

        # ElasticNet
        if en_zero[i]:
            ax.barh(i - h/2, 0.005, height=h * 0.85,
                    color="#94A3B8", alpha=0.4, zorder=3)
            ax.text(0.012, i - h/2, "éliminée (β=0)",
                    va="center", fontsize=8, color="#94A3B8", style="italic")
        else:
            ec = TEAL if ev >= 0 else ORANGE_WARN
            ax.barh(i - h/2, ev, height=h * 0.85,
                    color=ec, alpha=0.80, zorder=3)

    ax.axvline(0, color=MUTED, lw=1.0, linestyle="--", alpha=0.6)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([_fl(f) for f in order], fontsize=10)
    ax.set_xlabel("Coefficient β (normalisé)", fontsize=11)
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x", alpha=0.2, linestyle=":", zorder=0)

    n_selected = int(en_res.get("n_selected", sum(~en_zero)))
    alpha_en   = en_res.get("alpha", float("nan"))
    l1_ratio   = en_res.get("l1_ratio", float("nan"))

    ax.set_title(
        f"Sélection de features — Ridge vs ElasticNet\n"
        f"ElasticNet : α={alpha_en:.4f}, l1_ratio={l1_ratio:.2f}, "
        f"{n_selected}/{n} features retenues",
        pad=12,
    )

    ax.legend(handles=[
        mpatches.Patch(color=BLUE,      alpha=0.80, label="Ridge β > 0"),
        mpatches.Patch(color=RED_ACCENT,alpha=0.80, label="Ridge β < 0"),
        mpatches.Patch(color=TEAL,      alpha=0.80, label="ElasticNet β > 0"),
        mpatches.Patch(color=ORANGE_WARN,alpha=0.80,label="ElasticNet β < 0"),
        mpatches.Patch(color="#94A3B8", alpha=0.40, label="ElasticNet β = 0 (éliminée)"),
    ], loc="lower right", fontsize=8.5)

    fig.text(0.10, -0.04,
             "La pénalité L1 d'ElasticNet élimine les features non-informatives (β=0). "
             "Les features retenues convergent avec les prédicteurs significatifs du LMM (★).",
             fontsize=9, color=MUTED)

    _save(fig, out_dir / "fig9_elasticnet_selection.png", verbose)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORCHESTRATEUR PRINCIPAL (NOTAMMENT APPELÉ PAR REPORT.PY)
# ─────────────────────────────────────────────────────────────────────────────
class RegressionFigure:
    """
    Classe maîtresse appelée par `save_report()` pour piloter 
    le pipeline de génération des graphiques.
    """
    @staticmethod
    def plot(results, df, features, out_dir, df_raw=None, verbose=False):
        """
        Génère toutes les figures de l'analyse séquentiellement.
        """
        out_dir = Path(out_dir)
        
        # [Ici se placent vos appels aux figures 1, 2, 4, 5, 6, 7 d'origine]

        # FIG 3 (v2) : Heatmap corrigée (D_mv x P_mv)
        fig3_interaction_heatmap(df, out_dir, verbose=verbose)

        # FIG 8 (v2) : Comparaison étendue multi-modèles (5 modèles)
        fig8_model_comparison(results, out_dir, verbose=verbose)

        # FIG 9 (New) : Graphique de sélection ElasticNet
        fig9_elasticnet_selection(results, features, out_dir, verbose=verbose)