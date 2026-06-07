"""
regression/viz/figures.py  (v4 — VERSION MÉMOIRE)
===================================================

Figures produites :
    fig3  — Heatmap D_mv × P_mv (valeurs originales du design)
    fig8  — Comparaison R² et MAE (Ridge, ElasticNet, LMM)
    fig9  — Sélection ElasticNet vs Ridge (coefficients β)
    fig10 — Forest plot LMM avec intervalles de confiance à 95%
    fig11 — Scatter groove observé vs prédit (Ridge, ElasticNet, OOF)

Supprimées en v4 :
    fig12 — Effets marginaux D×P et D×S (nécessitait feature_set=interactions)
    fig13 — Résidus LMM (diagnostic utile mais non essentiel pour le mémoire)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
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
    "bg_amateur":  "Musicien amateur",
    "bg_semi_pro": "Musicien semi-pro",
    "bg_pro":      "Musicien pro",
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
    fig.savefig(path, facecolor=BG, bbox_inches="tight", dpi=200)
    plt.close(fig)
    if verbose:
        print(f"  [fig] {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Heatmap D_mv × P_mv
# ─────────────────────────────────────────────────────────────────────────────

def fig3_interaction_heatmap(df, out_dir, verbose=False):
    """
    Grille D_mv × P_mv colorée par groove moyen observé.
    Utilise les valeurs discrètes originales du design expérimental.
    """
    if "D_mv" not in df.columns or "P_mv" not in df.columns or "groove_mean" not in df.columns:
        if verbose:
            print("  [fig3] colonnes D_mv/P_mv/groove_mean manquantes — ignorée")
        return

    d_unique = sorted(df["D_mv"].dropna().unique())
    p_unique = sorted(df["P_mv"].dropna().unique())

    is_discrete = all(abs(v - round(v)) < 0.01 for v in d_unique + p_unique)
    if not is_discrete:
        if verbose:
            print("  [fig3] D_mv/P_mv normalisés — figure ignorée (utiliser données brutes)")
        return

    d_vals = [int(round(v)) for v in d_unique]
    p_vals = [int(round(v)) for v in p_unique]

    df2 = df.copy()
    df2["_D"] = df2["D_mv"].round().astype(int)
    df2["_P"] = df2["P_mv"].round().astype(int)

    means  = df2.groupby(["_D", "_P"])["groove_mean"].mean()
    counts = df2.groupby(["_D", "_P"])["groove_mean"].count()

    g_min, g_max = means.min(), means.max()
    g_range = g_max - g_min + 1e-9

    nd, np_ = len(d_vals), len(p_vals)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.88, bottom=0.18)
    _clean(ax)

    cmap = matplotlib.colormaps["RdYlGn"]

    p_lbl = {-1: "Laidback\n(P_mv = −1)", 0: "Grid\n(P_mv = 0)", 1: "Push\n(P_mv = +1)"}
    d_lbl = {0: "Faible\n(D_mv = 0)", 1: "Moyenne\n(D_mv = 1)", 2: "Forte\n(D_mv = 2)"}

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
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=tc, zorder=3)
            ax.text(j, i - 0.26, txt_n,
                    ha="center", va="center", fontsize=9.5,
                    color=(tc if bright else MUTED), zorder=3)

    ax.set_xticks(range(np_))
    ax.set_xticklabels([p_lbl.get(pv, str(pv)) for pv in p_vals], fontsize=10.5)
    ax.set_yticks(range(nd))
    ax.set_yticklabels([d_lbl.get(dv, str(dv)) for dv in d_vals], fontsize=10.5)
    ax.tick_params(length=0)
    ax.set_xlabel("Désalignement inter-voix (P_mv)", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_ylabel("Densité générative (D_mv)", fontweight="bold", fontsize=11, labelpad=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=g_min, vmax=g_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("Groove moyen observé (1–7)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Interaction Densité × Désalignement — Groove observé",
                 pad=12, fontsize=13, fontweight="bold")
    fig.text(0.10, 0.01,
             "Valeurs discrètes originales du design expérimental (D_mv × P_mv).\n"
             "Vert = groove élevé, rouge = faible.",
             fontsize=8.5, color=MUTED)

    _save(fig, out_dir / "fig3_interaction_heatmap.png", verbose)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Comparaison R² et MAE
# ─────────────────────────────────────────────────────────────────────────────

def fig8_model_comparison(results, out_dir, verbose=False):
    name_map = {
        "Ridge":      "Ridge\n(linéaire)",
        "ElasticNet": "ElasticNet\n(L1+L2)",
        "LMM":        "LMM\n(mixte, in-sample)",
    }
    model_order = ["Ridge", "ElasticNet", "LMM"]
    colors      = [BLUE, TEAL, PURPLE]

    names, r2s, maes, is_insample = [], [], [], []
    for nm in model_order:
        res = results.get(nm)
        if res is None:
            continue
        names.append(name_map[nm])
        if res.get("_is_lmm"):
            r2s.append(res.get("r2_marginal", 0))
            maes.append(res.get("mae_in_sample", 0))
            is_insample.append(True)
        else:
            r2s.append(res.get("r2_cv_mean", 0))
            maes.append(res.get("mae_cv_mean", 0))
            is_insample.append(False)

    if not names:
        return

    x = np.arange(len(names))
    w = 0.30

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)
    _clean(ax1)
    ax2 = ax1.twinx()

    bar_colors = [colors[model_order.index(nm)]
                  for nm in model_order if results.get(nm) is not None]

    ax1.bar(x - w/2, r2s, w, color=bar_colors, alpha=0.85,
            edgecolor="white", linewidth=0.8)
    ax2.bar(x + w/2, maes, w, color=bar_colors, alpha=0.30,
            edgecolor="white", linewidth=0.8, hatch="///")

    for bar, val, ins in zip(ax1.patches, r2s, is_insample):
        ypos = val + (0.006 if val >= 0 else -0.022)
        ax1.text(bar.get_x() + bar.get_width()/2, ypos,
                 f"{val:.3f}" + ("\n[IS]" if ins else ""),
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=DARK)

    for bar, val in zip(ax2.patches, maes):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + 0.006, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=8.5, color=MUTED)

    ax1.axhline(0, color=MUTED, lw=0.9, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("R²  (variance expliquée)", fontsize=11, color=DARK)
    ax2.set_ylabel("MAE  (erreur absolue, unités rating)", fontsize=11, color=MUTED)

    y_min = min(r2s) - 0.12
    ax1.set_ylim(y_min, max(r2s) * 1.7 + 0.05)
    ax2.set_ylim(0, max(maes) * 2.6)
    ax1.grid(axis="y", alpha=0.2, linestyle=":", zorder=0)

    leg1 = mpatches.Patch(color=MUTED, alpha=0.85, label="R² (axe gauche, barres pleines)")
    leg2 = mpatches.Patch(color=MUTED, alpha=0.30, hatch="///", label="MAE (axe droit, barres hachurées)")
    leg3 = mpatches.Patch(color=PURPLE, alpha=0.15, label="[IS] = in-sample (LMM)")
    ax1.legend(handles=[leg1, leg2, leg3], loc="upper left", fontsize=8.5)

    ax1.set_title("Comparaison des modèles — R² et MAE", pad=12, fontsize=13, fontweight="bold")
    fig.text(0.09, -0.05,
             "R² : CV 5-fold pour Ridge et ElasticNet · R²_marginal in-sample pour LMM.\n"
             "MAE en unités de rating (échelle 1–7).\n"
             "[IS] = in-sample, non comparable aux R² cross-validés.",
             fontsize=8.5, color=MUTED)

    _save(fig, out_dir / "fig8_model_comparison.png", verbose)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Sélection ElasticNet vs Ridge
# ─────────────────────────────────────────────────────────────────────────────

def fig9_elasticnet_selection(results, features, out_dir, verbose=False):
    ridge_res = results.get("Ridge", {})
    en_res    = results.get("ElasticNet", {})
    if not ridge_res or not en_res:
        return

    ridge_coefs = ridge_res.get("coefs", {})
    en_coefs    = en_res.get("coefs", {})
    if not ridge_coefs or not en_coefs:
        return

    order      = sorted(features, key=lambda f: abs(ridge_coefs.get(f, 0)), reverse=True)
    ridge_vals = np.array([ridge_coefs.get(f, 0) for f in order])
    en_vals    = np.array([en_coefs.get(f, 0)    for f in order])
    en_zero    = np.abs(en_vals) < 1e-6

    n, h = len(order), 0.30

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.55 + 2)))
    fig.patch.set_facecolor(BG)
    _clean(ax)

    for i, feat in enumerate(order):
        rv, ev = ridge_vals[i], en_vals[i]
        ax.barh(i + h/2, rv, height=h * 0.85,
                color=BLUE if rv >= 0 else RED_ACCENT, alpha=0.80, zorder=3)
        if en_zero[i]:
            ax.barh(i - h/2, 0.005, height=h * 0.85, color="#94A3B8", alpha=0.4, zorder=3)
            ax.text(0.012, i - h/2, "éliminée (β=0)",
                    va="center", fontsize=8, color="#94A3B8", style="italic")
        else:
            ax.barh(i - h/2, ev, height=h * 0.85,
                    color=TEAL if ev >= 0 else ORANGE_WARN, alpha=0.80, zorder=3)

    ax.axvline(0, color=MUTED, lw=1.0, linestyle="--", alpha=0.6)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([_fl(f) for f in order], fontsize=10)
    ax.set_xlabel("Coefficient β (normalisé)", fontsize=11)
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x", alpha=0.2, linestyle=":", zorder=0)

    n_sel    = int(en_res.get("n_selected", int(np.sum(~en_zero))))
    alpha_en = en_res.get("alpha", float("nan"))
    l1_ratio = en_res.get("l1_ratio", float("nan"))
    ax.set_title(
        f"Sélection de features — Ridge vs ElasticNet\n"
        f"ElasticNet : α={alpha_en:.4f}, l1_ratio={l1_ratio:.2f}, {n_sel}/{n} features retenues",
        pad=12, fontsize=12, fontweight="bold")

    ax.legend(handles=[
        mpatches.Patch(color=BLUE,        alpha=0.80, label="Ridge β > 0"),
        mpatches.Patch(color=RED_ACCENT,  alpha=0.80, label="Ridge β < 0"),
        mpatches.Patch(color=TEAL,        alpha=0.80, label="ElasticNet β > 0"),
        mpatches.Patch(color=ORANGE_WARN, alpha=0.80, label="ElasticNet β < 0"),
        mpatches.Patch(color="#94A3B8",   alpha=0.40, label="ElasticNet β = 0 (éliminée)"),
    ], loc="lower right", fontsize=8.5)

    fig.text(0.10, -0.04,
             "La pénalité L1 d'ElasticNet élimine les features non-informatives (β=0). "
             "Les features retenues convergent avec les prédicteurs significatifs du LMM (★).",
             fontsize=9, color=MUTED)

    _save(fig, out_dir / "fig9_elasticnet_selection.png", verbose)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Forest plot LMM
# ─────────────────────────────────────────────────────────────────────────────

def fig10_forest_plot_lmm(results, out_dir, verbose=False):
    """
    Forest plot des coefficients du LMM avec IC à 95%.
    Sépare visuellement effets principaux et bagage musical.
    Les prédicteurs significatifs (p<0.05) sont mis en évidence.
    """
    lmm_res = results.get("LMM", {})
    coefs   = lmm_res.get("coefs", {})
    if not coefs:
        if verbose:
            print("  [fig10] LMM coefs absent — ignorée")
        return

    main_items = [(k, v) for k, v in coefs.items() if not v.get("is_background")]
    bg_items   = [(k, v) for k, v in coefs.items() if v.get("is_background")]

    main_items = sorted(main_items, key=lambda x: abs(x[1]["coef"]), reverse=True)
    bg_items   = sorted(bg_items,   key=lambda x: abs(x[1]["coef"]), reverse=True)

    groups = [
        ("Effets principaux", main_items, BLUE),
        ("Bagage musical",    bg_items,   TEAL),
    ]
    groups = [(lbl, items, col) for lbl, items, col in groups if items]

    total = sum(len(items) for _, items, _ in groups) + len(groups)

    fig, ax = plt.subplots(figsize=(9, max(5, total * 0.45 + 1.5)))
    fig.patch.set_facecolor(BG)
    _clean(ax)

    y = 0
    yticks, ylabels = [], []
    group_sep_y = []

    for g_label, items, base_color in groups:
        if y > 0:
            group_sep_y.append(y + 0.3)
        y += 0.8

        for name, v in items:
            coef   = v["coef"]
            ci_low = v.get("ci_low",  coef - 1.96 * v.get("se", 0))
            ci_hi  = v.get("ci_high", coef + 1.96 * v.get("se", 0))
            pval   = v.get("p_value", 1.0)
            sig    = pval < 0.05

            color  = base_color if sig else MUTED
            alpha  = 1.0 if sig else 0.55
            lw     = 2.5 if sig else 1.2

            ax.plot([ci_low, ci_hi], [y, y], color=color, lw=lw, alpha=alpha,
                    solid_capstyle="round", zorder=3)
            marker = "D" if sig else "o"
            ms     = 8 if sig else 6
            ax.plot(coef, y, marker=marker, color=color, ms=ms, alpha=alpha,
                    zorder=4, markeredgecolor="white", markeredgewidth=0.8)

            side  = ci_hi + (ci_hi - ci_low) * 0.08 + 0.02
            p_str = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
            sig_s = " ★" if sig else ""
            ax.text(side, y, f"β={coef:+.3f}  {p_str}{sig_s}",
                    va="center", fontsize=8.5, color=color if sig else MUTED)

            yticks.append(y)
            ylabels.append(_fl(name))
            y += 1

    ax.axvline(0, color=MUTED, lw=1.2, linestyle="--", alpha=0.7, zorder=1)

    for sep_y in group_sep_y:
        ax.axhline(sep_y, color=SUBTLE, lw=1.0, zorder=0)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5)
    ax.set_xlabel("Coefficient β (standardisé)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, y + 0.5)
    ax.grid(axis="x", alpha=0.2, linestyle=":", zorder=0)
    ax.invert_yaxis()

    r2m  = lmm_res.get("r2_marginal",    float("nan"))
    r2c  = lmm_res.get("r2_conditional", float("nan"))
    icc  = lmm_res.get("icc_participant", float("nan"))
    n_ob = lmm_res.get("n_obs", "?")
    n_pp = lmm_res.get("n_participants", "?")

    stats_txt = (
        f"LMM (REML)  ·  n={n_ob} obs, {n_pp} participants  ·  "
        f"R²_marg={r2m:.3f}  ·  R²_cond={r2c:.3f}  ·  ICC={icc:.3f}"
    )
    ax.set_title("Coefficients du Modèle Linéaire Mixte (IC 95%)",
                 pad=12, fontsize=13, fontweight="bold")
    fig.text(0.10, 0.01, stats_txt, fontsize=8.5, color=MUTED)

    sig_patch   = mlines.Line2D([], [], marker="D", color=BLUE,  ms=8, ls="-", lw=2.5, label="p < 0.05 (★)")
    insig_patch = mlines.Line2D([], [], marker="o", color=MUTED, ms=6, ls="-", lw=1.2, label="p ≥ 0.05", alpha=0.55)
    ax.legend(handles=[sig_patch, insig_patch], loc="lower right", fontsize=9)

    _save(fig, out_dir / "fig10_forest_plot_lmm.png", verbose)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Scatter observé vs prédit
# ─────────────────────────────────────────────────────────────────────────────

def fig11_observed_vs_predicted(results, y_true, out_dir, verbose=False):
    """
    Scatter plot groove observé vs prédit en out-of-fold (Ridge et ElasticNet).
    """
    models_to_plot = []
    for name, color in [("Ridge", BLUE), ("ElasticNet", TEAL)]:
        res = results.get(name, {})
        oof = res.get("y_pred_oof")
        if oof is None:
            continue
        models_to_plot.append((name, color, np.array(oof), res.get("r2_cv_mean", float("nan"))))

    if not models_to_plot:
        if verbose:
            print("  [fig11] y_pred_oof absent — ignorée")
        return

    n_plots = len(models_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5.5), sharey=True)
    fig.patch.set_facecolor(BG)
    if n_plots == 1:
        axes = [axes]

    y_arr = np.array(y_true)
    global_min = min(y_arr.min(), min(p.min() for _, _, p, _ in models_to_plot)) - 0.2
    global_max = max(y_arr.max(), max(p.max() for _, _, p, _ in models_to_plot)) + 0.2

    for ax, (name, color, y_pred, r2) in zip(axes, models_to_plot):
        _clean(ax)

        try:
            xy   = np.vstack([y_arr, y_pred])
            dens = gaussian_kde(xy)(xy)
            sc   = ax.scatter(y_arr, y_pred, c=dens, cmap="YlOrRd",
                              s=35, alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3)
            fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.02, label="Densité")
        except Exception:
            ax.scatter(y_arr, y_pred, c=color, s=35, alpha=0.55,
                       edgecolors="white", linewidths=0.4, zorder=3)

        ax.plot([global_min, global_max], [global_min, global_max],
                color=MUTED, lw=1.2, linestyle="--", alpha=0.6, label="Idéal (y=x)", zorder=2)

        mask = np.isfinite(y_arr) & np.isfinite(y_pred)
        if mask.sum() > 2:
            m_reg, b_reg = np.polyfit(y_arr[mask], y_pred[mask], 1)
            xs = np.linspace(global_min, global_max, 100)
            ax.plot(xs, m_reg * xs + b_reg, color=color, lw=2, alpha=0.85,
                    label="OLS fit", zorder=4)

        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_aspect("equal")
        ax.set_xlabel("Groove observé (moyen par stimulus)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Groove prédit (OOF)", fontsize=10, fontweight="bold")
        ax.set_title(f"{name}", fontsize=12, fontweight="bold", pad=8)

        r2_s = f"{r2:.3f}" if np.isfinite(r2) else "nan"
        mae  = float(np.mean(np.abs(y_arr[mask] - y_pred[mask]))) if mask.sum() > 0 else float("nan")
        ax.text(0.05, 0.92, f"R² CV = {r2_s}\nMAE = {mae:.3f}",
                transform=ax.transAxes, fontsize=9.5, color=color,
                fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=color, lw=1))

        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(alpha=0.2, linestyle=":", zorder=0)

    fig.suptitle("Groove observé vs prédit — Prédictions out-of-fold (CV 5-fold)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.10, -0.04,
             "Chaque point = un stimulus. Prédictions OOF : le modèle n'a pas vu ce stimulus pendant l'entraînement.\n"
             "L'écart à la diagonale idéale (pointillés) reflète l'erreur de généralisation.",
             fontsize=8.5, color=MUTED)

    _save(fig, out_dir / "fig11_observed_vs_predicted.png", verbose)


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR
# ─────────────────────────────────────────────────────────────────────────────

class RegressionFigure:
    """
    Génère les figures de régression pour le mémoire.

    fig3  — Heatmap D_mv × P_mv (design expérimental)
    fig8  — Comparaison R² et MAE (Ridge, ElasticNet, LMM)
    fig9  — Sélection ElasticNet vs Ridge
    fig10 — Forest plot LMM (IC 95%)
    fig11 — Scatter observé vs prédit (OOF)
    """

    @staticmethod
    def plot(results, df, features, out_dir, df_raw=None, verbose=False):
        out_dir = Path(out_dir)
        y_true  = df["groove_mean"].values if "groove_mean" in df.columns else None

        fig3_interaction_heatmap(df, out_dir, verbose=verbose)
        fig8_model_comparison(results, out_dir, verbose=verbose)
        fig9_elasticnet_selection(results, features, out_dir, verbose=verbose)
        fig10_forest_plot_lmm(results, out_dir, verbose=verbose)

        if y_true is not None:
            fig11_observed_vs_predicted(results, y_true, out_dir, verbose=verbose)