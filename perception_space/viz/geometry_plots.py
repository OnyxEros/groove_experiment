"""
perception_space/viz/geometry_plots.py
======================================
Figures publication-ready pour la géométrie locale de l'espace perceptif.

Figures produites :
    plot_local_geometry   — 4 panneaux : local_mean, std, slope, coherence
    plot_permutation_test — distribution nulle + valeur observée
    plot_condition_stats  — moyennes groove par condition (barres + CI)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style partagé ─────────────────────────────────────────
_RC = {
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            9,
    "axes.labelsize":       9,
    "axes.titlesize":       10,
    "axes.titleweight":     "bold",
    "axes.titlelocation":   "left",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#cccccc",
    "figure.dpi":           150,
}

_BLUE   = "#4157ff"
_GREEN  = "#00c896"
_ORANGE = "#ff7043"
_RED    = "#ef4444"
_GRAY   = "#888888"


# =========================================================
# FIGURE 1 — Géométrie locale (4 panneaux)
# =========================================================

def plot_local_geometry(
    geometry: dict,
    embedding_2d: np.ndarray,
    title_prefix: str = "Groove",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    4 panneaux scatter dans l'espace 2D des embeddings,
    colorés par local_mean / local_std / local_slope / local_coherence.

    Args:
        geometry     : dict retourné par compute_local_geometry
        embedding_2d : projection 2D (PCA ou UMAP), shape (n, 2)
        title_prefix : "Groove" ou "Complexity"
        out_path     : chemin de sauvegarde (PNG)
    """
    plt.rcParams.update(_RC)

    metrics = [
        ("local_mean",      f"Moyenne locale {title_prefix}",      "RdYlGn"),
        ("local_std",       "Variabilité locale (std)",             "YlOrRd"),
        ("local_slope",     "Gradient local (slope)",               "RdBu_r"),
        ("local_coherence", "Cohérence locale (r distance-rating)", "PiYG"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(
        hspace=0.38, wspace=0.32,
        left=0.07, right=0.96, top=0.90, bottom=0.08
    )

    labels = ["A", "B", "C", "D"]
    emb    = embedding_2d

    for ax, (key, label, cmap), lbl in zip(axes.flat, metrics, labels):
        values = geometry[key]

        # Robustesse : clip les outliers extrêmes pour la colormap
        vmin = float(np.percentile(values, 2))
        vmax = float(np.percentile(values, 98))

        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=values,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            s=45,
            alpha=0.80,
            linewidths=0.3,
            edgecolors="white",
            zorder=3,
        )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.ax.tick_params(labelsize=7)

        ax.set_xlabel("Dim 1", fontsize=8)
        ax.set_ylabel("Dim 2", fontsize=8)
        ax.set_title(f"{lbl}  {label}", pad=7)
        ax.grid(alpha=0.15, linestyle=":", linewidth=0.6)

    fig.suptitle(
        f"Géométrie locale — {title_prefix} dans l'espace latent",
        fontsize=11, weight="bold", y=0.97
    )

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig


# =========================================================
# FIGURE 2 — Test de permutation
# =========================================================

def plot_permutation_test(
    perm_result: dict,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Distribution nulle du test de permutation + valeur observée.
    """
    plt.rcParams.update(_RC)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.14)

    null_dist  = np.array(perm_result["permutation_dist"])
    observed_r = perm_result["observed_r"]
    p_value    = perm_result["p_value"]
    sig        = perm_result.get("significant", p_value < 0.05)

    # Histogramme distribution nulle
    ax.hist(
        null_dist,
        bins=40,
        color=_GRAY,
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
        label="Distribution nulle (permutations)",
    )

    # Valeur observée
    color_obs = _RED if sig else _ORANGE
    ax.axvline(
        observed_r,
        color=color_obs,
        linewidth=2.5,
        zorder=5,
        label=f"r observé = {observed_r:.3f}",
    )

    # Seuil 95%
    thresh = float(np.percentile(null_dist, 95))
    ax.axvline(
        thresh,
        color=_BLUE,
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label=f"Seuil 95% = {thresh:.3f}",
    )

    # Ombrage zone critique
    x_fill = null_dist[null_dist >= thresh]
    if len(x_fill):
        ax.hist(
            x_fill,
            bins=40,
            color=_BLUE,
            alpha=0.25,
            edgecolor="none",
        )

    # Annotation p-value
    sig_txt = "★ significatif" if sig else "non significatif"
    ax.text(
        0.97, 0.95,
        f"p = {p_value:.3f}  ({sig_txt})\nn = {perm_result['n_permutations']} permutations",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("Corrélation distance-rating (r)", fontsize=9)
    ax.set_ylabel("Fréquence", fontsize=9)
    ax.set_title(
        "Test de permutation — Structure groove dans l'espace latent",
        pad=8
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.6, axis="y")

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig


# =========================================================
# FIGURE 3 — Stats par condition
# =========================================================

def plot_condition_stats(
    condition_stats: "pd.DataFrame",
    anova_results: "pd.DataFrame",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Moyennes groove ± CI95 pour chaque variable de design.
    Panneaux séparés par condition.

    Args:
        condition_stats : retour de compute_condition_stats (1 condition à la fois)
        anova_results   : retour de kruskal_by_condition ou anova_by_condition
    """
    import pandas as pd
    plt.rcParams.update(_RC)

    conditions = [c for c in ["S_mv", "D_mv", "E", "P"]
                  if c in condition_stats.columns]

    n_panels = len(conditions)
    if n_panels == 0:
        return None

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(
        wspace=0.35, left=0.10, right=0.97, top=0.88, bottom=0.14
    )

    labels_map = {
        "S_mv": "Syncopation ($S_{mv}$)",
        "D_mv": "Densité ($D_{mv}$)",
        "E":    "Micro-timing ($E$)",
        "P":    "Push/pull ($P$)",
    }

    colors = [_BLUE, _GREEN, _ORANGE, _RED]

    for ax, cond, color in zip(axes, conditions, colors):
        grp = (
            condition_stats
            .groupby(cond)["groove_mean"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grp["ci95"] = 1.96 * grp["std"] / np.sqrt(grp["count"])

        x      = grp[cond].values
        means  = grp["mean"].values
        ci95   = grp["ci95"].values

        ax.bar(
            x, means,
            color=color, alpha=0.75,
            width=0.6 * (x[1] - x[0]) if len(x) > 1 else 0.4,
            zorder=3,
        )
        ax.errorbar(
            x, means, yerr=ci95,
            fmt="none", color="#333333",
            capsize=5, linewidth=1.5, zorder=4,
        )

        # Annotation valeurs
        for xi, mi in zip(x, means):
            ax.text(xi, mi + 0.05, f"{mi:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

        # p-value depuis les résultats ANOVA/Kruskal
        if anova_results is not None and not anova_results.empty:
            row = anova_results[anova_results["condition"] == cond]
            if not row.empty:
                p   = row.iloc[0]["p_value"]
                et2 = row.iloc[0]["eta2"]
                sig = "★" if p < 0.05 else "n.s."
                ax.text(
                    0.97, 0.97,
                    f"η² = {et2:.3f}  {sig}\np = {p:.3f}",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9),
                )

        ax.set_xlabel(labels_map.get(cond, cond), fontsize=9)
        ax.set_ylabel("Groove moyen (rating)" if ax is axes[0] else "")
        ax.set_title(f"Groove ~ {cond}", pad=7)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18, linestyle=":", linewidth=0.6, axis="y")
        ax.set_xticks(x)

    fig.suptitle(
        "Effet des paramètres de design sur le groove perçu",
        fontsize=11, weight="bold", y=0.97
    )

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig
