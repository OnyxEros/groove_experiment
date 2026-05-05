"""
perception_space/viz/icc_plot.py
================================
Figures pour l'ICC et la variabilité inter-participants.

Figures :
    plot_icc_summary         — gauge ICC + IC 95% + interprétation
    plot_per_stimulus_variance — heatmap variabilité par stimulus
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.titlelocation": "left",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.dpi":         150,
}

_BLUE   = "#4157ff"
_GREEN  = "#00c896"
_ORANGE = "#ff9800"
_RED    = "#ef4444"


# =========================================================
# FIGURE 1 — ICC summary (gauge + CI)
# =========================================================

def plot_icc_summary(
    icc_groove: dict,
    icc_complexity: dict | None = None,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Visualisation de l'ICC sous forme de gauge horizontale.
    Optionnellement compare groove et complexity côte à côte.
    """
    plt.rcParams.update(_RC)

    n_panels = 2 if icc_complexity is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(
        wspace=0.4, left=0.08, right=0.97, top=0.88, bottom=0.12
    )

    datasets = [("Groove", icc_groove, _BLUE)]
    if icc_complexity is not None:
        datasets.append(("Complexity", icc_complexity, _GREEN))

    for ax, (label, result, color) in zip(axes, datasets):
        _draw_icc_gauge(ax, result, label, color)

    fig.suptitle(
        "Fiabilité inter-participants — ICC(2,1)",
        fontsize=11, weight="bold", y=0.97
    )

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig


def _draw_icc_gauge(ax, result: dict, label: str, color: str):
    """Dessine la gauge ICC pour un panneau."""
    icc  = result["icc"]
    low  = result["ci95_low"]
    high = result["ci95_high"]
    interp = result["interpretation"]

    # ── Fond de la gauge ────────────────────────────────
    # Zones colorées [0, 0.5, 0.75, 0.90, 1.0]
    zones = [
        (0.00, 0.50, "#ffebee", "Faible"),
        (0.50, 0.75, "#fff8e1", "Modérée"),
        (0.75, 0.90, "#e8f5e9", "Bonne"),
        (0.90, 1.00, "#e3f2fd", "Excellente"),
    ]

    for x0, x1, fc, _ in zones:
        ax.barh(0, x1 - x0, left=x0, height=0.35,
                color=fc, edgecolor="#dddddd", linewidth=0.5, zorder=1)

    # Frontières entre zones
    for xb in [0.50, 0.75, 0.90]:
        ax.axvline(xb, color="#cccccc", linewidth=0.8, zorder=2)

    # ── Barre ICC ───────────────────────────────────────
    ax.barh(0, max(icc, 0), height=0.2,
            color=color, alpha=0.85, zorder=3)

    # ── IC 95% ──────────────────────────────────────────
    ax.errorbar(
        icc, 0,
        xerr=[[icc - low], [high - icc]],
        fmt="o",
        color="#222222",
        capsize=6,
        linewidth=2,
        markersize=8,
        zorder=5,
    )

    # ── Labels zones ─────────────────────────────────────
    for x0, x1, _, zone_label in zones:
        ax.text((x0 + x1) / 2, 0.22, zone_label,
                ha="center", va="bottom", fontsize=7,
                color="#666666", style="italic")

    # ── Valeur centrale ──────────────────────────────────
    ax.text(
        icc, -0.22,
        f"ICC = {icc:.3f}\n[{low:.3f} – {high:.3f}]",
        ha="center", va="top",
        fontsize=9, weight="bold", color="#222222",
    )

    # ── Stats ────────────────────────────────────────────
    p = result["p_value"]
    sig = "★ p < 0.05" if p < 0.05 else f"p = {p:.3f}"
    ax.text(
        0.98, -0.33,
        f"F({result['df1']}, {result['df2']}) = {result['F']:.2f}   {sig}\n"
        f"n = {result['n_stimuli']} stimuli · {result['n_raters']} participants",
        ha="right", va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=7.5,
        color="#444444",
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("ICC(2,1)", fontsize=9)
    ax.set_title(f"{label} — Fiabilité : {interp}", pad=8)
    ax.grid(False)
    ax.spines["left"].set_visible(False)


# =========================================================
# FIGURE 2 — Variabilité par stimulus
# =========================================================

def plot_per_stimulus_variance(
    stim_variance: "pd.DataFrame",
    groove_col: str = "mean",
    std_col: str = "std",
    stim_col: str = "stimulus_id",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Barres triées par groove_mean, avec bandes d'erreur ±1 std.
    Met en évidence les stimuli ambigus (std élevée).
    """
    import pandas as pd
    plt.rcParams.update(_RC)

    df = stim_variance.sort_values(groove_col).reset_index(drop=True)
    n  = len(df)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), 5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.18)

    x      = np.arange(n)
    means  = df[groove_col].values
    stds   = df[std_col].values

    # Colormap par std (stimuli ambigus en rouge)
    norm_std = (stds - stds.min()) / (stds.max() - stds.min() + 1e-9)
    cmap     = plt.cm.get_cmap("RdYlGn_r")
    colors   = [cmap(v) for v in norm_std]

    bars = ax.bar(x, means, color=colors, alpha=0.80,
                  width=0.7, zorder=3)
    ax.errorbar(x, means, yerr=stds,
                fmt="none", color="#333333",
                capsize=3, linewidth=1.0, zorder=4)

    # Légende
    legend_patches = [
        mpatches.Patch(color=cmap(0.0), alpha=0.8, label="Faible variabilité"),
        mpatches.Patch(color=cmap(0.5), alpha=0.8, label="Variabilité modérée"),
        mpatches.Patch(color=cmap(1.0), alpha=0.8, label="Fort désaccord"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)

    # Labels stimuli (si pas trop nombreux)
    if n <= 40 and stim_col in df.columns:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(v) for v in df[stim_col].values],
            rotation=45, ha="right", fontsize=7
        )
    else:
        ax.set_xticks([])
        ax.set_xlabel("Stimuli (triés par groove moyen)", fontsize=9)

    ax.set_ylabel("Groove moyen (rating 1–7)", fontsize=9)
    ax.set_title("A  Groove moyen par stimulus ± variabilité inter-participants", pad=8)
    ax.set_ylim(0, 7.5)
    ax.axhline(4, color="#aaaaaa", linewidth=0.8, linestyle="--",
               label="Point médian (4)")
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.6, axis="y")

    fig.suptitle(
        "Profil perceptif des stimuli — Variabilité inter-participants",
        fontsize=11, weight="bold", y=0.97
    )

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig
