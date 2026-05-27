"""
perception_space/viz/icc_plot.py
================================
Figures pour l'ICC et la variabilité inter-participants.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# Import du style centralisé
from .style import apply_thesis_style

# Couleurs maintenues pour la cohérence des plots
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
    """Visualisation de l'ICC sous forme de gauge horizontale."""
    apply_thesis_style()

    n_panels = 2 if icc_complexity is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1: axes = [axes]

    fig.subplots_adjust(wspace=0.4, left=0.08, right=0.97, top=0.88, bottom=0.12)
    datasets = [("Groove", icc_groove, _BLUE)]
    if icc_complexity is not None:
        datasets.append(("Complexity", icc_complexity, _GREEN))

    for ax, (label, result, color) in zip(axes, datasets):
        _draw_icc_gauge(ax, result, label, color)

    fig.suptitle("Fiabilité inter-participants — ICC(2,1)", fontsize=12, weight="bold", y=0.98)
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  [fig] {out_path.name}")
    
    plt.close(fig)
    return fig


def _draw_icc_gauge(ax, result: dict, label: str, color: str):
    icc, low, high = result["icc"], result["ci95_low"], result["ci95_high"]
    
    # Fond de la gauge
    zones = [
        (0.00, 0.50, "#ffebee", "Faible"),
        (0.50, 0.75, "#fff8e1", "Modérée"),
        (0.75, 0.90, "#e8f5e9", "Bonne"),
        (0.90, 1.00, "#e3f2fd", "Excellente"),
    ]

    for x0, x1, fc, z_label in zones:
        ax.barh(0, x1 - x0, left=x0, height=0.35, color=fc, edgecolor="#dddddd", linewidth=0.5, zorder=1)
        ax.text((x0 + x1) / 2, 0.22, z_label, ha="center", va="bottom", fontsize=7, color="#666666", style="italic")

    # Barre ICC et erreurs
    ax.barh(0, max(icc, 0), height=0.2, color=color, alpha=0.85, zorder=3)
    ax.errorbar(icc, 0, xerr=[[icc - low], [high - icc]], fmt="o", color="#222222", 
                capsize=6, linewidth=2, markersize=8, zorder=5)

    # Textes
    ax.text(icc, -0.22, f"ICC = {icc:.3f}\n[{low:.3f} – {high:.3f}]", ha="center", va="top", fontsize=9, weight="bold")
    
    p = result["p_value"]
    sig = "★ p < 0.05" if p < 0.05 else f"p = {p:.3f}"
    ax.text(0.98, -0.33, f"F({result['df1']}, {result['df2']}) = {result['F']:.2f}  {sig}", 
            ha="right", va="top", transform=ax.get_xaxis_transform(), fontsize=7.5, color="#444444")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("ICC(2,1)", fontsize=9)
    ax.set_title(f"{label} — Fiabilité : {result['interpretation']}", pad=8)
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
    apply_thesis_style()
    df = stim_variance.sort_values(groove_col).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.35), 5))
    x = np.arange(len(df))
    
    # Colormap
    norm_std = (df[std_col] - df[std_col].min()) / (df[std_col].max() - df[std_col].min() + 1e-9)
    cmap = plt.get_cmap("RdYlGn_r")
    colors = [cmap(v) for v in norm_std]

    ax.bar(x, df[groove_col], color=colors, alpha=0.8, width=0.7, zorder=3)
    ax.errorbar(x, df[groove_col], yerr=df[std_col], fmt="none", color="#333333", capsize=3, linewidth=1.0)

    # Légende
    handles = [mpatches.Patch(color=cmap(v), label=lbl) 
               for v, lbl in [(0.0, "Faible var."), (0.5, "Modérée"), (1.0, "Fort désaccord")]]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    ax.set_ylim(0, 7.5)
    ax.set_ylabel("Groove moyen (rating 1–7)")
    ax.set_title("A. Groove moyen par stimulus ± variabilité inter-participants", pad=8)
    ax.grid(alpha=0.2, linestyle=":", axis="y")

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  [fig] {out_path.name}")
    
    plt.close(fig)
    return fig