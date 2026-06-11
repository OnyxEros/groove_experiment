"""
perception_space/viz/geometry_plots.py
======================================
Figures publication-ready pour la géométrie locale de l'espace perceptif.
Intégration avec la charte graphique globale.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

# Import du style centralisé
from .style import apply_thesis_style

# Couleurs sémantiques maintenues pour la clarté des plots géométriques
_BLUE   = "#4157ff"
_GREEN  = "#00c896"
_ORANGE = "#ff7043"
_RED    = "#ef4444"
_GRAY   = "#888888"

# =========================================================
# HELPERS
# =========================================================

def _save_figure(fig: plt.Figure, out_path: Path | None) -> None:
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  [fig] Sauvegardée : {out_path.name}")
        plt.close(fig) # Libération mémoire immédiate

def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# =========================================================
# FIGURE 1 — Géométrie locale
# =========================================================

def plot_local_geometry(
    geometry: dict,
    embedding_2d: np.ndarray,
    title_prefix: str = "Groove",
    out_path: Path | None = None,
) -> plt.Figure:
    apply_thesis_style()
    
    agreement_key = "local_agreement" if "local_agreement" in geometry else "local_coherence"
    agreement_label = "Accord local" if agreement_key == "local_agreement" else "Cohérence locale"

    metrics = [
        ("local_mean",    f"Moyenne locale {title_prefix}",  "RdYlGn"),
        ("local_std",     "Variabilité locale (std)",        "YlOrRd"),
        ("local_slope",   "Gradient local (slope)",          "RdBu_r"),
        (agreement_key,   agreement_label,                   "PiYG"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.08, right=0.97, top=0.9, bottom=0.08)

    for ax, (key, label, cmap), lbl in zip(axes.flat, metrics, ["A", "B", "C", "D"]):
        if key not in geometry:
            ax.set_visible(False)
            continue

        values = np.asarray(geometry[key], dtype=float)
        vmin, vmax = np.nanpercentile(values, [2, 98])

        norm = None
        if key in {"local_slope", "local_coherence"}:
            vmax_abs = max(abs(vmin), abs(vmax), 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
        elif key == "local_agreement":
            norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

        sc = ax.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=values, cmap=cmap, norm=norm,
            vmin=None if norm else vmin,
            vmax=None if norm else vmax,
            s=40, alpha=0.8, edgecolors="white", linewidths=0.2
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{lbl}. {label}", loc="left")
        ax.grid(alpha=0.2, linestyle=":")

    fig.suptitle(f"Géométrie locale — {title_prefix}", fontsize=12, weight="bold", y=0.98)
    _save_figure(fig, out_path)
    return fig

# =========================================================
# FIGURE 2 — Test de permutation
# =========================================================

def plot_permutation_test(perm_result: dict, out_path: Path | None = None) -> plt.Figure:
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    null_dist = np.asarray(perm_result["permutation_dist"], dtype=float)
    obs = float(perm_result["observed_r"])
    sig = perm_result.get("significant", perm_result["p_value"] < 0.05)

    ax.hist(null_dist, bins=40, color=_GRAY, alpha=0.5, edgecolor="white")
    ax.axvline(obs, color=_RED if sig else _ORANGE, lw=2.5, label=f"Observed r = {obs:.2f}")
    
    thresh = float(np.nanpercentile(null_dist, 95))
    ax.axvline(thresh, color=_BLUE, lw=1.5, ls="--", label=f"Seuil 95% = {thresh:.2f}")

    ax.set_title("Test de permutation (Mantel)", loc="left")
    ax.legend()
    
    _save_figure(fig, out_path)
    return fig
