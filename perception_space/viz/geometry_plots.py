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

from .style import apply_thesis_style

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
        plt.close(fig)

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

    agreement_key   = "local_agreement" if "local_agreement" in geometry else "local_coherence"
    agreement_label = "Accord local"

    metrics = [
        ("local_mean",  f"Moyenne locale {title_prefix}", "RdYlGn"),
        ("local_std",   "Variabilité locale (écart-type)", "YlOrRd"),
        ("local_slope", "Gradient local (pente)",          "RdBu_r"),
        (agreement_key, agreement_label,                   "PiYG"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.08, right=0.97, top=0.9, bottom=0.08)

    for ax, (key, label, cmap), lbl in zip(axes.flat, metrics, ["A", "B", "C", "D"]):
        if key not in geometry:
            ax.set_visible(False)
            continue

        values = np.asarray(geometry[key], dtype=float)

        norm = None
        vmin_sc = None
        vmax_sc = None

        if key in {"local_slope"}:
            # Divergent centré sur 0
            vmax_abs = float(np.nanpercentile(np.abs(values), 98))
            vmax_abs = max(vmax_abs, 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)

        elif key == agreement_key:
            # Calibrer sur les données réelles, pas sur [0, 1] entier
            vmin_data = float(np.nanpercentile(values, 2))
            vmax_data = float(np.nanpercentile(values, 98))
            # Garder une marge de 5 % de chaque côté
            margin = (vmax_data - vmin_data) * 0.05
            vmin_sc = max(0.0, vmin_data - margin)
            vmax_sc = min(1.0, vmax_data + margin)
            # Pas de TwoSlopeNorm ici — la palette PiYG est utilisée
            # mais on fixe vmin/vmax sur la plage réelle
            norm = None

        else:
            # Percentile 2–98 pour éviter les outliers
            vmin_sc, vmax_sc = np.nanpercentile(values, [2, 98])

        sc = ax.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=values, cmap=cmap, norm=norm,
            vmin=vmin_sc if norm is None else None,
            vmax=vmax_sc if norm is None else None,
            s=40, alpha=0.8, edgecolors="white", linewidths=0.2
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(f"{lbl}. {label}", loc="left", fontsize=10)
        ax.set_xlabel("Dim 1 (UMAP)", fontsize=9)
        ax.set_ylabel("Dim 2 (UMAP)", fontsize=9)
        ax.tick_params(labelbottom=False, labelleft=False)
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
    obs       = float(perm_result["observed_r"])
    sig       = perm_result.get("significant", perm_result["p_value"] < 0.05)
    p_val     = float(perm_result["p_value"])

    ax.hist(null_dist, bins=40, color=_GRAY, alpha=0.5, edgecolor="white",
            label="Distribution nulle (1 000 permutations)")

    ax.axvline(obs, color=_RED if sig else _ORANGE, lw=2.5,
               label=f"$r$ observé = {obs:.2f}  ($p = {p_val:.3f}$)")

    thresh = float(np.nanpercentile(null_dist, 95))
    ax.axvline(thresh, color=_BLUE, lw=1.5, ls="--",
               label=f"Seuil $p_{{95}}$ = {thresh:.2f}")

    ax.set_xlabel("Corrélation de Mantel ($r$)", fontsize=10)
    ax.set_ylabel("Nombre de permutations", fontsize=10)
    ax.set_title("Test de permutation (Mantel)", loc="left", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)

    _save_figure(fig, out_path)
    return fig