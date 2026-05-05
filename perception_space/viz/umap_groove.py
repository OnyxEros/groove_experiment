"""
perception_space/viz/umap_groove.py
====================================
Figure publication-ready : groove dans l'espace latent 2D.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def plot_umap_groove(
    embedding: np.ndarray,
    groove: np.ndarray,
    complexity: np.ndarray | None = None,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Scatter plot de l'espace latent 2D coloré par groove (et complexity si dispo).

    Args:
        embedding   : np.ndarray shape (n, d) — réduit à 2D si d > 2
        groove      : ratings groove (n,)
        complexity  : ratings complexity (n,) — optionnel
        out_path    : chemin PNG de sauvegarde
    """
    plt.rcParams.update(_RC)

    emb = _reduce_2d(embedding)
    groove = np.asarray(groove)

    n_panels = 2 if complexity is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(
        wspace=0.35, left=0.08, right=0.97, top=0.88, bottom=0.12
    )

    # ── Panneau A : Groove ───────────────────────────────
    _scatter_panel(
        axes[0], emb, groove,
        cmap="RdYlGn",
        label="A  Groove perçu",
        cbar_label="Rating groove (1–7)",
        vmin=1, vmax=7,
    )

    # ── Panneau B : Complexity ───────────────────────────
    if complexity is not None:
        _scatter_panel(
            axes[1], emb, np.asarray(complexity),
            cmap="RdYlBu_r",
            label="B  Complexité perçue",
            cbar_label="Rating complexity (1–7)",
            vmin=1, vmax=7,
        )

    fig.suptitle(
        "Ratings perceptifs dans l'espace latent des embeddings",
        fontsize=11, weight="bold", y=0.97
    )

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig


def _scatter_panel(ax, emb, values, cmap, label, cbar_label, vmin, vmax):
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=values,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        s=60,
        alpha=0.82,
        linewidths=0.4,
        edgecolors="white",
        zorder=3,
    )

    # Annotations : indices des extremes
    top_idx = int(np.argmax(values))
    bot_idx = int(np.argmin(values))
    for idx, tag in [(top_idx, "max"), (bot_idx, "min")]:
        ax.annotate(
            f"{values[idx]:.1f}",
            xy=(emb[idx, 0], emb[idx, 1]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7.5, color="#222222",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
        )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Dimension 1", fontsize=9)
    ax.set_ylabel("Dimension 2", fontsize=9)
    ax.set_title(label, pad=7)
    ax.grid(alpha=0.15, linestyle=":", linewidth=0.6)


def _reduce_2d(embedding: np.ndarray) -> np.ndarray:
    """Réduit à 2D via PCA si nécessaire."""
    if embedding.shape[1] <= 2:
        return embedding
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(embedding)