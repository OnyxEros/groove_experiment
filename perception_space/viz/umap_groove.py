"""
perception_space/viz/umap_groove.py
====================================
Figure publication-ready : groove et complexité dans l'espace latent.

Deux modes selon ce qui est disponible :

    Mode A (avec umap_2d depuis le run d'analyse) :
        Réutilise exactement la même projection UMAP que spaces_figure.png.
        Superpose les ratings groove sur les clusters déjà identifiés.
        → C'est la figure "superposition" du mémoire.

    Mode B (fallback PCA) :
        Calcule une réduction 2D à la volée si umap_2d.npy est absent.
        Moins cohérent visuellement mais fonctionnel.

Figure en 2 panneaux :
    A — UMAP coloré par groove_mean + contours de clusters
    B — UMAP coloré par complexité_mean (si disponible)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
from pathlib import Path
from scipy.spatial import ConvexHull

_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "axes.titleweight":   "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.6,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
}

CLUSTER_COLORS = [
    "#2563EB", "#16A34A", "#D97706", "#DC2626",
    "#7C3AED", "#0891B2", "#DB2777", "#65A30D",
]


def plot_umap_groove(
    embedding:    np.ndarray,
    groove:       np.ndarray,
    complexity:   np.ndarray | None = None,
    clusters:     np.ndarray | None = None,
    umap_2d:      np.ndarray | None = None,
    out_path:     Path | None = None,
) -> plt.Figure:
    """
    Args:
        embedding   : embeddings réalisés (n, d) — réduit à 2D si nécessaire
        groove      : ratings groove alignés (n,)
        complexity  : ratings complexité alignés (n,) — optionnel
        clusters    : labels de cluster (n,) — optionnel, pour les contours
        umap_2d     : projection 2D du run d'analyse (n_total, 2) — optionnel
                      Si fourni, les points sont placés dans le référentiel
                      de spaces_figure.png (cohérence visuelle).
        out_path    : chemin PNG
    """
    plt.rcParams.update(_RC)

    # ── Projection 2D ────────────────────────────────────
    if umap_2d is not None and umap_2d.shape[0] == len(embedding):
        emb = umap_2d
        proj_label = "UMAP (run d'analyse)"
    else:
        emb = _reduce_2d(embedding)
        proj_label = "PCA (fallback)"

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
        clusters=clusters,
        cmap="RdYlGn",
        label="A  Groove perçu dans l'espace latent",
        cbar_label="Rating groove (1–7)",
        vmin=1, vmax=7,
        proj_label=proj_label,
    )

    # ── Panneau B : Complexité ───────────────────────────
    if complexity is not None:
        _scatter_panel(
            axes[1], emb, np.asarray(complexity),
            clusters=clusters,
            cmap="RdYlBu_r",
            label="B  Complexité perçue dans l'espace latent",
            cbar_label="Rating complexité (1–7)",
            vmin=1, vmax=7,
            proj_label=proj_label,
        )

    fig.suptitle(
        "Ratings perceptifs superposés à l'espace latent",
        fontsize=11, weight="semibold", y=0.97,
    )

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig


def _scatter_panel(
    ax, emb, values, clusters, cmap,
    label, cbar_label, vmin, vmax, proj_label,
):
    # ── Contours de clusters en fond ─────────────────────
    if clusters is not None:
        unique = np.unique(clusters)
        colors = (CLUSTER_COLORS * 4)[:len(unique)]
        for i, k in enumerate(unique):
            mask = clusters == k
            pts  = emb[mask]
            if len(pts) >= 3:
                try:
                    hull  = ConvexHull(pts)
                    verts = np.append(hull.vertices, hull.vertices[0])
                    ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                            color=colors[i], alpha=0.07, zorder=1)
                    ax.plot(pts[verts, 0], pts[verts, 1],
                            color=colors[i], alpha=0.30,
                            linewidth=0.8, linestyle="--", zorder=2)
                    # Label cluster en gris sur le centroïde
                    cx, cy = pts.mean(axis=0)
                    ax.text(cx, cy, f"C{k}",
                            ha="center", va="center",
                            fontsize=7, color=colors[i],
                            fontweight="bold", alpha=0.6, zorder=3)
                except Exception:
                    pass

    # ── Points colorés par rating ────────────────────────
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=values,
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=55, alpha=0.85,
        linewidths=0.4, edgecolors="white",
        zorder=4,
    )

    # Annotations min/max
    top_idx = int(np.argmax(values))
    bot_idx = int(np.argmin(values))
    for idx, tag in [(top_idx, "max"), (bot_idx, "min")]:
        ax.annotate(
            f"{values[idx]:.1f}",
            xy=(emb[idx, 0], emb[idx, 1]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7.5, color="#111827",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7),
        )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.4)

    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.set_title(label, pad=7)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(alpha=0.15, linestyle=":", linewidth=0.5)

    # Note projection
    ax.text(0.99, 0.01, proj_label,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#9CA3AF", style="italic")


def _reduce_2d(embedding: np.ndarray) -> np.ndarray:
    if embedding.shape[1] <= 2:
        return embedding
    from sklearn.decomposition import PCA
    print("[umap_groove] projection 2D : PCA fallback (umap_2d.npy absent)")
    return PCA(n_components=2, random_state=42).fit_transform(embedding)