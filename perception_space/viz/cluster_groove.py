"""
perception_space/viz/cluster_groove.py
=======================================
Figure publication-ready : groove moyen par cluster + distribution.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def plot_cluster_groove(
    embedding: np.ndarray,
    clusters: np.ndarray,
    groove: np.ndarray,
    cluster_labels: dict | None = None,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    2 panneaux :
        A — Barres groove moyen par cluster ± CI 95%
        B — Scatter espace latent coloré par cluster

    Args:
        embedding      : np.ndarray (n, d) — réduit à 2D si nécessaire
        clusters       : labels de cluster (n,)
        groove         : ratings groove (n,)
        cluster_labels : dict {cluster_id: "label sémantique"} — optionnel
        out_path       : chemin PNG
    """
    plt.rcParams.update(_RC)

    unique = np.unique(clusters)
    n_c    = len(unique)
    cmap   = plt.cm.get_cmap("tab10", n_c)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(
        wspace=0.35, left=0.08, right=0.97, top=0.88, bottom=0.14
    )

    # ── Panneau A : Barres ───────────────────────────────
    ax = axes[0]

    means, stds, ns, cis = [], [], [], []
    for c in unique:
        vals = groove[clusters == c]
        n    = len(vals)
        m    = float(vals.mean())
        s    = float(vals.std(ddof=1)) if n > 1 else 0.0
        ci   = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
        means.append(m)
        stds.append(s)
        ns.append(n)
        cis.append(ci)

    x = np.arange(n_c)
    bars = ax.bar(
        x, means,
        color=[cmap(i) for i in range(n_c)],
        alpha=0.80,
        width=0.6,
        zorder=3,
    )
    ax.errorbar(
        x, means, yerr=cis,
        fmt="none", color="#333333",
        capsize=5, linewidth=1.5, zorder=4,
    )

    # Annotations : valeur + n
    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m + max(cis) + 0.08, f"{m:.2f}",
                ha="center", va="bottom", fontsize=7.5, weight="bold")
        ax.text(xi, -0.4, f"n={n}",
                ha="center", va="top", fontsize=7, color="#666666")

    # Labels clusters
    x_labels = []
    for i, c in enumerate(unique):
        lbl = cluster_labels.get(int(c), f"C{c}") if cluster_labels else f"Cluster {c}"
        x_labels.append(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Groove moyen (rating 1–7)")
    ax.set_ylim(0, 7.5)
    ax.axhline(4, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title("A  Groove moyen par cluster ± CI 95%", pad=7)
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.6, axis="y")

    # ── Panneau B : Scatter espace latent ─────────────────
    ax = axes[1]
    emb = _reduce_2d(embedding)

    for i, c in enumerate(unique):
        mask  = clusters == c
        color = cmap(i)
        lbl   = cluster_labels.get(int(c), f"Cluster {c}") if cluster_labels else f"Cluster {c}"

        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[color], s=45, alpha=0.75,
            linewidths=0.3, edgecolors="white",
            label=lbl, zorder=3,
        )

        # Centroïde
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        ax.scatter(cx, cy, marker="X", s=120, color=color,
                   edgecolors="#222222", linewidths=1.2, zorder=5)
        ax.text(cx, cy + 0.06, f"{means[i]:.1f}",
                ha="center", va="bottom", fontsize=7.5,
                color="#222222", weight="bold")

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("B  Espace latent coloré par cluster", pad=7)
    ax.legend(loc="best", fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.15, linestyle=":", linewidth=0.6)

    fig.suptitle(
        "Groove perçu par cluster rythmique",
        fontsize=11, weight="bold", y=0.97
    )

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig


def _reduce_2d(embedding: np.ndarray) -> np.ndarray:
    if embedding.shape[1] <= 2:
        return embedding
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(embedding)