"""
perception_space/viz/cluster_groove.py
=======================================
Figure publication-ready : groove moyen par cluster + distribution.
Intégration avec la charte graphique globale.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.decomposition import PCA

# On importe ta charte graphique et couleurs
from .style import apply_thesis_style

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
    """
    # Application du style global
    apply_thesis_style()
    
    unique = np.unique(clusters)
    n_c = len(unique)
    # Palette qualitative moderne
    cmap = plt.get_cmap("tab10", n_c)

    # ── Diagnostic ──────────────────────────────────────────
    print(f"\n[cluster_groove] n_stimuli={len(groove)} | n_clusters={n_c}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.3, left=0.08, right=0.98, top=0.85, bottom=0.15)

    # ── Panneau A : Barres ───────────────────────────────
    ax_bars = axes[0]
    
    means, stds, ns, cis = [], [], [], []
    for c in unique:
        vals = groove[clusters == c]
        n = len(vals)
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if n > 1 else 0.0
        # CI à 95%
        ci = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
        
        means.append(m)
        stds.append(s)
        ns.append(n)
        cis.append(ci)

    x = np.arange(n_c)
    ax_bars.bar(
        x, means,
        color=[cmap(i) for i in range(n_c)],
        alpha=0.8, width=0.6, zorder=3, edgecolor="white", linewidth=0.5
    )
    ax_bars.errorbar(
        x, means, yerr=cis, fmt="none", color="#333333",
        capsize=4, linewidth=1.2, zorder=4
    )

    # Annotations
    for xi, m, n in zip(x, means, ns):
        ax_bars.text(xi, m + (max(cis) if max(cis) > 0 else 0.5) + 0.1, f"{m:.2f}",
                     ha="center", va="bottom", fontsize=8, weight="bold")
        ax_bars.text(xi, -0.4, f"n={n}", ha="center", va="top", fontsize=7, color="#555555")

    # Labels
    x_labels = [cluster_labels.get(int(c), f"Cluster {c}") for c in unique] if cluster_labels else [f"C{c}" for c in unique]
    
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=8)
    ax_bars.set_ylabel("Groove moyen (rating 1–7)")
    ax_bars.set_ylim(0, 7.5)
    ax_bars.axhline(4, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_bars.set_title("A. Groove moyen par cluster", loc="left")
    ax_bars.grid(alpha=0.2, linestyle=":", axis="y")

    # ── Panneau B : Scatter ────────────────────────────────
    ax_scatter = axes[1]
    emb = _reduce_2d(embedding)

    for i, c in enumerate(unique):
        mask = clusters == c
        color = cmap(i)
        lbl = cluster_labels.get(int(c), f"Cluster {c}") if cluster_labels else f"Cluster {c}"

        ax_scatter.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[color], s=40, alpha=0.6,
            edgecolors="white", linewidths=0.3, label=lbl
        )

        # Centroïde
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        ax_scatter.scatter(cx, cy, marker="X", s=100, color=color,
                           edgecolors="white", linewidths=1, zorder=5)

    ax_scatter.set_xlabel("Dim 1")
    ax_scatter.set_ylabel("Dim 2")
    ax_scatter.set_title("B. Répartition dans l'espace latent", loc="left")
    ax_scatter.legend(loc="best", fontsize=7, framealpha=0.9, title="Clusters")
    ax_scatter.grid(alpha=0.15, linestyle=":")

    fig.suptitle("Analyse du groove par cluster rythmique", fontsize=12, weight="bold", y=0.98)

    # ── Sauvegarde ───────────────────────────────────────
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig) # Important pour éviter les fuites mémoire
        print(f"  [fig] Sauvegardée : {out_path.name}")

    return fig


def _reduce_2d(embedding: np.ndarray) -> np.ndarray:
    if embedding.shape[1] <= 2:
        return embedding
    return PCA(n_components=2, random_state=42).fit_transform(embedding)