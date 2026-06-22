"""
perception_space/viz/cluster_groove.py
=======================================
Figure publication-ready : groove moyen par cluster + distribution UMAP.

v3 — refonte complète alignée sur les vrais clusters issus de ClusteringStep :
    A — Barres groove moyen par cluster ± CI 95%
    B — Scatter UMAP coloré par cluster + enveloppe convexe
    C — Profil acoustique radar / heatmap par cluster
        (descripteurs réalisés D, S, E, P normalisés)

Le panneau C donne la valeur analytique manquante dans v1/v2 :
il permet de caractériser acoustiquement chaque cluster et de
légitimer les labels sémantiques produits par ClusteringStep.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from pathlib import Path
from sklearn.decomposition import PCA

from perception_space.viz.style import apply_thesis_style

# ── Palette — 6 clusters max, accessible aux daltoniens ──────────────────────
CLUSTER_PALETTE = [
    "#2563EB",  # bleu
    "#16A34A",  # vert
    "#D97706",  # ambre
    "#DC2626",  # rouge
    "#7C3AED",  # violet
    "#0891B2",  # cyan
]

DESCRIPTOR_LABELS = {
    "D": "Densité",
    "S": "Syncopation",
    "E": "Expressivité",
    "P": "Polarité",
}


def plot_cluster_groove(
    embedding:      np.ndarray,
    clusters:       np.ndarray,
    groove:         np.ndarray,
    cluster_labels: dict | None = None,
    semantics:      dict | None = None,
    df_realized:    "pd.DataFrame | None" = None,
    umap_2d:        np.ndarray | None = None,
    out_path:       Path | None = None,
) -> plt.Figure:
    """
    Args:
        embedding      : embeddings réalisés (n, d)
        clusters       : labels entiers (n,)
        groove         : ratings groove agrégés (n,)
        cluster_labels : dict {int → str} labels sémantiques courts
        semantics      : dict issu de ClusteringStep._compute_cluster_semantics
        df_realized    : DataFrame avec colonnes D, S, E, P pour le panneau C
        umap_2d        : projection 2D (n, 2) — PCA fallback si None
        out_path       : chemin de sortie PNG
    """
    apply_thesis_style()

    unique   = sorted(np.unique(clusters).tolist())
    n_c      = len(unique)
    palette  = (CLUSTER_PALETTE * 3)[:n_c]

    has_profile = df_realized is not None and all(
        c in df_realized.columns for c in DESCRIPTOR_LABELS
    )

    n_panels = 3 if has_profile else 2
    fig_w    = 5.5 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.5))
    if n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97, top=0.85, bottom=0.15)

    _panel_a_bars(axes[0], unique, clusters, groove, palette, cluster_labels, semantics)
    _panel_b_umap(axes[1], embedding, clusters, groove, palette, cluster_labels, umap_2d)
    if has_profile:
        _panel_c_profile(axes[2], unique, clusters, df_realized, palette, cluster_labels)

    fig.suptitle(
        "Structure de l'espace perceptif par cluster rythmique",
        fontsize=12, weight="semibold", y=0.97,
    )

    _log_figure(unique, clusters, groove, semantics)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [fig] Sauvegardée : {Path(out_path).name}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PANNEAU A — Groove moyen par cluster
# ─────────────────────────────────────────────────────────────────────────────

def _panel_a_bars(ax, unique, clusters, groove, palette, cluster_labels, semantics):
    groove = np.asarray(groove, dtype=float)
    means, cis, ns = [], [], []

    for c in unique:
        vals = groove[clusters == c]
        vals = vals[np.isfinite(vals)]
        n    = len(vals)
        m    = float(vals.mean()) if n > 0 else 0.0
        ci   = 1.96 * float(vals.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
        means.append(m)
        cis.append(ci)
        ns.append(n)

    x = np.arange(len(unique))
    ax.bar(x, means, color=palette, alpha=0.82, width=0.6,
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.errorbar(x, means, yerr=cis, fmt="none", color="#333333",
                capsize=5, linewidth=1.3, zorder=4)

    # Annotations groove + n
    y_top = max(means) + max(cis) + 0.15 if cis else max(means) + 0.3
    for xi, (m, ci, n) in enumerate(zip(means, cis, ns)):
        ax.text(xi, m + ci + 0.08, f"{m:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=palette[xi])
        ax.text(xi, -0.5, f"n={n}", ha="center", va="top",
                fontsize=7, color="#666666")

    # Labels axe x
    x_labels = _get_labels(unique, cluster_labels, semantics, short=True)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score de groove moyen (1–7)", fontsize=9)
    ax.set_ylim(0, 7.5)
    ax.axhline(4, color="#AAAAAA", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("A. Groove moyen par cluster", loc="left", fontsize=10)
    ax.grid(alpha=0.2, linestyle=":", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# PANNEAU B — UMAP coloré par cluster
# ─────────────────────────────────────────────────────────────────────────────

def _panel_b_umap(ax, embedding, clusters, groove, palette, cluster_labels, umap_2d):
    emb = umap_2d if (umap_2d is not None and umap_2d.shape[0] == len(embedding)) \
          else _reduce_2d(embedding)

    groove   = np.asarray(groove, dtype=float)
    unique   = sorted(np.unique(clusters).tolist())

    for i, c in enumerate(unique):
        mask  = clusters == c
        color = palette[i]
        pts   = emb[mask]

        # Enveloppe convexe
        if len(pts) >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull  = ConvexHull(pts)
                verts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                        color=color, alpha=0.09, zorder=1)
                ax.plot(pts[verts, 0], pts[verts, 1],
                        color=color, alpha=0.35, linewidth=0.9,
                        linestyle="--", zorder=2)
            except Exception:
                pass

        ax.scatter(pts[:, 0], pts[:, 1],
                   s=48, color=color, alpha=0.80,
                   edgecolors="white", linewidths=0.4, zorder=4,
                   label=f"C{c}")

        # Centroïde + label
        cx, cy = pts.mean(axis=0)
        g_mask = groove[mask]
        g_mean = float(np.nanmean(g_mask)) if np.isfinite(g_mask).any() else float("nan")
        label_str = f"C{c}" + (f"\n{g_mean:.1f}" if not np.isnan(g_mean) else "")
        ax.text(cx, cy, label_str, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=color)])

    ax.set_xlabel("Dim 1 (UMAP)", fontsize=9)
    ax.set_ylabel("Dim 2 (UMAP)", fontsize=9)
    ax.set_title("B. Répartition dans l'espace latent", loc="left", fontsize=10)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(alpha=0.15, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=palette[i], label=f"C{c}", alpha=0.85)
               for i, c in enumerate(unique)]
    ax.legend(handles=handles, fontsize=7, loc="best",
              framealpha=0.9, edgecolor="#E5E7EB")


# ─────────────────────────────────────────────────────────────────────────────
# PANNEAU C — Profil acoustique par cluster (heatmap normalisée)
# ─────────────────────────────────────────────────────────────────────────────

def _panel_c_profile(ax, unique, clusters, df_realized, palette, cluster_labels):
    """
    Heatmap : lignes = clusters, colonnes = descripteurs D/S/E/P.
    Valeurs = z-score par rapport à la distribution globale.
    Divergent autour de 0 → rouge positif, bleu négatif.
    """
    import pandas as pd
    from matplotlib.colors import TwoSlopeNorm

    cols   = list(DESCRIPTOR_LABELS.keys())
    data   = np.zeros((len(unique), len(cols)))
    df_arr = df_realized[cols].values.astype(float)

    # Standardisation globale
    g_mean = np.nanmean(df_arr, axis=0)
    g_std  = np.nanstd(df_arr, axis=0)
    g_std[g_std < 1e-10] = 1.0

    for i, c in enumerate(unique):
        mask  = clusters == c
        vals  = df_arr[mask]
        med   = np.nanmedian(vals, axis=0)
        data[i] = (med - g_mean) / g_std

    vmax = max(0.5, float(np.max(np.abs(data))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

    # Annotations valeurs
    for i in range(len(unique)):
        for j in range(len(cols)):
            val = data[i, j]
            tc  = "white" if abs(val) > vmax * 0.55 else "#222222"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8.5, color=tc, fontweight="bold" if abs(val) > 0.3 else "normal")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([DESCRIPTOR_LABELS[c] for c in cols], fontsize=9)
    y_labels = [f"C{c}" for c in unique]
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(y_labels, fontsize=9, fontweight="bold")

    # Colorier les labels y par couleur de cluster
    for ytick, color in zip(ax.get_yticklabels(), palette):
        ytick.set_color(color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="z-score par rapport à la moyenne globale")
    ax.set_title("C. Profil acoustique par cluster (z-score)", loc="left", fontsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_labels(unique, cluster_labels, semantics, short=False):
    labels = []
    for c in unique:
        if cluster_labels and c in cluster_labels:
            labels.append(cluster_labels[c])
        elif semantics and c in semantics:
            lbl = semantics[c].get("label", f"C{c}")
            labels.append(f"C{c}\n{lbl}" if not short else f"C{c}: {lbl[:18]}")
        else:
            labels.append(f"C{c}")
    return labels


def _reduce_2d(embedding: np.ndarray) -> np.ndarray:
    if embedding.shape[1] <= 2:
        return embedding
    return PCA(n_components=2, random_state=42).fit_transform(embedding)


def _log_figure(unique, clusters, groove, semantics):
    groove = np.asarray(groove, dtype=float)
    print(f"\n[cluster_groove] n_stimuli={len(clusters)} | n_clusters={len(unique)}")
    for c in unique:
        mask  = clusters == c
        vals  = groove[mask]
        vals  = vals[np.isfinite(vals)]
        n     = len(vals)
        g_str = f"μ_groove={np.mean(vals):.3f}" if n > 0 else "μ_groove=—"
        lbl   = semantics[c]["label"] if semantics and c in semantics else ""
        print(f"  C{c}  n={mask.sum():>4}  {g_str}  [{lbl}]")