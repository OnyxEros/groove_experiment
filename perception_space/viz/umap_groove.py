"""
perception_space/viz/umap_groove.py
====================================
Figure publication-ready : groove, complexité et profil musical dans l'espace latent.

v2 :
    Panneau C optionnel — UMAP coloré par musical_background.
    Chaque profil (non_musician, amateur, semi_pro, pro) obtient une couleur
    distincte. Les centroïdes sont annotés avec le groove moyen du groupe.
    Ce panneau n'est généré que si background est fourni (non None).

Figure en 2 ou 3 panneaux selon les données disponibles :
    A — UMAP coloré par groove_mean + contours de clusters
    B — UMAP coloré par complexité_mean (si disponible)
    C — UMAP coloré par profil musical  (si background disponible)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
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

# Couleurs par profil musical — suffisamment distinctes, accessibles aux daltoniens
BACKGROUND_COLORS = {
    "non_musician": "#9CA3AF",   # gris neutre
    "amateur":      "#60A5FA",   # bleu clair
    "semi_pro":     "#34D399",   # vert menthe
    "pro":          "#F59E0B",   # ambre
}

BACKGROUND_LABELS = {
    "non_musician": "Non-musicien·ne",
    "amateur":      "Amateur·rice",
    "semi_pro":     "Semi-pro",
    "pro":          "Professionnel·le",
}

# Ordre d'affichage dans la légende
BACKGROUND_ORDER = ["non_musician", "amateur", "semi_pro", "pro"]


def plot_umap_groove(
    embedding:    np.ndarray,
    groove:       np.ndarray,
    complexity:   np.ndarray | None = None,
    clusters:     np.ndarray | None = None,
    umap_2d:      np.ndarray | None = None,
    background:   np.ndarray | None = None,
    out_path:     Path | None = None,
) -> plt.Figure:
    """
    Args:
        embedding   : embeddings réalisés (n, d) — réduit à 2D si nécessaire
        groove      : ratings groove alignés (n,)
        complexity  : ratings complexité alignés (n,) — optionnel
        clusters    : labels de cluster (n,) — optionnel, pour les contours
        umap_2d     : projection 2D du run d'analyse (n, 2) — optionnel
        background  : profil musical par stimulus (n,) dtype=object — optionnel
                      Valeurs attendues : "non_musician" | "amateur" | "semi_pro" | "pro" | None
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

    # Determine valid backgrounds
    has_background = (
        background is not None
        and len(background) == len(embedding)
        and any(b in BACKGROUND_ORDER for b in background if b is not None)
    )

    # Nombre de panneaux
    n_panels = 1
    if complexity is not None:
        n_panels += 1
    if has_background:
        n_panels += 1

    fig_w = 6.5 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.5))
    if n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(
        wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.12
    )

    panel_idx = 0

    # ── Panneau A : Groove ───────────────────────────────
    _scatter_panel(
        axes[panel_idx], emb, groove,
        clusters=clusters,
        cmap="RdYlGn",
        label="A  Groove perçu dans l'espace latent",
        cbar_label="Rating groove (1–7)",
        vmin=1, vmax=7,
        proj_label=proj_label,
    )
    panel_idx += 1

    # ── Panneau B : Complexité ───────────────────────────
    if complexity is not None:
        panel_letter = "B"
        _scatter_panel(
            axes[panel_idx], emb, np.asarray(complexity),
            clusters=clusters,
            cmap="RdYlBu_r",
            label=f"{panel_letter}  Complexité perçue dans l'espace latent",
            cbar_label="Rating complexité (1–7)",
            vmin=1, vmax=7,
            proj_label=proj_label,
        )
        panel_idx += 1

    # ── Panneau C : Profil musical ───────────────────────
    if has_background:
        panel_letter = ["B", "C"][panel_idx - 1]
        _background_panel(
            axes[panel_idx],
            emb=emb,
            groove=groove,
            background=background,
            proj_label=proj_label,
            label=f"{panel_letter}  Profil musical dans l'espace latent",
        )
        panel_idx += 1

    fig.suptitle(
        "Ratings perceptifs superposés à l'espace latent",
        fontsize=11, weight="semibold", y=0.97,
    )

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [fig] {Path(out_path).name}")

    return fig


# =========================================================
# PANNEAU GROOVE / COMPLEXITÉ (inchangé)
# =========================================================

def _scatter_panel(
    ax, emb, values, clusters, cmap,
    label, cbar_label, vmin, vmax, proj_label,
):
    # Contours de clusters en fond
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
                    cx, cy = pts.mean(axis=0)
                    ax.text(cx, cy, f"C{k}",
                            ha="center", va="center",
                            fontsize=7, color=colors[i],
                            fontweight="bold", alpha=0.6, zorder=3)
                except Exception:
                    pass

    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=values,
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=55, alpha=0.85,
        linewidths=0.4, edgecolors="white",
        zorder=4,
    )

    top_idx = int(np.argmax(values))
    bot_idx = int(np.argmin(values))
    for idx in [top_idx, bot_idx]:
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

    ax.text(0.99, 0.01, proj_label,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#9CA3AF", style="italic")


# =========================================================
# PANNEAU C — Profil musical
# =========================================================

def _background_panel(
    ax,
    emb:        np.ndarray,
    groove:     np.ndarray,
    background: np.ndarray,
    proj_label: str,
    label:      str,
) -> None:
    """
    Scatter UMAP coloré par profil musical.

    Pour chaque groupe :
      - Scatter des points avec la couleur du profil
      - Enveloppe convexe légère
      - Centroïde annoté avec le groove moyen du groupe

    Les points sans background renseigné sont tracés en gris neutre.
    """
    legend_handles: list[mpatches.Patch] = []

    # Points sans background — tracés en premier (fond)
    unknown_mask = np.array([b is None or b not in BACKGROUND_ORDER for b in background])
    if unknown_mask.any():
        ax.scatter(
            emb[unknown_mask, 0], emb[unknown_mask, 1],
            s=30, color="#D1D5DB", alpha=0.35,
            linewidths=0, zorder=2,
        )

    # Un groupe par profil
    for lvl in BACKGROUND_ORDER:
        mask  = np.array([b == lvl for b in background])
        n_pts = int(mask.sum())
        if n_pts == 0:
            continue

        color = BACKGROUND_COLORS[lvl]
        lbl   = BACKGROUND_LABELS[lvl]

        # ── Enveloppe convexe ─────────────────────────────
        pts = emb[mask]
        if n_pts >= 3:
            try:
                hull  = ConvexHull(pts)
                verts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                        color=color, alpha=0.10, zorder=1)
                ax.plot(pts[verts, 0], pts[verts, 1],
                        color=color, alpha=0.35,
                        linewidth=0.9, linestyle="--", zorder=2)
            except Exception:
                pass

        # ── Scatter ───────────────────────────────────────
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=50, color=color, alpha=0.80,
            linewidths=0.4, edgecolors="white",
            zorder=4,
        )

        # ── Centroïde annoté ──────────────────────────────
        cx, cy       = pts.mean(axis=0)
        groove_mean  = float(np.mean(groove[mask]))

        ax.scatter(cx, cy, marker="D", s=90, color=color,
                   edgecolors="white", linewidths=1.2, zorder=6)

        ax.text(
            cx, cy,
            f"{groove_mean:.1f}",
            ha="center", va="center",
            fontsize=7.5, fontweight="bold",
            color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground=color)],
        )

        legend_handles.append(
            mpatches.Patch(
                color=color,
                label=f"{lbl}  (n={n_pts}, ḡ={groove_mean:.2f})",
                alpha=0.85,
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7,
        framealpha=0.92,
        edgecolor="#E5E7EB",
        borderpad=0.5,
    )

    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.set_title(label, pad=7)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(alpha=0.15, linestyle=":", linewidth=0.5)

    ax.text(0.99, 0.01, proj_label,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#9CA3AF", style="italic")


# =========================================================
# HELPER
# =========================================================

def _reduce_2d(embedding: np.ndarray) -> np.ndarray:
    if embedding.shape[1] <= 2:
        return embedding
    from sklearn.decomposition import PCA
    print("[umap_groove] projection 2D : PCA fallback (umap_2d.npy absent)")
    return PCA(n_components=2, random_state=42).fit_transform(embedding)