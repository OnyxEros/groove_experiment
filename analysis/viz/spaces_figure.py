"""
analysis/viz/spaces_figure.py
================================
Figure multi-espaces publication-ready — mémoire TSMA2.

Trois panneaux :
    A — Espace paramétrique discret (Smv × Dmv), coloré par E
        Jitter léger pour visualiser les répétitions.
    B — UMAP espace émergent (D, I, V, Smv), coloré par Smv
        Trajectoire Smv=0→1→2 avec flèches claires + zones de densité.
    C — UMAP espace réalisé (D, I, V, Sreal, Ereal), clusters KMeans
        Enveloppes convexes + effectif de chaque cluster.

Corrections v2 :
    - Panneau B : flèches de trajectoire plus épaisses, labels plus grands,
      ellipses de confiance autour des centroïdes (±1 std).
    - Panneau C : annotation "Cn (n=XX)" sur chaque centroïde,
      enveloppe convexe avec transparence graduelle.
    - Flèches de transition inter-panneaux retirées (confondantes).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import numpy as np
from scipy.spatial import ConvexHull
try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


# ── Palette ───────────────────────────────────────────────
CLUSTER_COLORS = [
    "#2563EB",  # 0 — bleu
    "#16A34A",  # 1 — vert
    "#D97706",  # 2 — ambre
    "#DC2626",  # 3 — rouge
    "#7C3AED",  # 4 — violet
    "#0891B2",  # 5 — cyan
    "#DB2777",  # 6 — rose
    "#65A30D",  # 7 — vert clair
]

RC = {
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            8,
    "axes.labelsize":       8,
    "axes.titlesize":       9,
    "axes.titleweight":     "semibold",
    "axes.titlelocation":   "left",
    "axes.titlepad":        6,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.6,
    "axes.labelcolor":      "#374151",
    "axes.edgecolor":       "#D1D5DB",
    "xtick.labelsize":      7.5,
    "ytick.labelsize":      7.5,
    "xtick.color":          "#6B7280",
    "ytick.color":          "#6B7280",
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "grid.color":           "#E5E7EB",
    "grid.linewidth":       0.4,
    "legend.fontsize":      7.5,
    "legend.framealpha":    1.0,
    "legend.edgecolor":     "#E5E7EB",
    "legend.borderpad":     0.5,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
    "text.color":           "#111827",
}


def _add_panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.08, label,
            transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="top", ha="left", color="#111827")


def _draw_convex_hull(ax, points: np.ndarray, color: str,
                      alpha_fill: float = 0.10, alpha_line: float = 0.45) -> None:
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
        verts = np.append(hull.vertices, hull.vertices[0])
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                color=color, alpha=alpha_fill)
        ax.plot(points[verts, 0], points[verts, 1],
                color=color, alpha=alpha_line, linewidth=0.9, linestyle="-")
    except Exception:
        pass


def _draw_ellipse(ax, points: np.ndarray, color: str, n_std: float = 1.5) -> None:
    """Dessine une ellipse de confiance autour d'un nuage de points."""
    if len(points) < 4:
        return
    from matplotlib.patches import Ellipse
    cx, cy = points.mean(axis=0)
    cov = np.cov(points.T)
    if cov.ndim < 2 or cov.shape != (2, 2):
        return
    try:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * n_std * np.sqrt(vals)
        ell = Ellipse(
            (cx, cy), width=w, height=h, angle=angle,
            facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=0.8, linestyle="--",
        )
        ax.add_patch(ell)
    except Exception:
        pass


class SpacesFigure:

    def plot(self, df, umap_emergent, umap_realized, labels, path):
        plt.rcParams.update(RC)

        fig = plt.figure(figsize=(15, 4.8))
        gs = gridspec.GridSpec(
            1, 3,
            figure=fig,
            wspace=0.40,
            left=0.07,
            right=0.97,
            top=0.89,
            bottom=0.12,
        )

        axA = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1])
        axC = fig.add_subplot(gs[2])

        self._panel_A(axA, df)
        self._panel_B(axB, df, umap_emergent)
        self._panel_C(axC, df, umap_realized, labels)

        fig.suptitle(
            "Structure multi-espaces des stimuli rythmiques",
            fontsize=10.5, fontweight="semibold",
            color="#111827", y=0.98,
        )

        plt.savefig(path)
        plt.close()

    # ── Panel A : Espace paramétrique ─────────────────────

    def _panel_A(self, ax, df):
        rng = np.random.default_rng(42)
        n   = len(df)
        jitter = rng.uniform(-0.12, 0.12, (n, 2))

        e_vals = df["E"].values
        e_norm = (e_vals - e_vals.min()) / (np.ptp(e_vals) + 1e-9)

        sc = ax.scatter(
            df["S_mv"].values + jitter[:, 0],
            df["D_mv"].values + jitter[:, 1],
            c=e_norm, cmap="YlOrBr",
            s=22, alpha=0.70,
            linewidths=0.3, edgecolors="white",
            zorder=3,
        )

        # Annotations des conditions
        s_levels = sorted(df["S_mv"].unique())
        d_levels = sorted(df["D_mv"].unique())
        for s in s_levels:
            for d in d_levels:
                n_cell = ((df["S_mv"] == s) & (df["D_mv"] == d)).sum()
                ax.text(s, d - 0.28, f"n={n_cell}",
                        ha="center", va="top", fontsize=6, color="#9CA3AF")

        ax.set_xticks(s_levels)
        ax.set_xticklabels([f"$S_{{mv}}={s}$" for s in s_levels], fontsize=7.5)
        ax.set_yticks(d_levels)
        ax.set_yticklabels([f"$D_{{mv}}={d}$" for d in d_levels], fontsize=7.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.grid(alpha=0.30, linestyle=":", linewidth=0.5)
        ax.set_xlabel("$S_{mv}$ — distribution métrique", labelpad=5)
        ax.set_ylabel("$D_{mv}$ — densité stochastique", labelpad=5)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.set_label("$E$ (micro-timing)", fontsize=7.5, color="#6B7280")
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(["0.0", "0.5", "1.0"])
        cbar.outline.set_linewidth(0.4)

        _add_panel_label(ax, "A")
        ax.set_title("Espace paramétrique discret")

    # ── Panel B : UMAP émergent ────────────────────────────

    def _panel_B(self, ax, df, umap_emergent):
        if umap_emergent is None:
            ax.text(0.5, 0.5, "UMAP non calculé",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#6B7280")
            _add_panel_label(ax, "B")
            ax.set_title("Espace émergent (UMAP)")
            return

        s_vals  = sorted(df["S_mv"].unique())
        s_array = df["S_mv"].values
        s_norm  = (s_array - min(s_vals)) / (max(s_vals) - min(s_vals) + 1e-9)

        sc = ax.scatter(
            umap_emergent[:, 0], umap_emergent[:, 1],
            c=s_norm, cmap="Blues",
            s=20, alpha=0.70,
            linewidths=0.3, edgecolors="white",
            zorder=3,
        )

        # Densité de fond (si seaborn dispo)
        if _HAS_SNS and len(umap_emergent) > 20:
            try:
                sns.kdeplot(
                    x=umap_emergent[:, 0], y=umap_emergent[:, 1],
                    ax=ax, levels=3, color="#94A3B8",
                    alpha=0.18, linewidths=0.5,
                )
            except Exception:
                pass

        # Ellipses de confiance + centroïdes + trajectoire
        centroids = []
        for s in s_vals:
            mask = s_array == s
            if mask.sum() < 2:
                continue
            pts = umap_emergent[mask]
            c   = pts.mean(axis=0)
            centroids.append((s, c))

            # Ellipse
            _draw_ellipse(ax, pts, color="#1E3A5F", n_std=1.2)

            # Centroïde
            ax.scatter(c[0], c[1], s=80, color="white", zorder=6,
                       edgecolors="#1E3A5F", linewidths=1.4)

            # Label
            y_range = np.ptp(umap_emergent[:, 1]) if len(umap_emergent) > 1 else 1
            ax.annotate(
                f"$S_{{mv}}={int(s)}$",
                xy=(c[0], c[1]),
                xytext=(c[0], c[1] + y_range * 0.07),
                fontsize=8, ha="center", color="#1E3A5F",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            )

        # Flèches de trajectoire entre centroïdes
        if len(centroids) > 1:
            pts = np.array([c for _, c in centroids])
            for i in range(len(pts) - 1):
                ax.annotate(
                    "",
                    xy=pts[i + 1], xytext=pts[i],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=1.8,
                        color="#1E3A5F",
                        connectionstyle="arc3,rad=0.18",
                        mutation_scale=12,
                    ),
                    zorder=7,
                )

        ax.set_xlabel("UMAP dim. 1", labelpad=5)
        ax.set_ylabel("UMAP dim. 2", labelpad=5)
        ax.grid(alpha=0.20, linestyle=":", linewidth=0.4)
        ax.tick_params(labelbottom=False, labelleft=False)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.set_label("$S_{mv}$", fontsize=7.5, color="#6B7280")
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels([str(int(s_vals[0])),
                             str(int(np.median(s_vals))),
                             str(int(s_vals[-1]))])
        cbar.outline.set_linewidth(0.4)

        _add_panel_label(ax, "B")
        ax.set_title("Espace émergent (UMAP)")

    # ── Panel C : UMAP réalisé + clusters ─────────────────

    def _panel_C(self, ax, df, umap_realized, labels):
        unique_labels = np.unique(labels)
        n_clusters    = len(unique_labels)
        colors = (CLUSTER_COLORS * 4)[:n_clusters]

        for i, k in enumerate(unique_labels):
            mask  = labels == k
            pts   = umap_realized[mask]
            color = colors[i]
            n_k   = int(mask.sum())

            # Enveloppe convexe
            _draw_convex_hull(ax, pts, color, alpha_fill=0.10, alpha_line=0.40)

            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=18, color=color, alpha=0.72,
                linewidths=0.3, edgecolors="white",
                zorder=3,
            )

            # Centroïde avec annotation effectif
            cx, cy = pts.mean(axis=0)
            ax.scatter(cx, cy, marker="X", s=65, color=color,
                       edgecolors="white", linewidths=0.9, zorder=5)

            # Label "Cn (n=XX)"
            ax.text(
                cx, cy,
                f"C{k}\n(n={n_k})",
                ha="center", va="center",
                fontsize=6.5, fontweight="bold",
                color=color, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            )

        ax.set_xlabel("UMAP dim. 1", labelpad=5)
        ax.set_ylabel("UMAP dim. 2", labelpad=5)
        ax.grid(alpha=0.20, linestyle=":", linewidth=0.4)
        ax.tick_params(labelbottom=False, labelleft=False)

        # Légende compacte
        legend_handles = [
            mpatches.Patch(color=colors[i], label=f"C{k}")
            for i, k in enumerate(unique_labels)
        ]
        ax.legend(
            handles=legend_handles,
            loc="best", fontsize=6.5,
            ncol=2, handlelength=0.8,
            handleheight=0.8, borderpad=0.4,
            labelspacing=0.3, columnspacing=0.6,
            framealpha=0.95,
        )

        _add_panel_label(ax, "C")
        ax.set_title("Espace réalisé (UMAP, clusters KMeans)")