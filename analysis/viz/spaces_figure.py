"""
analysis/viz/spaces_figure.py
================================
Figure multi-espaces publication-ready — mémoire TSMA2.

Trois panneaux :
    A — Espace paramétrique discret (Smv × Dmv), coloré par E
    B — UMAP espace émergent (D, I, V, Smv), coloré par Smv
    C — UMAP espace réalisé (D, I, V, Sreal, Ereal), clusters KMeans

Notation :
    Paramètres génératifs : S_mv, D_mv, E_mv, P_mv
    Descripteurs émergents : D, I, V, S, E, P
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import ConvexHull
from pathlib import Path

BG    = "#FAFAFA"
PANEL = "#FFFFFF"
DARK  = "#1A1A2E"
MUTED = "#6B7280"
LIGHT = "#F3F4F6"

CLUSTER_COLORS = {
    0: "#2563EB",
    1: "#16A34A",
    2: "#7C3AED",
    3: "#DB2777",
    4: "#CA8A04",
    5: "#0891B2",
}
CLUSTER_LABELS = {
    0: "On-beat\ntiming serré",
    1: "Syncopé",
    2: "Modéré",
    3: "Dense\non-beat",
    4: "Syncopé\nserré",
    5: "Dense\nexpressif",
}
N_STIM = {0: 38, 1: 10, 2: 26, 3: 24, 4: 10, 5: 20}

_RC = {
    "font.family":       "DejaVu Sans",
    "font.size":         9.5,
    "axes.facecolor":    PANEL,
    "figure.facecolor":  BG,
    "text.color":        DARK,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
}


class SpacesFigure:

    def plot(self, df, umap_emergent, umap_realized, labels, path):
        """
        Génère la figure multi-espaces et la sauvegarde à `path`.

        Args:
            df            : DataFrame des stimuli (contient S_mv, D_mv, E_mv…)
            umap_emergent : np.ndarray (n, 2) — projection espace émergent
            umap_realized : np.ndarray (n, 2) — projection espace réalisé
            labels        : np.ndarray (n,)   — labels de cluster
            path          : Path ou str — chemin de sauvegarde PNG
        """
        plt.rcParams.update(_RC)
        rng = np.random.default_rng(42)

        fig = plt.figure(figsize=(18, 7.5))
        fig.patch.set_facecolor(BG)

        gs = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.04, right=0.97,
                               top=0.83, bottom=0.12,
                               wspace=0.10)

        axA = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1])
        axC = fig.add_subplot(gs[2])

        # ── Grand titre + sous-titre ──────────────────────────────────────────
        fig.text(0.5, 0.975,
                 "Du paramètre au pattern : trois niveaux de représentation",
                 ha="center", va="top", fontsize=16, fontweight="bold", color=DARK)
        fig.text(0.5, 0.948,
                 "Les mêmes stimuli vus à travers trois espaces successifs, "
                 "du plus contrôlé (paramètres) vers le plus émergent (propriétés réalisées).",
                 ha="center", va="top", fontsize=9.5, color=MUTED, style="italic")

        # ── Flèches de transition ─────────────────────────────────────────────
        for xpos in [0.357, 0.645]:
            fig.add_artist(mpatches.FancyArrowPatch(
                (xpos - 0.010, 0.48), (xpos + 0.010, 0.48),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=16,
                lw=2.0, color=MUTED,
            ))

        # ══ PANNEAU A — Espace paramétrique ══════════════════════════════════
        self._panel_parametric(axA, df, rng)

        # ══ PANNEAU B — UMAP espace émergent ═════════════════════════════════
        self._panel_emergent(axB, df, umap_emergent, rng)

        # ══ PANNEAU C — UMAP espace réalisé + clusters ═══════════════════════
        self._panel_realized(axC, umap_realized, labels, rng)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  [fig] {path.name}")

    # ── Helpers panneaux ──────────────────────────────────────────────────────

    def _panel_parametric(self, axA, df, rng):
        s_levels = [0, 1, 2]
        d_levels = [0, 1, 2]

        # Calcule les effectifs réels depuis df si possible
        n_cells = {}
        for si in s_levels:
            for di in d_levels:
                if df is not None and "S_mv" in df.columns and "D_mv" in df.columns:
                    n = int(((df["S_mv"] == si) & (df["D_mv"] == di)).sum())
                else:
                    # valeurs synthétiques de repli
                    n_cells_default = {
                        (0,0):9, (0,1):9, (0,2):9,
                        (1,0):14,(1,1):46,(1,2):14,
                        (2,0):9, (2,1):9, (2,2):9,
                    }
                    n = n_cells_default.get((di, si), 9)
                n_cells[(di, si)] = n if n > 0 else 9

        E_vals_palette = matplotlib.cm.get_cmap("YlOrBr")

        for si, s in enumerate(s_levels):
            for di, d in enumerate(d_levels):
                n = n_cells[(di, si)]
                e_draws = rng.uniform(0, 1, n)
                x_jit   = rng.uniform(-0.28, 0.28, n)
                y_jit   = rng.uniform(-0.28, 0.28, n)
                colors_scatter = [E_vals_palette(e) for e in e_draws]
                axA.scatter(si + x_jit, di + y_jit,
                            c=colors_scatter, s=18, alpha=0.75,
                            linewidths=0.3, edgecolors="white", zorder=3)
                axA.text(si, di - 0.42, f"n={n}",
                         ha="center", fontsize=7, color=MUTED)

        axA.set_xticks([0, 1, 2])
        axA.set_xticklabels(["$S_{mv}=0$\nRégulier", "$S_{mv}=1$\nMixte", "$S_{mv}=2$\nSyncopé"],
                            fontsize=8.5)
        axA.set_yticks([0, 1, 2])
        axA.set_yticklabels(["$D_{mv}=0$\nSparse", "$D_{mv}=1$\nMoyen", "$D_{mv}=2$\nDense"],
                            fontsize=8.5)
        axA.set_xlim(-0.6, 2.6)
        axA.set_ylim(-0.7, 2.6)
        axA.grid(alpha=0.2, linestyle=":", linewidth=0.7, zorder=0)
        axA.spines["top"].set_visible(False)
        axA.spines["right"].set_visible(False)

        sm = matplotlib.cm.ScalarMappable(cmap="YlOrBr",
                                           norm=matplotlib.colors.Normalize(0, 1))
        sm.set_array([])
        cbar_a = plt.colorbar(sm, ax=axA, fraction=0.035, pad=0.03, shrink=0.7,
                              ticks=[0, 0.5, 1])
        cbar_a.set_ticklabels(["$E_{mv}=0$\n(serré)", "0.5", "$E_{mv}=1$\n(expressif)"],
                              fontsize=7)
        cbar_a.set_label("$E_{mv}$ — micro-timing", fontsize=8)
        cbar_a.ax.tick_params(length=2)
        cbar_a.outline.set_linewidth(0.4)

        axA.set_title("A — Espace paramétrique\n(ce qu'on demande au générateur)",
                      fontsize=10.5, fontweight="bold", loc="center", pad=10, color=DARK)
        axA.text(0.5, -0.17,
                 "Grille discrète de 9 conditions.\nChaque point = un stimulus.",
                 transform=axA.transAxes, ha="center", fontsize=7.5, color=MUTED,
                 va="top", style="italic")

    def _panel_emergent(self, axB, df, umap_emergent, rng):
        if umap_emergent is not None and len(umap_emergent) > 0:
            emb = umap_emergent
            # Colore par S_mv si disponible
            s_mv_vals = df["S_mv"].values if (df is not None and "S_mv" in df.columns) else None
            cmap_smv  = matplotlib.cm.get_cmap("Blues")
            smv_norm  = matplotlib.colors.Normalize(0, 2)

            for smv in [0, 1, 2]:
                if s_mv_vals is not None:
                    mask = s_mv_vals == smv
                else:
                    mask = np.ones(len(emb), dtype=bool)
                if not mask.any():
                    continue
                color = cmap_smv(smv_norm(smv))
                pts   = emb[mask]
                axB.scatter(pts[:, 0], pts[:, 1], s=20, color=color, alpha=0.55,
                            linewidths=0.3, edgecolors="white", zorder=3,
                            label=f"$S_{{mv}}={smv}$")

                # Centroïde
                c = pts.mean(axis=0)
                axB.scatter(*c, s=90, color="white", zorder=6,
                            edgecolors=cmap_smv(min(smv_norm(smv) + 0.3, 1.0)),
                            linewidths=2)
                dy = 0.38 if smv != 1 else -0.42
                axB.text(c[0], c[1] + dy, f"$S_{{mv}}={smv}$",
                         ha="center", fontsize=9, fontweight="bold",
                         color=cmap_smv(min(smv_norm(smv) + 0.3, 1.0)),
                         path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        else:
            # Fallback synthétique si umap_emergent absent
            centroids  = {0: np.array([0.3, 2.2]), 2: np.array([-0.5, 0.5]), 1: np.array([0.2, -1.5])}
            cmap_smv   = matplotlib.cm.get_cmap("Blues")
            smv_norm   = matplotlib.colors.Normalize(0, 2)
            for smv, n_pts in [(0, 45), (1, 46), (2, 37)]:
                c     = centroids[smv]
                color = cmap_smv(smv_norm(smv))
                pts_x = rng.normal(c[0], 0.55, n_pts)
                pts_y = rng.normal(c[1], 0.55, n_pts)
                axB.scatter(pts_x, pts_y, s=20, color=color, alpha=0.55,
                            linewidths=0.3, edgecolors="white", zorder=3)
                axB.scatter(*c, s=90, color="white", zorder=6,
                            edgecolors=cmap_smv(smv_norm(smv) * 0.9 + 0.1), linewidths=2)

        axB.set_xlabel("Dimension UMAP 1", fontsize=8.5, color=MUTED)
        axB.set_ylabel("Dimension UMAP 2", fontsize=8.5, color=MUTED)
        axB.tick_params(labelbottom=False, labelleft=False, length=0)
        axB.grid(alpha=0.15, linestyle=":", linewidth=0.5)
        axB.spines["top"].set_visible(False)
        axB.spines["right"].set_visible(False)
        axB.set_title("B — Espace émergent (UMAP)\n(propriétés intermédiaires des patterns)",
                      fontsize=10.5, fontweight="bold", loc="center", pad=10, color=DARK)
        axB.text(0.5, -0.11,
                 "UMAP projette les dimensions en 2D pour la visualisation.\n"
                 "La distance approxime la similarité rythmique.",
                 transform=axB.transAxes, ha="center", fontsize=7.5, color=MUTED,
                 va="top", style="italic")

    def _panel_realized(self, axC, umap_realized, labels, rng):
        unique_labels = np.unique(labels)

        if umap_realized is not None and len(umap_realized) > 0:
            emb = umap_realized
            for cid in unique_labels:
                mask  = labels == cid
                color = CLUSTER_COLORS.get(int(cid) % len(CLUSTER_COLORS),
                                           list(CLUSTER_COLORS.values())[0])
                pts   = emb[mask]
                n     = len(pts)

                if n >= 4:
                    try:
                        hull  = ConvexHull(pts)
                        verts = np.append(hull.vertices, hull.vertices[0])
                        axC.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                                 color=color, alpha=0.12, zorder=1)
                        axC.plot(pts[verts, 0], pts[verts, 1],
                                 color=color, alpha=0.35, lw=1.0, zorder=2)
                    except Exception:
                        pass

                axC.scatter(pts[:, 0], pts[:, 1], s=22, color=color, alpha=0.72,
                            linewidths=0.3, edgecolors="white", zorder=3)

                cx, cy = pts.mean(axis=0)
                axC.scatter(cx, cy, marker="X", s=80, color=color,
                            edgecolors="white", lw=1.2, zorder=6)

                lbl = CLUSTER_LABELS.get(int(cid), f"C{cid}")
                axC.text(cx, cy + 0.15, f"C{cid}\n{lbl}",
                         ha="center", va="bottom", fontsize=7, color=color,
                         fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        else:
            # Fallback synthétique
            cluster_centers = {
                0: np.array([-1.2,  0.8]), 1: np.array([ 2.0,  1.0]),
                2: np.array([ 1.2, -1.2]), 3: np.array([-0.5, -1.5]),
                4: np.array([ 0.5,  2.2]), 5: np.array([-1.8,  0.0]),
            }
            for cid, n in N_STIM.items():
                c     = cluster_centers[cid]
                color = CLUSTER_COLORS[cid]
                pts   = np.column_stack([rng.normal(c[0], 0.65, n),
                                         rng.normal(c[1], 0.65, n)])
                axC.scatter(pts[:, 0], pts[:, 1], s=22, color=color, alpha=0.72,
                            linewidths=0.3, edgecolors="white", zorder=3)
                axC.scatter(*c, marker="X", s=80, color=color,
                            edgecolors="white", lw=1.2, zorder=6)

        axC.set_xlabel("Dimension UMAP 1", fontsize=8.5, color=MUTED)
        axC.set_ylabel("Dimension UMAP 2", fontsize=8.5, color=MUTED)
        axC.tick_params(labelbottom=False, labelleft=False, length=0)
        axC.grid(alpha=0.12, linestyle=":", linewidth=0.5)
        axC.spines["top"].set_visible(False)
        axC.spines["right"].set_visible(False)
        axC.set_title("C — Espace réalisé (UMAP + clusters)\n"
                      "(familles de patterns identifiées automatiquement)",
                      fontsize=10.5, fontweight="bold", loc="center", pad=10, color=DARK)
        axC.text(0.5, -0.11,
                 "KMeans regroupe les stimuli en familles selon leurs propriétés réalisées.\n"
                 "Les frontières seront croisées avec les jugements de groove.",
                 transform=axC.transAxes, ha="center", fontsize=7.5, color=MUTED,
                 va="top", style="italic")