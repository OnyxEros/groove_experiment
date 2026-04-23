import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull


class SpacesFigure:

    _GS_LEFT   = 0.07
    _GS_RIGHT  = 0.96
    _GS_TOP    = 0.87
    _GS_BOTTOM = 0.22   # plus d'espace en bas pour colorbars + note
    _GS_WSPACE = 0.38
    _GS_NCOLS  = 3

    @classmethod
    def _panel_xbounds(cls):
        left, right = cls._GS_LEFT, cls._GS_RIGHT
        n, ws = cls._GS_NCOLS, cls._GS_WSPACE
        pw = (right - left) / (n + (n - 1) * ws)
        sp = ws * pw
        return [(left + i * (pw + sp), left + i * (pw + sp) + pw) for i in range(n)]

    def plot(self, df, umap_gen, umap_emergent, umap_audio, labels, path):

        # =====================================================
        # STYLE — publication-grade
        # =====================================================
        plt.rcParams.update({
            "font.family":          "serif",
            "font.serif":           ["Times New Roman", "DejaVu Serif"],
            "font.size":            10,
            "axes.labelsize":       10,
            "axes.titlesize":       11,
            "axes.titleweight":     "bold",
            "axes.titlelocation":   "left",
            "axes.titlepad":        8,
            "axes.spines.top":      False,
            "axes.spines.right":    False,
            "axes.linewidth":       0.8,
            "xtick.labelsize":      8,
            "ytick.labelsize":      8,
            "xtick.major.width":    0.8,
            "ytick.major.width":    0.8,
            "legend.fontsize":      8,
            "legend.framealpha":    0.9,
            "legend.edgecolor":     "#cccccc",
            "figure.dpi":           150,
        })

        fig = plt.figure(figsize=(18, 6.5))

        gs = gridspec.GridSpec(
            1, self._GS_NCOLS,
            figure=fig,
            wspace=self._GS_WSPACE,
            left=self._GS_LEFT,
            right=self._GS_RIGHT,
            top=self._GS_TOP,
            bottom=self._GS_BOTTOM,
        )

        axA = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1])
        axC = fig.add_subplot(gs[2])

        bounds = self._panel_xbounds()

        # =====================================================
        # (A) CONTROL SPACE
        # =====================================================
        rng = np.random.default_rng(42)
        jitter_x = rng.uniform(-0.06, 0.06, len(df))
        jitter_y = rng.uniform(-0.06, 0.06, len(df))

        scA = axA.scatter(
            df["S_mv"] + jitter_x,
            df["D_mv"] + jitter_y,
            c=df["E"], cmap="plasma",
            s=22, alpha=0.70, linewidths=0, zorder=3
        )

        axA.set_xticks(sorted(df["S_mv"].unique()))
        axA.set_yticks(sorted(df["D_mv"].unique()))
        axA.set_xlabel("Syncopation ($S_{mv}$)")
        axA.set_ylabel("Density ($D_{mv}$)")
        axA.set_title("(A) Control Space")
        axA.grid(alpha=0.12, linestyle=":", linewidth=0.7)

        cax_A = inset_axes(
            axA, width="100%", height="5%",
            loc="lower left",
            bbox_to_anchor=(0, -0.28, 1, 1),
            bbox_transform=axA.transAxes, borderpad=0
        )
        cb_A = fig.colorbar(scA, cax=cax_A, orientation="horizontal")
        cb_A.set_label("Micro-timing ($E$)", fontsize=8, labelpad=3)
        cb_A.ax.tick_params(labelsize=7)

        # =====================================================
        # (B) EMERGENT SPACE
        # =====================================================
        scB = axB.scatter(
            umap_emergent[:, 0], umap_emergent[:, 1],
            c=df["S_mv"], cmap="viridis",
            s=20, alpha=0.80, linewidths=0, zorder=3
        )

        try:
            sns.kdeplot(
                x=umap_emergent[:, 0], y=umap_emergent[:, 1],
                ax=axB, levels=4,
                color="#999999", alpha=0.15, linewidths=0.6
            )
        except Exception:
            pass

        ordered_s = sorted(df["S_mv"].unique())
        means = []
        for s in ordered_s:
            mask = (df["S_mv"].values == s)
            subset = umap_emergent[mask]
            if len(subset) > 0:
                c = subset.mean(axis=0)
                means.append((s, c))
                axB.scatter(c[0], c[1], c="white", s=80, zorder=5,
                            edgecolors="#111111", linewidths=1.2)
                y_range = np.ptp(umap_emergent[:, 1])
                axB.text(c[0], c[1] + y_range * 0.035,
                         f"$S_{{mv}}={s}$",
                         ha="center", fontsize=7.5, color="#111111",
                         zorder=6, fontweight="bold")

        if len(means) > 1:
            pts = np.array([m[1] for m in means])
            for i in range(len(pts) - 1):
                axB.annotate(
                    "",
                    xy=pts[i + 1], xytext=pts[i],
                    arrowprops=dict(
                        arrowstyle="-|>", lw=1.2, color="#333333",
                        connectionstyle="arc3,rad=0.18"
                    ), zorder=4
                )

        axB.set_xlabel("UMAP Dim. 1")
        axB.set_ylabel("UMAP Dim. 2")
        axB.set_title("(B) Emergent Feature Structure")
        axB.grid(alpha=0.12, linestyle=":", linewidth=0.7)

        sm_B = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(df["S_mv"].min(), df["S_mv"].max())
        )
        sm_B.set_array([])
        cax_B = inset_axes(
            axB, width="100%", height="5%",
            loc="lower left",
            bbox_to_anchor=(0, -0.28, 1, 1),
            bbox_transform=axB.transAxes, borderpad=0
        )
        cb_B = fig.colorbar(sm_B, cax=cax_B, orientation="horizontal")
        cb_B.set_label("Syncopation ($S_{mv}$)", fontsize=8, labelpad=3)
        cb_B.ax.tick_params(labelsize=7)
        cb_B.set_ticks(sorted(df["S_mv"].unique()))

        # =====================================================
        # (C) PERCEPTUAL ORGANIZATION
        # =====================================================
        unique_labels = np.unique(labels)
        cmap_c = plt.cm.get_cmap("tab10", len(unique_labels))

        for k in unique_labels:
            mask = (labels == k)
            subset = umap_audio[mask]
            color = cmap_c(k % 10)

            axC.scatter(
                subset[:, 0], subset[:, 1],
                c=[color], s=18, alpha=0.75, linewidths=0, zorder=3
            )

            if len(subset) > 3:
                try:
                    hull = ConvexHull(subset)
                    hull_pts = subset[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]])
                    axC.fill(hull_pts[:, 0], hull_pts[:, 1],
                             alpha=0.08, color=color, zorder=1)
                    axC.plot(hull_pts[:, 0], hull_pts[:, 1],
                             color=color, alpha=0.40, linewidth=0.7, zorder=2)
                except Exception:
                    pass

            centroid = subset.mean(axis=0)
            axC.scatter(centroid[0], centroid[1],
                        c="white", s=70, zorder=5,
                        edgecolors="#111111", linewidths=1.2)

        axC.set_xlabel("UMAP Dim. 1")
        axC.set_ylabel("UMAP Dim. 2")
        axC.set_title("(C) Perceptual Organization")
        axC.grid(alpha=0.12, linestyle=":", linewidth=0.7)

        legend_handles = [
            mpatches.Patch(color=cmap_c(k % 10), label=f"Cluster {k}")
            for k in unique_labels
        ]
        axC.legend(
            handles=legend_handles,
            loc="upper right", fontsize=7.5,
            framealpha=0.92, edgecolor="#cccccc",
            handlelength=1.0, handleheight=0.85,
            borderpad=0.6
        )

        # =====================================================
        # FLÈCHES INTER-PANELS
        # Positionnées dans le tiers supérieur — au-dessus des labels Y
        # =====================================================
        y_arr  = self._GS_BOTTOM + (self._GS_TOP - self._GS_BOTTOM) * 0.78
        y_lbl  = y_arr + 0.04

        def draw_arrow(x0_panel, x1_panel, lines):
            x0 = x0_panel + 0.012
            x1 = x1_panel - 0.012
            xm = (x0 + x1) / 2
            fig.patches.append(mpatches.FancyArrowPatch(
                posA=(x0, y_arr), posB=(x1, y_arr),
                arrowstyle="-|>", mutation_scale=13,
                lw=1.5, color="#555555",
                transform=fig.transFigure, zorder=10, clip_on=False
            ))
            fig.text(xm, y_lbl, "\n".join(lines),
                     ha="center", va="bottom",
                     fontsize=8, color="#555555",
                     linespacing=1.3)

        draw_arrow(bounds[0][1], bounds[1][0], ["Feature Extraction", "+ UMAP"])
        draw_arrow(bounds[1][1], bounds[2][0], ["Perceptual", "Projection"])

        # =====================================================
        # TITRE & NOTE DE BAS DE PAGE
        # =====================================================
        fig.suptitle(
            "From Parametric Control to Perceptual Structure: "
            "A Multi-Space Representation of Groove",
            fontsize=12, weight="bold", y=0.98,
            fontfamily="serif"
        )

        # Note UMAP en bas de figure — sous les colorbars
        fig.text(
            0.5, 0.01,
            r"$\dagger$ UMAP: $\mathcal{X} \subset \mathbb{R}^n \rightarrow \mathcal{Y} \subset \mathbb{R}^2$"
            r" — Uniform Manifold Approximation and Projection (McInnes et al., 2018).",
            ha="center", va="bottom",
            fontsize=7.5, color="#888888",
            style="italic"
        )

        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()