import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull


class SpacesFigure:
    """
    Three-panel figure showing the transformation from parametric design space
    to emergent feature structure to realized descriptor organization.
    
    Panel A: Parametric design space (S_mv, D_mv, E)
    Panel B: UMAP projection of emergent descriptors (D, I, V, S_mv)
    Panel C: UMAP projection of realized descriptors (D, I, V, S_real, E_real)
    """

    _GS_LEFT   = 0.06
    _GS_RIGHT  = 0.97
    _GS_TOP    = 0.88
    _GS_BOTTOM = 0.20
    _GS_WSPACE = 0.35
    _GS_NCOLS  = 3

    @classmethod
    def _panel_xbounds(cls):
        left, right = cls._GS_LEFT, cls._GS_RIGHT
        n, ws = cls._GS_NCOLS, cls._GS_WSPACE
        pw = (right - left) / (n + (n - 1) * ws)
        sp = ws * pw
        return [(left + i * (pw + sp), left + i * (pw + sp) + pw) for i in range(n)]

    def plot(self, df, umap_emergent, umap_realized, labels, path):

        # =====================================================
        # PUBLICATION-GRADE STYLE
        # =====================================================
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.fontsize": 7,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#999999",
            "figure.dpi": 150,
        })

        fig = plt.figure(figsize=(17, 5.8))

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
        # PANEL A: PARAMETRIC DESIGN SPACE
        # =====================================================
        rng = np.random.default_rng(42)
        jitter_x = rng.uniform(-0.08, 0.08, len(df))
        jitter_y = rng.uniform(-0.08, 0.08, len(df))

        scA = axA.scatter(
            df["S_mv"] + jitter_x,
            df["D_mv"] + jitter_y,
            c=df["E"],
            cmap="plasma",
            s=28,
            alpha=0.65,
            linewidths=0.3,
            edgecolors="white",
            zorder=3
        )

        axA.set_xticks(sorted(df["S_mv"].unique()))
        axA.set_yticks(sorted(df["D_mv"].unique()))
        axA.set_xlabel("Syncopation control ($S_{\\mathrm{mv}}$)", fontsize=9)
        axA.set_ylabel("Density control ($D_{\\mathrm{mv}}$)", fontsize=9)
        axA.set_title("A  Parametric Design Space", fontsize=10, pad=8)
        axA.grid(alpha=0.20, linestyle=":", linewidth=0.6, color="#cccccc")
        axA.set_xlim(-0.3, 2.3)
        axA.set_ylim(-0.3, 2.3)

        cax_A = inset_axes(
            axA,
            width="100%",
            height="4%",
            loc="lower left",
            bbox_to_anchor=(0, -0.32, 1, 1),
            bbox_transform=axA.transAxes,
            borderpad=0
        )
        cb_A = fig.colorbar(scA, cax=cax_A, orientation="horizontal")
        cb_A.set_label("Micro-timing amplitude ($E$)", fontsize=7.5, labelpad=2)
        cb_A.ax.tick_params(labelsize=6.5)

        # =====================================================
        # PANEL B: EMERGENT DESCRIPTORS SPACE
        # =====================================================
        scB = axB.scatter(
            umap_emergent[:, 0],
            umap_emergent[:, 1],
            c=df["S_mv"],
            cmap="viridis",
            s=24,
            alpha=0.75,
            linewidths=0.3,
            edgecolors="white",
            zorder=3
        )

        try:
            sns.kdeplot(
                x=umap_emergent[:, 0],
                y=umap_emergent[:, 1],
                ax=axB,
                levels=3,
                color="#777777",
                alpha=0.12,
                linewidths=0.5
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
                axB.scatter(
                    c[0], c[1],
                    c="white",
                    s=100,
                    zorder=5,
                    edgecolors="#000000",
                    linewidths=1.5
                )
                y_range = np.ptp(umap_emergent[:, 1])
                axB.text(
                    c[0], c[1] + y_range * 0.04,
                    f"${s}$",
                    ha="center",
                    fontsize=7,
                    color="#000000",
                    zorder=6,
                    weight="bold"
                )

        if len(means) > 1:
            pts = np.array([m[1] for m in means])
            for i in range(len(pts) - 1):
                axB.annotate(
                    "",
                    xy=pts[i + 1],
                    xytext=pts[i],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=1.3,
                        color="#222222",
                        connectionstyle="arc3,rad=0.15"
                    ),
                    zorder=4
                )

        axB.set_xlabel("UMAP dimension 1", fontsize=9)
        axB.set_ylabel("UMAP dimension 2", fontsize=9)
        axB.set_title("B  Emergent Feature Space", fontsize=10, pad=8)
        axB.grid(alpha=0.18, linestyle=":", linewidth=0.6, color="#cccccc")

        sm_B = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(df["S_mv"].min(), df["S_mv"].max())
        )
        sm_B.set_array([])
        cax_B = inset_axes(
            axB,
            width="100%",
            height="4%",
            loc="lower left",
            bbox_to_anchor=(0, -0.32, 1, 1),
            bbox_transform=axB.transAxes,
            borderpad=0
        )
        cb_B = fig.colorbar(sm_B, cax=cax_B, orientation="horizontal")
        cb_B.set_label("Syncopation ($S_{\\mathrm{mv}}$)", fontsize=7.5, labelpad=2)
        cb_B.ax.tick_params(labelsize=6.5)
        cb_B.set_ticks(sorted(df["S_mv"].unique()))

        # =====================================================
        # PANEL C: REALIZED DESCRIPTORS SPACE
        # =====================================================
        unique_labels = np.unique(labels)
        cmap_c = plt.cm.get_cmap("tab10", len(unique_labels))

        for k in unique_labels:
            mask = (labels == k)
            subset = umap_realized[mask]
            color = cmap_c(k % 10)

            axC.scatter(
                subset[:, 0],
                subset[:, 1],
                c=[color],
                s=20,
                alpha=0.70,
                linewidths=0.3,
                edgecolors="white",
                zorder=3
            )

            if len(subset) > 3:
                try:
                    hull = ConvexHull(subset)
                    hull_pts = subset[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]])
                    axC.fill(
                        hull_pts[:, 0],
                        hull_pts[:, 1],
                        alpha=0.12,
                        color=color,
                        zorder=1
                    )
                    axC.plot(
                        hull_pts[:, 0],
                        hull_pts[:, 1],
                        color=color,
                        alpha=0.50,
                        linewidth=0.9,
                        zorder=2
                    )
                except Exception:
                    pass

            centroid = subset.mean(axis=0)
            axC.scatter(
                centroid[0],
                centroid[1],
                c="white",
                s=90,
                zorder=5,
                edgecolors="#000000",
                linewidths=1.5,
                marker="X"
            )

        axC.set_xlabel("UMAP dimension 1", fontsize=9)
        axC.set_ylabel("UMAP dimension 2", fontsize=9)
        axC.set_title("C  Realized Descriptor Space", fontsize=10, pad=8)
        axC.grid(alpha=0.18, linestyle=":", linewidth=0.6, color="#cccccc")

        legend_handles = [
            mpatches.Patch(color=cmap_c(k % 10), label=f"Cluster {k}")
            for k in unique_labels
        ]
        axC.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=6.5,
            framealpha=0.95,
            edgecolor="#999999",
            handlelength=0.9,
            handleheight=0.8,
            borderpad=0.5,
            labelspacing=0.3
        )

        # =====================================================
        # ARROWS BETWEEN PANELS
        # =====================================================
        y_arr = self._GS_BOTTOM + (self._GS_TOP - self._GS_BOTTOM) * 0.80
        y_lbl = y_arr + 0.035

        def draw_arrow(x0_panel, x1_panel, label_text):
            x0 = x0_panel + 0.015
            x1 = x1_panel - 0.015
            xm = (x0 + x1) / 2

            fig.patches.append(
                mpatches.FancyArrowPatch(
                    posA=(x0, y_arr),
                    posB=(x1, y_arr),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    lw=1.6,
                    color="#555555",
                    transform=fig.transFigure,
                    zorder=10,
                    clip_on=False
                )
            )
            fig.text(
                xm,
                y_lbl,
                label_text,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#444444",
                style="italic"
            )

        draw_arrow(bounds[0][1], bounds[1][0], "Emergent descriptors + UMAP")
        draw_arrow(bounds[1][1], bounds[2][0], "Realized descriptors + UMAP")

        # =====================================================
        # FIGURE TITLE
        # =====================================================
        fig.suptitle(
            "Multi-Space Representation of Groove: From Parametric Control to Realized Structure",
            fontsize=11.5,
            weight="bold",
            y=0.97
        )

        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()