"""
analysis/viz/generative_validation.py
======================================
Figure de validation du modèle génératif — style publication mémoire.

Quatre panneaux :
    A — Matrice de corrélations de Pearson (paramètres génératifs → descripteurs)
        Axes X : D, I, V, S, E, P  (descripteurs émergents, sans indice)
        Axes Y : S_mv, D_mv, E_mv, P_mv  (paramètres génératifs, suffixe _mv)

    B — Stochasticité intra-condition (violin plots, CV par descripteur)
        Descripteurs émergents : D, I, V, S, E

    C — Couverture de l'espace de génération (heatmap S_mv × D_mv)

    D — Multicolinéarité (barres VIF horizontales)
        Mélange DÉLIBÉRÉ des deux espaces pour détecter la colinéarité
        ENTRE paramètres génératifs ET descripteurs émergents.
        Ce mélange est justifié : si E_mv et E sont colinéaires (r~1),
        on ne peut pas les utiliser dans le même modèle de régression.

Notation :
    Paramètres génératifs (manipulés) : S_mv, D_mv, E_mv, P_mv
    Descripteurs émergents (réalisés)  : D, I, V, S, E, P
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

BLUE   = "#2563EB"
BLUE_L = "#DBEAFE"
RED    = "#DC2626"
RED_L  = "#FEE2E2"
AMBER  = "#D97706"
AMBER_L= "#FEF3C7"
GREEN  = "#16A34A"
GREEN_L= "#DCFCE7"
GRAY   = "#6B7280"
GRAY_L = "#F3F4F6"

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
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
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
    ax.text(-0.08, 1.08, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="top", ha="left", color="#111827")


class GenerativeValidation:

    def plot(self, df: pd.DataFrame, path, verbose: bool = False) -> None:
        plt.rcParams.update(RC)

        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(
            2, 2,
            figure=fig,
            hspace=0.52, wspace=0.38,
            left=0.09, right=0.96,
            top=0.93, bottom=0.09,
        )

        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 0])
        axD = fig.add_subplot(gs[1, 1])

        self._panel_A_corr(axA, df, verbose)
        self._panel_B_stochasticity(axB, df, verbose)
        self._panel_C_coverage(axC, df, verbose)
        self._panel_D_vif(axD, df, verbose)

        fig.suptitle(
            "Validation du modèle génératif",
            fontsize=11, fontweight="semibold",
            color="#111827", y=0.97,
        )

        plt.savefig(path)
        plt.close()
        if verbose:
            print(f"  [fig] {path}")

    # ── Panel A : Corrélations paramètres génératifs → descripteurs émergents

    def _panel_A_corr(self, ax, df, verbose):
        # Paramètres génératifs en lignes
        params_candidates = ["S_mv", "D_mv", "E_mv", "P_mv"]
        params = [p for p in params_candidates if p in df.columns]

        # Descripteurs émergents en colonnes (sans indice)
        desc_candidates = ["D", "I", "V", "S", "E", "P"]
        descriptors = [d for d in desc_candidates if d in df.columns]

        if not params or not descriptors:
            ax.text(0.5, 0.5, "Données insuffisantes",
                    ha="center", va="center", transform=ax.transAxes)
            _add_panel_label(ax, "A")
            ax.set_title("Paramètres génératifs → descripteurs émergents")
            return

        corr = np.zeros((len(params), len(descriptors)))
        for i, p in enumerate(params):
            for j, d in enumerate(descriptors):
                try:
                    r, _ = pearsonr(df[p], df[d])
                    corr[i, j] = r
                except Exception:
                    corr[i, j] = np.nan

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        for i in range(len(params)):
            for j in range(len(descriptors)):
                v = corr[i, j]
                color = "white" if abs(v) > 0.45 else "#374151"
                ax.text(j, i,
                        f"{v:.2f}" if not np.isnan(v) else "—",
                        ha="center", va="center",
                        fontsize=7.5, fontweight="medium", color=color)

        desc_tex = {
            "D": "$D$", "I": "$I$", "V": "$V$",
            "S": "$S$", "E": "$E$", "P": "$P$",
        }
        param_tex = {
            "S_mv": "$S_{mv}$", "D_mv": "$D_{mv}$",
            "E_mv": "$E_{mv}$", "P_mv": "$P_{mv}$",
        }

        ax.set_xticks(range(len(descriptors)))
        ax.set_xticklabels([desc_tex.get(d, d) for d in descriptors], fontsize=8)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([param_tex.get(p, p) for p in params], fontsize=8)
        ax.tick_params(length=0)

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.9)
        cbar.set_label("Pearson $r$", fontsize=7.5, color="#6B7280")
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.outline.set_linewidth(0.4)

        _add_panel_label(ax, "A")
        ax.set_title("Paramètres génératifs → descripteurs émergents")

    # ── Panel B : Stochasticité intra-condition

    def _panel_B_stochasticity(self, ax, df, verbose):
        # Descripteurs émergents uniquement (sans indice)
        descriptors = ["D", "I", "V", "S", "E"]
        available   = [d for d in descriptors if d in df.columns]

        if not available:
            ax.text(0.5, 0.5, "Descripteurs absents",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color=GRAY)
            _add_panel_label(ax, "B")
            ax.set_title("Stochasticité intra-condition")
            return

        desc_tex = {"D": "$D$", "I": "$I$", "V": "$V$",
                    "S": "$S$", "E": "$E$"}

        # Grouper par paramètres génératifs
        group_cols = [c for c in ["S_mv", "D_mv", "E_mv"] if c in df.columns]
        cv_data = {d: [] for d in available}
        n_groups_with_repeats = 0

        if group_cols:
            for _, group in df.groupby(group_cols):
                if len(group) < 2:
                    continue
                n_groups_with_repeats += 1
                for d in available:
                    mu    = group[d].mean()
                    sigma = group[d].std(ddof=1)
                    if abs(mu) > 1e-10:
                        cv_data[d].append(float(sigma / abs(mu)))

        if n_groups_with_repeats == 0:
            ax.text(
                0.5, 0.55,
                "Stochasticité non estimable\n(Phase 3 : 1 réalisation/condition)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8.5, color=GRAY, style="italic",
                multialignment="center",
            )
            ax.text(
                0.5, 0.35,
                "Phases 1 & 2 : 4–5 répétitions/condition\n"
                "(données non incluses dans ce run)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=7.5, color="#9CA3AF",
                multialignment="center",
            )
            _add_panel_label(ax, "B")
            ax.set_title("Stochasticité intra-condition")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return

        plot_data = [cv_data[d] for d in available]
        positions = np.arange(1, len(available) + 1)

        valid_idx = [i for i, x in enumerate(plot_data) if len(x) > 1]
        if valid_idx:
            vp = ax.violinplot(
                [plot_data[i] for i in valid_idx],
                positions=[positions[i] for i in valid_idx],
                showmedians=True, showextrema=False, widths=0.55,
            )
            for body in vp["bodies"]:
                body.set_facecolor(BLUE_L)
                body.set_edgecolor(BLUE)
                body.set_alpha(0.85)
                body.set_linewidth(0.7)
            vp["cmedians"].set_color(BLUE)
            vp["cmedians"].set_linewidth(1.5)

        rng = np.random.default_rng(42)
        for i, vals in enumerate(plot_data):
            if not vals:
                continue
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(positions[i] + jitter, vals,
                       s=6, color=BLUE, alpha=0.35, linewidths=0)

        ax.set_xticks(positions)
        ax.set_xticklabels([desc_tex.get(d, d) for d in available], fontsize=8)
        ax.set_ylabel("Coefficient de variation", labelpad=6)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, integer=False))
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        ax.set_xlim(0.3, len(available) + 0.7)

        ax.text(0.98, 0.97,
                f"{n_groups_with_repeats} conditions avec répétitions",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#9CA3AF")

        _add_panel_label(ax, "B")
        ax.set_title("Stochasticité intra-condition")

    # ── Panel C : Couverture

    def _panel_C_coverage(self, ax, df, verbose):
        s_vals = sorted(df["S_mv"].unique())
        d_vals = sorted(df["D_mv"].unique())

        coverage = np.zeros((len(d_vals), len(s_vals)))
        for i, d in enumerate(d_vals):
            for j, s in enumerate(s_vals):
                coverage[i, j] = ((df["S_mv"] == s) & (df["D_mv"] == d)).sum()

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "coverage", ["#FEF3C7", "#D97706", "#92400E"], N=256
        )

        im = ax.imshow(coverage, cmap=cmap, aspect="auto", origin="lower")

        for i in range(len(d_vals)):
            for j in range(len(s_vals)):
                v = int(coverage[i, j])
                color = "white" if v >= coverage.max() * 0.5 else "#374151"
                ax.text(j, i, str(v),
                        ha="center", va="center",
                        fontsize=9, fontweight="medium", color=color)

        ax.set_xticks(range(len(s_vals)))
        ax.set_xticklabels([f"$S_{{mv}}={s}$" for s in s_vals], fontsize=8)
        ax.set_yticks(range(len(d_vals)))
        ax.set_yticklabels([f"$D_{{mv}}={d}$" for d in d_vals], fontsize=8)
        ax.set_xlabel("$S_{mv}$", labelpad=6)
        ax.set_ylabel("$D_{mv}$", labelpad=6)
        ax.tick_params(length=0)

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.9)
        cbar.set_label("Nombre de stimuli", fontsize=7.5, color="#6B7280")
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.outline.set_linewidth(0.4)

        max_cell = int(coverage.max())
        min_cell = int(coverage[coverage > 0].min())
        ax.text(0.98, 0.02,
                f"Min={min_cell}  Max={max_cell}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#9CA3AF")

        _add_panel_label(ax, "C")
        ax.set_title("Couverture de l'espace de design")

    # ── Panel D : VIF (mélange délibéré des deux espaces)

    def _panel_D_vif(self, ax, df, verbose):
        # Mélange DÉLIBÉRÉ : paramètres génératifs + descripteurs émergents.
        # Objectif : détecter la colinéarité ENTRE les deux espaces.
        # Si E_mv et E ont un VIF > 10, ils ne peuvent pas coexister
        # dans le même modèle de régression.
        # L'ordre : génératifs d'abord, émergents ensuite.
        candidates = ["S_mv", "D_mv", "E_mv", "P_mv", "D", "I", "V", "S", "E", "P"]
        available  = [p for p in candidates if p in df.columns]

        if len(available) < 2:
            ax.text(0.5, 0.5, "Données insuffisantes",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color=GRAY)
            _add_panel_label(ax, "D")
            ax.set_title("Multicolinéarité (VIF)")
            return

        X = df[available].dropna().values
        vifs = []
        for i in range(len(available)):
            try:
                v = variance_inflation_factor(X, i)
            except Exception:
                v = np.nan
            vifs.append(v)

        vif_df = sorted(zip(available, vifs),
                        key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                        reverse=True)
        names  = [x[0] for x in vif_df]
        values = [x[1] for x in vif_df]

        tex_map = {
            "S_mv": "$S_{mv}$", "D_mv": "$D_{mv}$",
            "E_mv": "$E_{mv}$", "P_mv": "$P_{mv}$",
            "D": "$D$", "I": "$I$", "V": "$V$",
            "S": "$S$", "E": "$E$", "P": "$P$",
        }

        colors = []
        for v in values:
            if np.isnan(v) or v > 10:
                colors.append(RED)
            elif v > 5:
                colors.append(AMBER)
            else:
                colors.append(GREEN)

        y = np.arange(len(names))
        bars = ax.barh(y, values, color=colors, alpha=0.85,
                       height=0.55, edgecolor="none")

        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}",
                        va="center", fontsize=7.5, color="#374151")

        ax.axvline(5,  color=AMBER, linewidth=0.9, linestyle="--", alpha=0.7, zorder=0)
        ax.axvline(10, color=RED,   linewidth=0.9, linestyle="--", alpha=0.7, zorder=0)

        ax.set_yticks(y)
        ax.set_yticklabels([tex_map.get(n, n) for n in names], fontsize=8)
        ax.set_xlabel("Variance Inflation Factor", labelpad=6)
        ax.set_xlim(left=0)
        ax.grid(axis="x", linestyle=":", linewidth=0.5)

        legend_patches = [
            mpatches.Patch(color=GREEN, alpha=0.85, label="VIF ≤ 5"),
            mpatches.Patch(color=AMBER, alpha=0.85, label="VIF 5–10"),
            mpatches.Patch(color=RED,   alpha=0.85, label="VIF > 10"),
        ]
        ax.legend(handles=legend_patches, loc="lower right",
                  fontsize=7, framealpha=1.0)

        # Note explicative sur le mélange des deux espaces
        ax.text(0.98, 0.02,
                "Mix génératifs + émergents\n(test colinéarité inter-espaces)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color="#9CA3AF", style="italic")

        _add_panel_label(ax, "D")
        ax.set_title("Multicolinéarité (VIF)")