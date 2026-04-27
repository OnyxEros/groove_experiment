import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor


class GenerativeValidation:
    """
    Four-panel validation figure for the generative model.

    Panel A: Correlation matrix between generative params and emergent descriptors
    Panel B: Intra-condition variance (violin plots)
    Panel C: Coverage heatmap (stimulus density in design space)
    Panel D: VIF bar chart (multicollinearity check)
    """

    def plot(self, df, path, verbose=False):

        # =====================================================
        # STYLE
        # =====================================================
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        })

        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(
            2, 2,
            figure=fig,
            hspace=0.35,
            wspace=0.30,
            left=0.08,
            right=0.95,
            top=0.93,
            bottom=0.07
        )

        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 0])
        axD = fig.add_subplot(gs[1, 1])

        if verbose:
            print("\n=== GENERATIVE VALIDATION ===")
            print("N stimuli:", len(df))
            print("Columns:", df.columns.tolist())

        # =====================================================
        # PANEL A: CORRELATION MATRIX
        # =====================================================
        params = ["S_mv", "D_mv", "E"]
        descriptors = ["D", "I", "V", "S_real", "E_real"]

        corr_data = []
        for p in params:
            row = []
            for d in descriptors:
                try:
                    r, _ = pearsonr(df[p], df[d])
                except Exception:
                    r = np.nan
                row.append(r)
            corr_data.append(row)

        corr_matrix = np.array(corr_data)
        if verbose:
            print("\n[Panel A] Correlation matrix (S_mv, D_mv, E → descriptors)")
            print(pd.DataFrame(
                corr_matrix,
                index=params,
                columns=descriptors
            ))

        im = axA.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axA.set_xticks(range(len(descriptors)))
        axA.set_xticklabels(descriptors, rotation=45, ha="right")
        axA.set_yticks(range(len(params)))
        axA.set_yticklabels(params)
        axA.set_title("A  Generative → Emergent Correlations", pad=8)

        for i in range(len(params)):
            for j in range(len(descriptors)):
                val = corr_matrix[i, j]
                axA.text(
                    j, i,
                    f"{val:.2f}" if not np.isnan(val) else "nan",
                    ha="center",
                    va="center",
                    color="white" if abs(val) > 0.5 else "black" if not np.isnan(val) else "black",
                    fontsize=7,
                    weight="bold"
                )

        cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # =====================================================
        # PANEL B: INTRA-CONDITION VARIANCE
        # =====================================================
        # Group by condition (S_mv, D_mv, E) and compute variance of realized descriptors
        variance_data = []

        for (s, d, e), group in df.groupby(["S_mv", "D_mv", "E"]):
            if len(group) > 1:
                for desc in ["D", "I", "V", "S_real", "E_real"]:
                    variance_data.append({
                        "descriptor": desc,
                        "variance": group[desc].var(),
                        "cv": group[desc].std() / (group[desc].mean() + 1e-10)
                    })

        var_df = pd.DataFrame(variance_data)

        if verbose:
            print("\n[Panel B] Intra-condition variability (CV summary)")
            if len(var_df) > 0:
                print(var_df.groupby("descriptor")["cv"].describe())
            else:
                print("No repeated conditions → no variance computed")

        if len(var_df) > 0:
            sns.violinplot(
                data=var_df,
                x="descriptor",
                y="cv",
                ax=axB,
                color="#7fc97f",
                inner="quartile"
            )
            axB.set_xlabel("Descriptor")
            axB.set_ylabel("Coefficient of Variation")
            axB.set_title("B  Intra-Condition Stochasticity", pad=8)
            axB.tick_params(axis='x', rotation=45)
            axB.grid(alpha=0.2, axis='y', linestyle=":")
        else:
            axB.text(
                0.5, 0.5,
                "No repeated conditions\n(all repeats = 0)",
                ha="center",
                va="center",
                transform=axB.transAxes,
                fontsize=9,
                color="#999999"
            )
            axB.set_title("B  Intra-Condition Stochasticity", pad=8)

        # =====================================================
        # PANEL C: COVERAGE HEATMAP
        # =====================================================
        s_vals = sorted(df["S_mv"].unique())
        d_vals = sorted(df["D_mv"].unique())
        e_vals = sorted(df["E"].unique())

        if len(e_vals) <= 3:
            # Small number of E values — show all as separate heatmaps is overkill
            coverage = np.zeros((len(d_vals), len(s_vals)))

            for i, d in enumerate(d_vals):
                for j, s in enumerate(s_vals):
                    coverage[i, j] = len(df[(df["S_mv"] == s) & (df["D_mv"] == d)])

            if verbose:
                print("\n[Panel C] Design space coverage")
                print("S values:", s_vals)
                print("D values:", d_vals)
                print("E values:", e_vals)
                print("Coverage matrix:\n", coverage)

            im = axC.imshow(coverage, cmap="YlOrRd", aspect="auto", origin="lower")
            axC.set_xticks(range(len(s_vals)))
            axC.set_xticklabels([f"{int(s)}" for s in s_vals])
            axC.set_yticks(range(len(d_vals)))
            axC.set_yticklabels([f"{int(d)}" for d in d_vals])
            axC.set_xlabel("$S_{\\mathrm{mv}}$")
            axC.set_ylabel("$D_{\\mathrm{mv}}$")
            axC.set_title("C  Design Space Coverage (all E)", pad=8)

            for i in range(len(d_vals)):
                for j in range(len(s_vals)):
                    val = coverage[i, j]
                    axC.text(
                        j, i,
                        f"{int(val)}",
                        ha="center",
                        va="center",
                        color="white" if val > coverage.max() / 2 else "black",
                        fontsize=7,
                        weight="bold"
                    )

            cbar = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
            cbar.set_label("Stimulus count", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        else:
            axC.text(
                0.5, 0.5,
                f"Design space:\n{len(s_vals)} × {len(d_vals)} × {len(e_vals)}\n= {len(df)} stimuli",
                ha="center",
                va="center",
                transform=axC.transAxes,
                fontsize=9
            )
            axC.set_title("C  Design Space Coverage", pad=8)

        # =====================================================
        # PANEL D: VIF
        # =====================================================
        predictors = ["S_mv", "D", "I", "E", "V"]
        available_preds = [p for p in predictors if p in df.columns]

        if len(available_preds) >= 2:
            X = df[available_preds].values

            vif_data = []
            for i, pred in enumerate(available_preds):
                try:
                    vif = variance_inflation_factor(X, i)
                except Exception:
                    vif = np.nan
                vif_data.append({"predictor": pred, "VIF": vif})

            vif_df = pd.DataFrame(vif_data)

            if verbose:
                print("\n[Panel D] VIF diagnostics")
                print(vif_df.sort_values("VIF", ascending=False))

            colors = [
                "#d62728" if v > 10 else "#2ca02c" if v < 5 else "#ff7f0e"
                for v in vif_df["VIF"]
            ]

            axD.barh(vif_df["predictor"], vif_df["VIF"], color=colors, alpha=0.8)
            axD.axvline(5, linestyle="--", linewidth=1.5, alpha=0.7)
            axD.axvline(10, linestyle="--", linewidth=1.5, alpha=0.7)

            axD.set_xlabel("Variance Inflation Factor")
            axD.set_title("D  Multicollinearity Check", pad=8)
            axD.grid(alpha=0.2, axis='x', linestyle=":")

            axD.text(
                0.98, 0.02,
                "VIF > 10 → severe multicollinearity\nVIF > 5 → moderate",
                transform=axD.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                color="#555555",
                style="italic"
            )

        else:
            axD.text(
                0.5, 0.5,
                "Insufficient predictors\nfor VIF calculation",
                ha="center",
                va="center",
                transform=axD.transAxes,
                fontsize=9,
                color="#999999"
            )
            axD.set_title("D  Multicollinearity Check", pad=8)

        # =====================================================
        # TITLE
        # =====================================================
        fig.suptitle(
            "Generative Model Validation: Parameter Dependencies and Design Space Coverage",
            fontsize=11,
            weight="bold",
            y=0.98
        )

        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()