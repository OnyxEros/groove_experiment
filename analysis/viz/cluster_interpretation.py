import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from math import pi


class ClusterInterpretation:
    """
    Cluster interpretation figure with radar charts and semantic labels.

    Shows normalized profiles of each cluster across rhythm descriptors,
    plus quantitative statistics and qualitative labels.
    """

    def plot(self, df, labels, path):

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
        })

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Compute cluster profiles
        descriptors = ["D", "I", "V", "S_real", "E_real"]
        available_desc = [d for d in descriptors if d in df.columns]

        if len(available_desc) == 0:
            available_desc = ["S_mv", "D_mv", "E"]

        cluster_profiles = []
        cluster_stats = []

        for k in unique_labels:
            mask = (labels == k)
            subset = df[mask]

            # Mean profile (normalized 0-1)
            profile = {}
            for desc in available_desc:
                vals = subset[desc].values
                min_val = df[desc].min()
                max_val = df[desc].max()
                profile[desc] = (vals.mean() - min_val) / (max_val - min_val + 1e-10)

            cluster_profiles.append(profile)

            stats = {
                "cluster": k,
                "size": len(subset),
                "pct": 100 * len(subset) / len(df),
            }

            if "phase" in df.columns:
                phase_counts = subset["phase"].value_counts()
                stats["dominant_phase"] = phase_counts.idxmax() if len(phase_counts) > 0 else None

            stats["label"] = self._generate_label(profile, available_desc)
            cluster_stats.append(stats)

        # =====================================================
        # FIGURE LAYOUT
        # =====================================================
        n_cols = min(3, n_clusters)
        n_rows = int(np.ceil(n_clusters / n_cols))

        fig = plt.figure(figsize=(14, 4 + 3 * n_rows))

        gs = gridspec.GridSpec(
            n_rows + 1, n_cols,
            figure=fig,
            height_ratios=[3] * n_rows + [1],
            hspace=0.4,
            wspace=0.3,
            left=0.08,
            right=0.95,
            top=0.92,
            bottom=0.08
        )

        # =====================================================
        # RADAR CHARTS
        # =====================================================
        categories = available_desc
        n_vars = len(categories)

        angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
        angles += angles[:1]

        cmap = plt.cm.get_cmap("tab10", n_clusters)

        for idx, k in enumerate(unique_labels):
            row = idx // n_cols
            col = idx % n_cols

            ax = fig.add_subplot(gs[row, col], projection="polar")

            profile = cluster_profiles[idx]
            values = [profile[cat] for cat in categories]
            values += values[:1]

            color = cmap(idx % 10)

            ax.plot(angles, values, 'o-', linewidth=2, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6, color="#666666")
            ax.grid(True, linestyle=":", alpha=0.5)

            stats = cluster_stats[idx]

            ax.set_title(
                f"Cluster {k}: {stats['label']}\n({stats['size']} stimuli, {stats['pct']:.1f}%)",
                fontsize=9,
                weight="bold",
                pad=15
            )

        # =====================================================
        # TABLE
        # =====================================================
        ax_table = fig.add_subplot(gs[n_rows, :])
        ax_table.axis("tight")
        ax_table.axis("off")

        table_data = []
        table_data.append(["Cluster", "Label", "Size", "%", "Dominant Phase"])

        for stats in cluster_stats:
            table_data.append([
                f"{stats['cluster']}",
                stats['label'],
                f"{stats['size']}",
                f"{stats['pct']:.1f}%",
                f"{stats.get('dominant_phase', 'N/A')}"
            ])

        table = ax_table.table(
            cellText=table_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.1, 0.4, 0.1, 0.1, 0.15]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        for i in range(len(table_data[0])):
            cell = table[(0, i)]
            cell.set_facecolor("#cccccc")
            cell.set_text_props(weight="bold")

        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor("#f0f0f0")

        # =====================================================
        # TITLE
        # =====================================================
        fig.suptitle(
            "Cluster Interpretation: Rhythm Descriptor Profiles and Semantic Labels",
            fontsize=12,
            weight="bold",
            y=0.98
        )

        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def _generate_label(self, profile, descriptors):

        labels = []

        if "D" in profile:
            if profile["D"] < 0.3:
                labels.append("sparse")
            elif profile["D"] > 0.7:
                labels.append("dense")

        if "S_real" in profile:
            if profile["S_real"] > 0.6:
                labels.append("syncopated")
            elif profile["S_real"] < 0.3:
                labels.append("on-beat")

        if "E_real" in profile:
            if profile["E_real"] > 0.6:
                labels.append("loose timing")
            elif profile["E_real"] < 0.3:
                labels.append("tight timing")

        if "V" in profile and profile["V"] > 0.6:
            labels.append("high variance")

        if "I" in profile and profile["I"] > 0.6:
            labels.append("imbalanced")

        return ", ".join(labels) if labels else "moderate"