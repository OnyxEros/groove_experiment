import matplotlib.pyplot as plt
import numpy as np


class ThesisFigure:

    def plot(
        self,
        df,
        umap_audio,
        umap_gen,
        labels,
        path
    ):

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # =====================================================
        # (A) AUDIO SPACE
        # =====================================================

        x, y = umap_audio[:, 0], umap_audio[:, 1]

        ax = axes[0, 0]
        sc = ax.scatter(x, y, c=df["S_mv"], cmap="viridis", s=12)
        ax.set_title("(A) Audio space (S_mv)")
        plt.colorbar(sc, ax=ax)

        # =====================================================
        # (B) CLUSTERS
        # =====================================================

        ax = axes[0, 1]
        ax.scatter(x, y, c=labels, cmap="tab10", s=12)
        ax.set_title("(B) Emergent clusters")

        # =====================================================
        # (C) INTERACTION NORMALISÉE
        # =====================================================

        ax = axes[1, 0]

        interaction = df["S_mv"] * df["E"]
        interaction = (interaction - interaction.mean()) / interaction.std()

        sc = ax.scatter(x, y, c=interaction, cmap="coolwarm", s=12)
        ax.set_title("(C) Coupled interaction (normalized)")
        plt.colorbar(sc, ax=ax)

        # =====================================================
        # (D) GENERATIVE SPACE
        # =====================================================

        ax = axes[1, 1]

        if umap_gen is not None:
            ax.scatter(
                umap_gen[:, 0],
                umap_gen[:, 1],
                c=labels,
                cmap="tab10",
                s=12
            )
            ax.set_title("(D) Generative space")
        else:
            ax.text(0.5, 0.5, "No generative space", ha="center")

        # =====================================================
        # STYLE GLOBAL
        # =====================================================

        for ax in axes.ravel():
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=8)

        plt.suptitle("Groove Space Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()