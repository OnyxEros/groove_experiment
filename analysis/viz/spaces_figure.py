import matplotlib.pyplot as plt


class SpacesFigure:

    def plot(
        self,
        df,
        umap_gen,
        umap_emergent,
        umap_audio,
        labels,
        path
    ):

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # =====================================================
        # (A) GENERATIVE SPACE
        # =====================================================

        ax = axes[0]

        if all(col in df.columns for col in ["S_mv", "D_mv", "E"]):

            sc = ax.scatter(
                df["S_mv"],
                df["D_mv"],
                c=df["E"],
                cmap="viridis",
                s=10
            )

            ax.set_title("(A) Generative space")
            ax.set_xlabel("S_mv")
            ax.set_ylabel("D_mv")
            plt.colorbar(sc, ax=ax)

        else:
            ax.text(0.5, 0.5, "Missing generative features", ha="center")
            ax.set_title("(A) Generative space")

        # =====================================================
        # (B) EMERGENT SPACE
        # =====================================================

        ax = axes[1]

        if umap_emergent is not None:

            sc = ax.scatter(
                umap_emergent[:, 0],
                umap_emergent[:, 1],
                c=df.get("S_mv", None),
                cmap="viridis",
                s=10
            )

            ax.set_title("(B) Emergent space")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")

        else:
            ax.text(0.5, 0.5, "No emergent projection", ha="center")
            ax.set_title("(B) Emergent space")

        # =====================================================
        # (C) AUDIO SPACE
        # =====================================================

        ax = axes[2]

        if umap_audio is not None and labels is not None:

            ax.scatter(
                umap_audio[:, 0],
                umap_audio[:, 1],
                c=labels,
                cmap="tab10",
                s=10
            )

            ax.set_title("(C) Audio/perceptual space")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")

        else:
            ax.text(0.5, 0.5, "Missing audio embedding", ha="center")
            ax.set_title("(C) Audio/perceptual space")

        # =====================================================
        # STYLE GLOBAL
        # =====================================================

        for ax in axes:
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=8)

        ax.set_aspect("equal", adjustable="box")

        plt.suptitle(
            "Groove: Generative → Emergent → Perceptual Spaces",
            fontsize=14
        )

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()