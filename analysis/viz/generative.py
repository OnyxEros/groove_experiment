import matplotlib.pyplot as plt


class GenerativeViz:

    # =====================================================
    # (A) PARAMETRIC GENERATIVE SPACE (S, D, E)
    # =====================================================

    def plot_3d(self, df, labels, path):

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = df["S_mv"]
        y = df["D_mv"]
        z = df["E"]

        sc = ax.scatter(
            x, y, z,
            c=labels,
            cmap="tab10",
            s=10
        )

        ax.set_xlabel("S_mv (syncopation)")
        ax.set_ylabel("D_mv (density)")
        ax.set_zlabel("E (micro-timing)")

        ax.set_title("Generative space (S, D, E)")

        plt.colorbar(sc)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    # =====================================================
    # (B) LATENT SPACE (UMAP AUDIO / EMBEDDING)
    # =====================================================

    def plot_latent_3d(self, embedding_3d, labels, path):

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            embedding_3d[:, 0],
            embedding_3d[:, 1],
            embedding_3d[:, 2],
            c=labels,
            cmap="tab10",
            s=10
        )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")

        ax.set_title("Latent space (UMAP audio embedding)")

        plt.colorbar(sc)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()