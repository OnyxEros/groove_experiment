import matplotlib.pyplot as plt


class UMAPViz:

    def plot(self, embedding, labels, path, color_by=None, title="UMAP projection"):

        plt.figure(figsize=(8, 6))

        if color_by is not None:
            sc = plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=color_by,
                cmap="viridis",
                s=12
            )
            plt.colorbar(sc)
        else:
            plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="tab10",
                s=12
            )

        plt.title(title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(alpha=0.2)

        plt.savefig(path, dpi=300)
        plt.close()