import matplotlib.pyplot as plt


class UMAPViz:

    def plot_2d(self, embedding, labels, path):

        plt.figure()

        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab10",
            s=10
        )

        plt.title("UMAP 2D - Groove Space")

        plt.savefig(path)
        plt.close()
