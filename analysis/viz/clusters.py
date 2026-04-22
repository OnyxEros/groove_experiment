import matplotlib.pyplot as plt


class ClusterViz:

    def plot(self, embedding, labels, path):

        plt.figure()

        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab10",
            s=10
        )

        plt.title("Cluster Distribution")

        plt.savefig(path)
        plt.close()
