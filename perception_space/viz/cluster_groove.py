import numpy as np
import matplotlib.pyplot as plt


def plot_cluster_groove(embedding, clusters, groove):

    unique = np.unique(clusters)

    means = [
        groove[clusters == c].mean()
        for c in unique
    ]

    plt.figure()
    plt.bar(unique, means)
    plt.title("Mean groove per cluster")
    plt.show()
