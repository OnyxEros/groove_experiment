import matplotlib.pyplot as plt
import numpy as np


class HeatmapViz:

    def plot(self, matrix, path):

        plt.figure()

        plt.imshow(matrix, aspect="auto")
        plt.colorbar()

        plt.title("Feature Heatmap")

        plt.savefig(path)
        plt.close()
