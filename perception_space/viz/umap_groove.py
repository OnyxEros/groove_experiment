def plot_umap_groove(embedding, groove):
    import numpy as np
    import matplotlib.pyplot as plt

    if embedding.shape[1] > 2:
        from sklearn.decomposition import PCA
        embedding = PCA(n_components=2).fit_transform(embedding)

    groove = np.asarray(groove)

    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=groove, s=12, alpha=0.8)
    plt.title("Groove in latent space")
    plt.colorbar()
    plt.show()