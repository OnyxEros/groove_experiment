import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_local_geometry(X, y, k=15):

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)

    _, indices = nn.kneighbors(X)

    local_mean = np.array([y[idx].mean() for idx in indices])
    local_std  = np.array([y[idx].std()  for idx in indices])

    local_slope = np.array([
        np.polyfit(np.arange(k), y[idx], 1)[0]
        for idx in indices
    ])

    # Cohérence locale : corrélation entre le rating y et la
    # distance au centroïd local dans l'espace X.
    # Mesure si les voisins proches ont des ratings similaires.
    local_coherence = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        if len(idx) < 3:
            local_coherence[i] = 0.0
            continue
        centroid  = X[idx].mean(axis=0)
        distances = np.linalg.norm(X[idx] - centroid, axis=1)
        y_local   = y[idx]
        if distances.std() < 1e-10 or y_local.std() < 1e-10:
            local_coherence[i] = 0.0
        else:
            local_coherence[i] = np.corrcoef(distances, y_local)[0, 1]

    return {
        "local_mean":      local_mean,
        "local_std":       local_std,
        "local_slope":     local_slope,
        "local_coherence": local_coherence,
    }