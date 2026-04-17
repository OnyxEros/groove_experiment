import umap
from sklearn.preprocessing import StandardScaler


FEATURES = ["D", "V", "S_real"]


def compute_umap_groove(df):
    X = df[FEATURES].values

    X = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.05,
        metric="cosine",
        random_state=42
    )

    emb = reducer.fit_transform(X)

    return emb, reducer