import umap
from sklearn.preprocessing import StandardScaler

def umap_3d(df):
    features = ["D", "V", "S_real", "I"]

    X = StandardScaler().fit_transform(df[features].values)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.05,
        metric="cosine",
        random_state=42
    )

    emb = reducer.fit_transform(X)

    df = df.copy()
    df["u1"] = emb[:, 0]
    df["u2"] = emb[:, 1]
    df["u3"] = emb[:, 2]

    return df, reducer
