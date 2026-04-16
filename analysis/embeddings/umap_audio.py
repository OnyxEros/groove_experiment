import umap

def umap_audio_3d(vectors):
    reducer = umap.UMAP(
        n_components=3,
        random_state=42,
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine"
    )

    emb = reducer.fit_transform(vectors)
    return emb, reducer
