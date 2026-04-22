import numpy as np
import plotly.express as px

from .utils import PLOTLY_TEMPLATE, apply_3d_layout, save_figure


def plot_audio_space(emb: np.ndarray, paths=None, save_path=None):
    """
    Plot 3D UMAP embedding of audio features.

    Args:
        emb (np.ndarray): Embedding array (N, >=3)
        paths (list[str], optional): Audio file paths for hover
        save_path (str or Path, optional): Custom save path

    Returns:
        plotly.graph_objects.Figure
    """

    if emb is None or len(emb) == 0:
        raise ValueError("Empty embedding provided")

    if emb.shape[1] < 3:
        raise ValueError("Embedding must have at least 3 dimensions")

    fig = px.scatter_3d(
        x=emb[:, 0],
        y=emb[:, 1],
        z=emb[:, 2],
        hover_name=paths,
        title="Audio embedding space (MFCC + UMAP)",
        template=PLOTLY_TEMPLATE
    )

    fig.update_traces(
        marker=dict(size=3, opacity=0.7),
        hovertemplate="%{hovertext}<extra></extra>"
    )

    fig = apply_3d_layout(fig)

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        )
    )

    save_figure(fig, save_path, name="audio_space")

    return fig