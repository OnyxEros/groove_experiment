import pandas as pd
import plotly.express as px

from .utils import PLOTLY_TEMPLATE, apply_3d_layout, save_figure


def plot_groove_space(df: pd.DataFrame, save_path=None):
    """
    Plot 3D UMAP embedding of groove features.

    Args:
        df (pd.DataFrame): Must contain columns u1, u2, u3, phase
        save_path (str or Path, optional): Custom save path

    Returns:
        plotly.graph_objects.Figure
    """

    required_cols = {"u1", "u2", "u3", "phase"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    if df.empty:
        raise ValueError("Empty dataframe provided")

    fig = px.scatter_3d(
        df,
        x="u1",
        y="u2",
        z="u3",
        color="phase",
        title="Groove space (UMAP)",
        template=PLOTLY_TEMPLATE
    )

    fig.update_traces(
        marker=dict(size=4, opacity=0.8)
    )

    fig = apply_3d_layout(fig)

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        legend_title_text="Phase",
    )

    save_figure(fig, save_path, name="groove_space")

    return fig


def animate_groove(df: pd.DataFrame, save_path=None):
    """
    Animate groove feature evolution.

    Args:
        df (pd.DataFrame): Must contain D, V, S_real, phase
        save_path (str or Path, optional): Custom save path

    Returns:
        plotly.graph_objects.Figure
    """

    required_cols = {"D", "V", "S_real", "phase"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    if df.empty:
        raise ValueError("Empty dataframe provided")

    fig = px.scatter_3d(
        df,
        x="D",
        y="V",
        z="S_real",
        color="phase",
        animation_frame="phase",
        opacity=0.6,
        title="Groove evolution",
        template=PLOTLY_TEMPLATE
    )

    fig.update_traces(
        marker=dict(size=3)
    )

    fig = apply_3d_layout(fig)

    fig.update_layout(
        scene=dict(
            xaxis_title="Density (D)",
            yaxis_title="Variance (V)",
            zaxis_title="Syncopation (S_real)",
        )
    )

    save_figure(fig, save_path, name="groove_animation")

    return fig