import plotly.express as px

PLOTLY_TEMPLATE = "plotly_white"

def plot_audio_space(emb, paths, save_path=None):

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

    fig.update_layout(
        title_font_size=16,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            bgcolor="white"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_groove_space(df, save_path=None):

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

    fig.update_layout(
        title_font_size=16,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            bgcolor="white"
        ),
        legend_title_text="Phase",
        margin=dict(l=0, r=0, t=40, b=0)
    )

    if save_path:
        fig.write_html(save_path)

    return fig

def animate_groove(df, save_path=None):

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

    fig.update_layout(
        scene=dict(
            xaxis_title="Density (D)",
            yaxis_title="Variance (V)",
            zaxis_title="Syncopation (S_real)",
            bgcolor="white"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    if save_path:
        fig.write_html(save_path)

    return fig