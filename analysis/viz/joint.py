import plotly.express as px


def plot_joint(Z, labels):
    fig = px.scatter_3d(
        x=Z[:, 0],
        y=Z[:, 1],
        z=Z[:, 2],
        color=labels,
        title="Joint Audio-Groove Latent Space"
    )

    fig.update_traces(marker=dict(size=3, opacity=0.75))

    return fig