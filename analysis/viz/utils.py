from pathlib import Path
from datetime import datetime
from config import ANALYSIS_DIR

PLOTLY_TEMPLATE = "plotly_white"


def apply_3d_layout(fig, title=None):
    fig.update_layout(
        title=title,
        title_font_size=16,
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
            bgcolor="white"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def save_figure(fig, save_path=None, name="figure"):

    # Ensure directory exists
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Auto filename if not provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.html"
        save_path = ANALYSIS_DIR / filename
    else:
        save_path = Path(save_path)

    # Save figure
    fig.write_html(str(save_path))

    print(f"📊 Figure saved → {save_path}")