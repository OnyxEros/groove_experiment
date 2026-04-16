import streamlit as st
import plotly.express as px
from analysis.explorer.builder import build_explorer_dataset
from streamlit.components.v1 import html


df = build_explorer_dataset()

st.title("🧠 Groove Scientific Explorer")

# =========================
# FILTERS
# =========================
phase = st.selectbox("Phase", ["all"] + sorted(df["phase"].unique()))

if phase != "all":
    df_view = df[df["phase"] == phase]
else:
    df_view = df


# =========================
# 3D EMBEDDING
# =========================
fig = px.scatter_3d(
    df_view,
    x="x",
    y="y",
    z="z",
    color="phase",
    hover_data=["D", "V", "S_real", "I", "BPM"],
    title="Audio embedding space (UMAP)"
)

fig.update_traces(marker=dict(size=4))

st.plotly_chart(fig, use_container_width=True)


# =========================
# CLICKABLE AUDIO PLAYER
# =========================
st.subheader("🎧 Selected stimulus")

selected = st.selectbox(
    "Choose stimulus",
    df_view.index,
    format_func=lambda i: df_view.loc[i, "path"].split("/")[-1]
)

audio_path = df_view.loc[selected, "path"]

st.audio(audio_path)


# =========================
# FEATURE DISPLAY
# =========================
st.subheader("📊 Features")

st.write(df_view.loc[selected, ["D", "V", "S_real", "I", "BPM", "phase"]])
