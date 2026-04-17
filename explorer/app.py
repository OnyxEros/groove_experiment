import streamlit as st
import plotly.express as px
import pandas as pd

from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

from analysis.explorer.builder import build_explorer_dataset


# =========================================================
# SUPABASE CLIENT
# =========================================================
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_data(ttl=10)
def load_responses():
    res = supabase.table("responses").select("*").execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()


# =========================================================
# DATA
# =========================================================
df_audio = build_explorer_dataset()
df_resp = load_responses()


st.title("🧠 Groove Scientific Explorer (Live)")


# =========================================================
# SIDEBAR METRICS
# =========================================================
st.sidebar.header("📊 Dataset status")

st.sidebar.metric("Stimuli", len(df_audio))
st.sidebar.metric("Responses", len(df_resp))


if not df_resp.empty:
    st.sidebar.metric("Participants", df_resp["participant_id"].nunique())


# =========================================================
# RESPONSES ANALYSIS
# =========================================================
st.subheader("📈 Global results")

if not df_resp.empty:

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df_resp,
            x="groove",
            nbins=7,
            title="Groove distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df_resp,
            x="complexity",
            nbins=7,
            title="Complexity distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df_resp,
        x="groove",
        y="complexity",
        color="rt",
        hover_data=["stim_id", "participant_id"],
        title="Groove vs Complexity (colored by RT)"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No responses yet")


# =========================================================
# JOIN AUDIO + RESPONSES
# =========================================================
st.subheader("🎧 Stimuli insights")

if not df_resp.empty:

    merged = df_resp.merge(
        df_audio,
        left_on="stim_id",
        right_on="audio_file",
        how="left"
    )

    fig = px.scatter(
        merged,
        x="groove",
        y="BPM",
        color="complexity",
        hover_data=["stim_id", "participant_id"],
        title="Groove vs BPM"
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# ORIGINAL UMAP EXPLORER (UNCHANGED)
# =========================================================
st.subheader("🧠 Audio embedding space")

phase = st.selectbox(
    "Phase",
    ["all"] + sorted(df_audio["phase"].unique())
)

if phase != "all":
    df_view = df_audio[df_audio["phase"] == phase]
else:
    df_view = df_audio


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


# =========================================================
# AUDIO PLAYER
# =========================================================
st.subheader("🎧 Selected stimulus")

selected = st.selectbox(
    "Choose stimulus",
    df_view.index,
    format_func=lambda i: df_view.loc[i, "audio_file"]
)

audio_path = df_view.loc[selected, "path"]

st.audio(audio_path)

st.write(df_view.loc[selected, ["D", "V", "S_real", "I", "BPM", "phase"]])