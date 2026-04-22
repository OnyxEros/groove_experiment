from tqdm import tqdm

from analysis.embeddings.manager import EmbeddingManager
from analysis.embeddings.clustering import cluster_latent_space
from analysis.io.run_manager import RunManager
from config import get_run_dir

from analysis.viz.audio import plot_audio_space
from analysis.viz.groove import plot_groove_space


# =========================================================
# FULL ANALYSIS PIPELINE
# =========================================================

def run_full_analysis(
    X_audio=None,
    X_groove=None,
    mode="audio",
    steps=None,
    save=True,
    paths=None
):

    # =====================================================
    # DEFAULT STEPS
    # =====================================================
    if steps is None:
        if mode in ["audio", "groove"]:
            steps = ["embedding", "clustering", "viz"]
        else:
            raise ValueError("mode must be: audio | groove")

    # =====================================================
    # INIT
    # =====================================================
    manager = EmbeddingManager()
    run_dir = RunManager(get_run_dir())

    Z = None
    labels = None

    # =====================================================
    # STEP PROGRESS BAR
    # =====================================================
    step_bar = tqdm(
        total=len(steps),
        desc=f"🧠 Analysis ({mode})",
        unit="step",
        dynamic_ncols=True
    )

    try:

        # =====================================================
        # AUDIO MODE
        # =====================================================
        if mode == "audio":

            if X_audio is None:
                raise ValueError(
                    "X_audio is required. "
                    "Load dataset before running analysis."
                )

            # -------------------------------------------------
            # EMBEDDING
            # -------------------------------------------------
            if "embedding" in steps:
                step_bar.set_description("🔢 embedding (audio)")

                Z = manager.fit("audio", X_audio)

                step_bar.update(1)

            # -------------------------------------------------
            # CLUSTERING
            # -------------------------------------------------
            if "clustering" in steps:
                step_bar.set_description("📦 clustering (audio)")

                if Z is None:
                    raise ValueError("Clustering requires embedding step")

                labels, _ = cluster_latent_space(Z)

                step_bar.update(1)

            # -------------------------------------------------
            # VIZ
            # -------------------------------------------------
            if save and "viz" in steps:
                step_bar.set_description("📊 viz (audio)")

                if Z is None:
                    raise ValueError("Viz requires embedding step")

                run_dir.save_embedding("audio", Z)
                run_dir.save_clusters("audio", labels)
                run_dir.save_config({"mode": "audio", "steps": steps})

                plot_audio_space(
                    Z,
                    paths=paths,
                    save_path=run_dir.path("audio_space.html")
                )

                step_bar.update(1)

            return Z, labels, manager, run_dir

        # =====================================================
        # GROOVE MODE
        # =====================================================
        if mode == "groove":

            if X_groove is None:
                raise ValueError(
                    "X_groove is required. "
                    "Load dataset before running analysis."
                )

            # -------------------------------------------------
            # EMBEDDING
            # -------------------------------------------------
            if "embedding" in steps:
                step_bar.set_description("🔢 embedding (groove)")

                Z = manager.fit("groove", X_groove)

                step_bar.update(1)

            # -------------------------------------------------
            # CLUSTERING
            # -------------------------------------------------
            if "clustering" in steps:
                step_bar.set_description("📦 clustering (groove)")

                if Z is None:
                    raise ValueError("Clustering requires embedding step")

                labels, _ = cluster_latent_space(Z)

                step_bar.update(1)

            # -------------------------------------------------
            # VIZ
            # -------------------------------------------------
            if save and "viz" in steps:
                step_bar.set_description("📊 viz (groove)")

                if Z is None:
                    raise ValueError("Viz requires embedding step")

                run_dir.save_embedding("groove", Z)
                run_dir.save_clusters("groove", labels)
                run_dir.save_config({"mode": "groove", "steps": steps})

                plot_groove_space(
                    Z,
                    save_path=run_dir.path("groove_space.html")
                )

                step_bar.update(1)

            return Z, labels, manager, run_dir

        # =====================================================
        # ERROR
        # =====================================================
        raise ValueError("mode must be: audio | groove")

    finally:
        step_bar.close()