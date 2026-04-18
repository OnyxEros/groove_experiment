import pandas as pd
import numpy as np

from analysis.dataset.audio_dataset import build_audio_embeddings
from analysis.pipelines.audio_space import run_audio_space
from analysis.pipelines.groove_space import run_groove_space
from analysis.pipelines.interpret_clusters import run_interpretation


# =========================================================
# CONFIG
# =========================================================

MP3_DIR = "data/mp3"


# =========================================================
# MAIN TEST
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. AUDIO
    # -----------------------------------------------------
    print("\n🎧 Loading audio...")

    X_audio, stim_ids, paths = build_audio_embeddings(MP3_DIR)

    print("Audio shape:", X_audio.shape)


    # -----------------------------------------------------
    # 2. GROOVE (TEMP FAKE)
    # -----------------------------------------------------
    print("\n🥁 Creating fake groove data...")

    X_groove = np.column_stack([
        X_audio[:, 0] * 0.1,
        X_audio[:, 1] * 0.1,
        X_audio[:, 2] * 0.1
    ])

    print("Groove shape:", X_groove.shape)


    # -----------------------------------------------------
    # 3. AUDIO SPACE (UMAP)
    # -----------------------------------------------------
    print("\n📊 Running audio space...")

    Z_audio, labels_audio = run_audio_space(X_audio)

    print("Audio embedding:", Z_audio.shape)


    # -----------------------------------------------------
    # 4. GROOVE SPACE (UMAP)
    # -----------------------------------------------------
    print("\n📊 Running groove space...")

    Z_groove, labels_groove = run_groove_space(X_groove)

    print("Groove embedding:", Z_groove.shape)


    # -----------------------------------------------------
    # 5. CLUSTER INTERPRETATION
    # -----------------------------------------------------
    print("\n🧠 Running cluster interpretation...")

    import pandas as pd

    df_groove = pd.DataFrame(X_groove, columns=["D", "V", "S_real"])

    profiles = run_interpretation(df_groove, labels_groove)

    print("\n🧠 Cluster interpretation:")
    print(profiles)


    print("\n✅ TEST COMPLETE")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()