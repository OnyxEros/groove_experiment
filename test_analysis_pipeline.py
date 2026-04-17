from analysis.dataset.audio_dataset import build_audio_embeddings
from analysis.dataset.audio_dataset import load_audio_paths

from analysis.pipelines.audio_space import run_audio_space
from analysis.pipelines.groove_space import run_groove_space
from analysis.embeddings.joint import compute_joint_embedding


import pandas as pd


# =========================================================
# CONFIG (change path ici)
# =========================================================

MP3_DIR = "data/mp3"


# =========================================================
# 1. AUDIO TEST
# =========================================================

print("\n🎧 Loading audio...")

X_audio, stim_ids, paths = build_audio_embeddings(MP3_DIR)

print("Audio shape:", X_audio.shape)


# =========================================================
# 2. GROOVE TEST (FAKE si pas encore pipeline complet)
# =========================================================

print("\n🥁 Creating fake groove data...")

df_groove = pd.DataFrame({
    "stim_id": stim_ids,
    "D": X_audio[:, 0] * 0.1,
    "V": X_audio[:, 1] * 0.1,
    "S_real": X_audio[:, 2] * 0.1,
})

print("Groove shape:", df_groove.shape)


# =========================================================
# 3. AUDIO SPACE
# =========================================================

print("\n📊 Running audio space...")

Z_audio, labels_audio = run_audio_space(X_audio)

print("Audio embedding:", Z_audio.shape)


# =========================================================
# 4. GROOVE SPACE
# =========================================================

print("\n📊 Running groove space...")

Z_groove, labels_groove = run_groove_space(df_groove)

print("Groove embedding:", Z_groove.shape)


# =========================================================
# 5. JOINT SPACE (STEP 2)
# =========================================================

print("\n🧠 Running joint embedding...")

from analysis.embeddings.manager import EmbeddingManager

manager = EmbeddingManager()

Z_joint = manager.fit_joint(X_audio, df_groove[["D", "V", 
"S_real"]].values)

print("Joint embedding:", Z_joint.shape)


# =========================================================
# 6. Cluster interpretations
# =========================================================

print("\n🧠 Running cluster interpretation...")

from analysis.pipelines.interpret_clusters import run_interpretation

profiles = run_interpretation(df_groove, labels_groove)

print("\n🧠 Cluster interpretation:")
print(profiles)


print("\n✅ TEST COMPLETE")
