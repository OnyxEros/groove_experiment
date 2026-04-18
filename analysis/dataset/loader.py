import os
from config import MP3_DIR

def load_audio_paths():
    paths = []

    for root, _, files in os.walk(MP3_DIR):
        for f in files:
            if f.endswith(".mp3"):
                paths.append(os.path.join(root, f))

    return sorted(paths)