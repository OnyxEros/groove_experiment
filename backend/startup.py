from config import ensure_data_dirs, MP3_DIR


def check_environment():
    ensure_data_dirs()

    files = list(MP3_DIR.rglob("*.mp3"))

    print("🎧 Environment ready")
    print(f"   MP3 directory: {MP3_DIR.resolve()}")
    print(f"   MP3 files detected: {len(files)}")

    if len(files) == 0:
        print("⚠️ Warning: no MP3 files found in dataset")