from config import ensure_data_dirs, MP3_DIR


def check_environment():
    ensure_data_dirs()

    if not MP3_DIR.exists():
        raise RuntimeError("MP3 directory missing")

    files = list(MP3_DIR.rglob("*.mp3"))

    if len(files) == 0:
        raise RuntimeError("No MP3 files found")

    print("🎧 Environment ready")
    print(f"   MP3 files detected: {len(files)}")